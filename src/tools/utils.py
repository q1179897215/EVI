import os
import sys

sys.path.append("..")
import unittest
from abc import ABC, abstractmethod
from typing import Any

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.modules.domain_discriminator import DomainDiscriminator
from torchmetrics.classification import BinaryAUROC

import wandb

from .metrics import cal, cal_auc

# import seaborn as sns


def single_train(config, model_pl, litcallback, dataloaders, train=True, test=True, val=True):
    pl.seed_everything(config["global_seed"], workers=True)
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    if config["using_wandb"]:
        wandb_logger = WandbLogger(
            name=config["name"],
            project=config["project"],
            config=config,
            reinit=True,
            group=config["group_name"],
            log_model=config["log_model"],
            save_code=True,
            settings=wandb.Settings(code_dir="."),
        )
    else:
        wandb_logger = None

    model_checkpoint = (
        ModelCheckpoint(
            dirpath=config["check_point_path"],
            save_top_k=config["save_top_k"],
            mode=config["checkpoint_monitor_mode"],
            monitor=config["checkpoint_monitor_metric"],
        )
        if config["using_model_checkpoint"]
        else None
    )

    early_stop = (
        EarlyStopping(
            monitor=config["monitor_metric"],
            mode=config["mode"],
            patience=config["earlystop_epoch"],
            min_delta=config["min_delta"],
        )
        if config["using_early_stop"]
        else None
    )

    call_backs = [litcallback, early_stop, model_checkpoint]
    # filter the None
    call_backs = [call_back for call_back in call_backs if call_back is not None]

    trainer = pl.Trainer(
        max_epochs=config["total_epoch"],
        callbacks=call_backs,
        logger=wandb_logger,
        profiler=config["profiler"],
        log_every_n_steps=2,
        deterministic=True,
    )
    # torch.use_deterministic_algorithms(True, warn_only=True)

    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(aitm_pl, datamodule=ali_dm)
    # tuner.lr_find(aitm_pl, datamodule=ali_dm)
    if train == True and val == True and test == True:
        trainer.fit(
            model=model_pl,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["val"],
        )
        trainer.test(model=model_pl, dataloaders=dataloaders["test"])
    elif train == False and val == False and test == True:
        trainer.test(
            model=model_pl,
            dataloaders=dataloaders["test"],
            ckpt_path=config["test_model_path"] if config["test_model_path"] else "best",
        )
    elif train == True and val == True and test == False:
        trainer.fit(
            model=model_pl,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["val"],
        )
    elif train == True and val == False and test == True:
        trainer.fit(model=model_pl, train_dataloaders=dataloaders["train"])
        trainer.test(
            model=model_pl,
            dataloaders=dataloaders["test"],
            ckpt_path=config["test_model_path"] if config["test_model_path"] else "best",
        )
    if config["group_name"] is not None:
        wandb.finish()


class LitCallback_all(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def on_train_start(self, trainer, pl_module):
        print("start training")

    def on_validation_start(self, trainer, pl_module):
        self.click_pred = []
        self.conversion_pred = []
        self.conversion_pred_filter = []
        self.click_conversion_pred = []
        self.click_label = []
        self.conversion_label = []
        self.conversion_label_filter = []
        self.click_conversion_label = []
        self.total_val_loss = 0.0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.total_val_loss += outputs["val_loss"].item()
        self.click_pred.append(outputs["click_pred"])
        self.conversion_pred.append(outputs["conversion_pred"])
        self.conversion_pred_filter.append(outputs["conversion_pred_filter"])
        self.click_conversion_pred.append(outputs["click_conversion_pred"])
        self.click_label.append(outputs["click_label"])
        self.conversion_label.append(outputs["conversion_label"])
        self.conversion_label_filter.append(outputs["conversion_label_filter"])
        self.click_conversion_label.append(outputs["click_conversion_label"])

    def on_validation_epoch_end(self, trainer, pl_module):
        step = len(self.click_pred)
        click_auc, click_precision, click_recall, click_f1 = cal(
            self.click_label, self.click_pred, sigmoid=self.config["sigmoid"]
        )
        conversion_auc, conversion_precision, conversion_recall, conversion_f1 = cal(
            self.conversion_label, self.conversion_pred, sigmoid=self.config["sigmoid"]
        )
        (
            conversion_auc_filter,
            conversion_precision_filter,
            conversion_recall_filter,
            conversion_f1_filter,
        ) = cal(
            self.conversion_label_filter,
            self.conversion_pred_filter,
            sigmoid=self.config["sigmoid"],
        )
        (
            click_conversion_auc,
            click_conversion_precision,
            click_conversion_recall,
            click_conversion_f1,
        ) = cal(
            self.click_conversion_label, self.click_conversion_pred, sigmoid=self.config["sigmoid"]
        )
        combined_auc = (click_auc + click_conversion_auc) / 2.0
        print(
            "click_auc: {} click_precision: {} click_recall: {} click_f1: {}".format(
                click_auc, click_precision, click_recall, click_f1
            )
        )
        print(
            "conversion_auc: {} conversion_precision: {} conversion_recall: {} conversion_f1: {}".format(
                conversion_auc, conversion_precision, conversion_recall, conversion_f1
            )
        )
        print(
            "conversion_auc_filter: {} conversion_precision_filter: {} conversion_recall_filter: {} conversion_f1_filter: {}".format(
                conversion_auc_filter,
                conversion_precision_filter,
                conversion_recall_filter,
                conversion_f1_filter,
            )
        )
        print(
            "click_conversion_auc: {} click_conversion_precision: {} click_conversion_recall: {} click_conversion_f1: {}".format(
                click_conversion_auc,
                click_conversion_precision,
                click_conversion_recall,
                click_conversion_f1,
            )
        )
        print(f"val_loss: {self.total_val_loss / step}")
        print(f"combined_auc: {combined_auc}")

        self.log_dict(
            {
                "click_auc": click_auc,
                "conversion_auc": conversion_auc,
                "conversion_auc_filter": conversion_auc_filter,
                "click_conversion_auc": click_conversion_auc,
                "val_loss": self.total_val_loss / step,
                "combined_auc": combined_auc,
            }
        )
        self.log_dict(
            {
                "click_precision": click_precision,
                "click_recall": click_recall,
                "click_f1": click_f1,
                "conversion_precision": conversion_precision,
                "conversion_recall": conversion_recall,
                "conversion_f1": conversion_f1,
                "conversion_precision_filter": conversion_precision_filter,
                "conversion_recall_filter": conversion_recall_filter,
                "conversion_f1_filter": conversion_f1_filter,
                "click_conversion_precision": click_conversion_precision,
                "click_conversion_recall": click_conversion_recall,
                "click_conversion_f1": click_conversion_f1,
            }
        )

    def on_test_start(self, trainer, pl_module):
        self.click_pred_test = []
        self.conversion_pred_test = []
        self.conversion_pred_filter_test = []
        self.click_conversion_pred_test = []
        self.click_label_test = []
        self.conversion_label_test = []
        self.conversion_label_filter_test = []
        self.click_conversion_label_test = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.click_pred_test.append(outputs["click_pred_test"])
        self.conversion_pred_test.append(outputs["conversion_pred_test"])
        self.conversion_pred_filter_test.append(outputs["conversion_pred_filter_test"])
        self.click_conversion_pred_test.append(outputs["click_conversion_pred_test"])
        self.click_label_test.append(outputs["click_label_test"])
        self.conversion_label_test.append(outputs["conversion_label_test"])
        self.conversion_label_filter_test.append(outputs["conversion_label_filter_test"])
        self.click_conversion_label_test.append(outputs["click_conversion_label_test"])

    def on_test_epoch_end(self, trainer, pl_module):
        click_auc_test, click_precision_test, click_recall_test, click_f1_test = cal(
            self.click_label_test, self.click_pred_test, sigmoid=self.config["sigmoid"]
        )
        (
            conversion_auc_test,
            conversion_precision_test,
            conversion_recall_test,
            conversion_f1_test,
        ) = cal(
            self.conversion_label_test, self.conversion_pred_test, sigmoid=self.config["sigmoid"]
        )
        (
            conversion_auc_filter_test,
            conversion_precision_filter_test,
            conversion_recall_filter_test,
            conversion_f1_filter_test,
        ) = cal(
            self.conversion_label_filter_test,
            self.conversion_pred_filter_test,
            sigmoid=self.config["sigmoid"],
        )
        (
            click_conversion_auc_test,
            click_conversion_precision_test,
            click_conversion_recall_test,
            click_conversion_f1_test,
        ) = cal(
            self.click_conversion_label_test,
            self.click_conversion_pred_test,
            sigmoid=self.config["sigmoid"],
        )
        print(
            "click_auc_test: {} click_precision_test: {} click_recall_test: {} click_f1_test: {}".format(
                click_auc_test, click_precision_test, click_recall_test, click_f1_test
            )
        )
        print(
            "conversion_auc_test: {} conversion_precision_test: {} conversion_recall_test: {} conversion_f1_test: {}".format(
                conversion_auc_test,
                conversion_precision_test,
                conversion_recall_test,
                conversion_f1_test,
            )
        )
        print(
            "conversion_auc_filter_test: {} conversion_precision_filter_test: {} conversion_recall_filter_test: {} conversion_f1_filter_test: {}".format(
                conversion_auc_filter_test,
                conversion_precision_filter_test,
                conversion_recall_filter_test,
                conversion_f1_filter_test,
            )
        )
        print(
            "click_conversion_auc_test: {} click_conversion_precision_test: {} click_conversion_recall_test: {} click_conversion_f1_test: {}".format(
                click_conversion_auc_test,
                click_conversion_precision_test,
                click_conversion_recall_test,
                click_conversion_f1_test,
            )
        )

        self.log_dict(
            {
                "click_auc_test": click_auc_test,
                "conversion_auc_test": conversion_auc_test,
                "conversion_auc_filter_test": conversion_auc_filter_test,
                "click_conversion_auc_test": click_conversion_auc_test,
            }
        )
        self.log_dict(
            {
                "click_precision_test": click_precision_test,
                "click_recall_test": click_recall_test,
                "click_f1_test": click_f1_test,
                "conversion_precision_test": conversion_precision_test,
                "conversion_recall_test": conversion_recall_test,
                "conversion_f1_test": conversion_f1_test,
                "conversion_precision_filter_test": conversion_precision_filter_test,
                "conversion_recall_filter_test": conversion_recall_filter_test,
                "conversion_f1_filter_test": conversion_f1_filter_test,
                "click_conversion_precision_test": click_conversion_precision_test,
                "click_conversion_recall_test": click_conversion_recall_test,
                "click_conversion_f1_test": click_conversion_f1_test,
            }
        )


class LitCallback(Callback):
    def __init__(self, config):
        self.config = config
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("start training")

    def on_validation_start(self, trainer, pl_module):
        self.click_pred = []
        self.conversion_pred = []
        self.conversion_pred_filter = []
        self.click_conversion_pred = []
        self.click_label = []
        self.conversion_label = []
        self.conversion_label_filter = []
        self.click_conversion_label = []
        self.total_val_loss = 0.0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.total_val_loss += outputs["val_loss"].item()
        self.click_pred.append(outputs["click_pred"])
        self.conversion_pred.append(outputs["conversion_pred"])
        self.conversion_pred_filter.append(outputs["conversion_pred_filter"])
        self.click_conversion_pred.append(outputs["click_conversion_pred"])
        self.click_label.append(outputs["click_label"])
        self.conversion_label.append(outputs["conversion_label"])
        self.conversion_label_filter.append(outputs["conversion_label_filter"])
        self.click_conversion_label.append(outputs["click_conversion_label"])

    def on_validation_epoch_end(self, trainer, pl_module):
        step = len(self.click_pred)
        click_auc = cal_auc(self.click_label, self.click_pred)
        conversion_auc = cal_auc(self.conversion_label, self.conversion_pred)
        conversion_auc_filter = cal_auc(self.conversion_label_filter, self.conversion_pred_filter)
        click_conversion_auc = cal_auc(self.click_conversion_label, self.click_conversion_pred)
        combined_auc = (click_auc + click_conversion_auc) / 2.0
        print(
            "click_auc: {} conversion_auc: {} conversion_auc_filter: {} click_conversion_auc: {} val_loss: {}".format(
                click_auc,
                conversion_auc,
                conversion_auc_filter,
                click_conversion_auc,
                self.total_val_loss / step,
            )
        )

        self.log_dict(
            {
                "click_auc": click_auc,
                "conversion_auc": conversion_auc,
                "conversion_auc_filter": conversion_auc_filter,
                "click_conversion_auc": click_conversion_auc,
                "val_loss": self.total_val_loss / step,
                "combined_auc": combined_auc,
            }
        )

    def on_test_start(self, trainer, pl_module):
        self.click_pred_test = []
        self.conversion_pred_test = []
        self.conversion_pred_filter_test = []
        self.click_conversion_pred_test = []
        self.click_label_test = []
        self.conversion_label_test = []
        self.conversion_label_filter_test = []
        self.click_conversion_label_test = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.click_pred_test.append(outputs["click_pred_test"])
        self.conversion_pred_test.append(outputs["conversion_pred_test"])
        self.conversion_pred_filter_test.append(outputs["conversion_pred_filter_test"])
        self.click_conversion_pred_test.append(outputs["click_conversion_pred_test"])
        self.click_label_test.append(outputs["click_label_test"])
        self.conversion_label_test.append(outputs["conversion_label_test"])
        self.conversion_label_filter_test.append(outputs["conversion_label_filter_test"])
        self.click_conversion_label_test.append(outputs["click_conversion_label_test"])

    def on_test_epoch_end(self, trainer, pl_module):
        click_auc_test = cal_auc(self.click_label_test, self.click_pred_test)
        conversion_auc_test = cal_auc(self.conversion_label_test, self.conversion_pred_test)
        conversion_auc_filter_test = cal_auc(
            self.conversion_label_filter_test, self.conversion_pred_filter_test
        )
        click_conversion_auc_test = cal_auc(
            self.click_conversion_label_test, self.click_conversion_pred_test
        )

        self.click_label_test = torch.cat(self.click_label_test).cpu().detach().numpy()
        self.click_pred_test = torch.cat(self.click_pred_test).cpu().detach().numpy()
        self.conversion_pred_test = torch.cat(self.conversion_pred_test).cpu().detach().numpy()
        self.conversion_label_test = torch.cat(self.conversion_label_test).cpu().detach().numpy()

        # plt.hist(self.click_pred_test, bins=20, edgecolor='black', alpha=0.5)
        # plt.savefig('./out/tmp.png', format='png')
        # wandb.log({"Distribution of click prediction in exposure space": wandb.Image('./out/tmp.png')})
        # plt.clf()

        # plt.hist(self.click_pred_test[self.click_label_test==1], bins=20, edgecolor='black', alpha=0.5, label='click', color='orange')
        # plt.savefig('./out/tmp.png', format='png')
        # wandb.log({"Distribution of click prediction in click space": wandb.Image('./out/tmp.png')})
        # plt.clf()

        # plt.hist(self.click_pred_test[self.click_label_test==0], bins=20, edgecolor='black', alpha=0.5, label='unclick', color='blue')
        # plt.savefig('./out/tmp.png', format='png')
        # wandb.log({"Distribution of click prediction in unclick space": wandb.Image('./out/tmp.png')})
        # plt.clf()

        # plt.hist(self.click_pred_test[self.conversion_label_test==1], bins=20, edgecolor='black', alpha=0.5, label='conversion', color='orange')
        # plt.savefig('./out/tmp.png', format='png')
        # wandb.log({"Distribution of click prediction in conversion space": wandb.Image('./out/tmp.png')})
        # plt.clf()

        # plt.hist(self.click_pred_test[self.conversion_label_test==0], bins=20, edgecolor='black', alpha=0.5, label='no conversion', color='blue')
        # plt.savefig('./out/tmp.png', format='png')
        # wandb.log({"Distribution of click prediction in no conversion space": wandb.Image('./out/tmp.png')})
        # plt.clf()

        # plt.hist(self.conversion_pred_test, bins=20, edgecolor='black', alpha=0.5)
        # plt.savefig('./out/tmp.png', format='png')
        # wandb.log({"Distribution of conversion prediction in exposure space": wandb.Image('./out/tmp.png')})
        # plt.clf()

        # plt.hist(self.conversion_pred_test[self.click_label_test==1], bins=20, edgecolor='black', alpha=0.5, label='click', color='orange')
        # plt.savefig('./out/tmp.png', format='png')
        # wandb.log({"Distribution of conversion prediction in click space": wandb.Image('./out/tmp.png')})
        # plt.clf()

        # plt.hist(self.conversion_pred_test[self.click_label_test==0], bins=20, edgecolor='black', alpha=0.5, label='unclick', color='blue')
        # plt.savefig('./out/tmp.png', format='png')
        # wandb.log({"Distribution of conversion prediction in unclick space": wandb.Image('./out/tmp.png')})
        # plt.clf()

        # plt.hist(self.conversion_pred_test[self.conversion_label_test==1], bins=20, edgecolor='black', alpha=0.5, label='conversion', color='orange')
        # plt.savefig('./out/tmp.png', format='png')
        # wandb.log({"Distribution of conversion prediction in conversion space": wandb.Image('./out/tmp.png')})
        # plt.clf()

        # plt.hist(self.conversion_pred_test[self.conversion_label_test==0], bins=20, edgecolor='black', alpha=0.5, label='no conversion', color='blue')
        # plt.savefig('./out/tmp.png', format='png')
        # wandb.log({"Distribution of conversion prediction in no conversion space": wandb.Image('./out/tmp.png')})
        # plt.clf()

        self.log_dict(
            {
                "click_auc_test": click_auc_test,
                "conversion_auc_test": conversion_auc_test,
                "conversion_auc_filter_test": conversion_auc_filter_test,
                "click_conversion_auc_test": click_conversion_auc_test,
            }
        )
        print(
            "click_auc_test: {} conversion_auc_test: {} conversion_auc_filter_test: {} click_conversion_auc_test: {}".format(
                click_auc_test,
                conversion_auc_test,
                conversion_auc_filter_test,
                click_conversion_auc_test,
            )
        )


class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = config["model"](config)
        self.loss = config["model_loss"](config)
        self.lr = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.batch_transform = BatchTransform(config)

    def training_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred = self.model(features)
        loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred = self.model(features)
        # filter the conversion_pred where click is 0
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        val_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)
        self.log("val_loss", val_loss, on_epoch=True, on_step=True)

        return {
            "val_loss": val_loss,
            "click_pred": click_pred,
            "conversion_pred": conversion_pred,
            "conversion_pred_filter": conversion_pred_filter,
            "click_conversion_pred": click_conversion_pred,
            "click_label": click,
            "conversion_label": conversion,
            "conversion_label_filter": conversion_filter,
            "click_conversion_label": click * conversion,
        }

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]

        return {
            "click_pred_test": click_pred,
            "conversion_pred_test": conversion_pred,
            "conversion_pred_filter_test": conversion_pred_filter,
            "click_conversion_pred_test": click_conversion_pred,
            "click_label_test": click.float(),
            "conversion_label_test": conversion.float(),
            "conversion_label_filter_test": conversion_filter.float(),
            "click_conversion_label_test": click.float() * conversion.float(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.config["weight_decay"]
        )
        return optimizer


class BatchTransform:
    def __init__(self, config):
        self.config = config

    def __call__(self, batch):
        if self.config["batch_type"] == "in":
            click, conversion, features = (
                batch["click"].squeeze(1).float(),
                batch["conversion"].squeeze(1).float(),
                batch["features"],
            )
            features = features.reshape(
                len(features), self.config["feature_num"], self.config["single_feature_len"]
            )

        elif self.config["batch_type"] == "ali-ccp":
            click, conversion, features = batch
            click = click.float()
            conversion = conversion.float()
        else:
            click, conversion, features = batch

        return click, conversion, features


class MultiTaskModelFactory(ABC):
    @abstractmethod
    def get_multi_task_model(self, config):
        pass

    @abstractmethod
    def get_multi_task_model(self, config):
        pass


class LitModelFactory(ABC):
    def get_lit_model(self, config):
        pass


class MultiTaskLitModel(pl.LightningModule):
    def __init__(self, config, multitask_model, multitask_loss):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = multitask_model
        self.loss = multitask_loss
        self.ctr_auc = BinaryAUROC(thresholds=None)
        self.cvr_auc = BinaryAUROC(thresholds=None)
        self.ctcvr_auc = BinaryAUROC(thresholds=None)
        self.batch_transform = BatchTransform(config)

    def training_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred = self.model(features)
        # filter the conversion_pred where click is 0
        val_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)
        self.log("val_loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.ctr_auc(click_pred, click)
        self.cvr_auc(conversion_pred_filter, conversion_filter)
        self.ctcvr_auc(click_conversion_pred, click * conversion)
        self.log("ctr_auc", self.ctr_auc, on_epoch=True)
        self.log("cvr_auc", self.cvr_auc, on_epoch=True)
        self.log("ctcvr_auc", self.ctcvr_auc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        return optimizer
