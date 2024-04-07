import os

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import get_dataloaders
from lightning.pytorch.callbacks import Callback

from models.cgam import CGAM_V3, ContrastiveLoss, Loss_Student
from utils import BatchTransform, single_train

vocabulary_size = {
    "101": 238635,
    "121": 98,
    "122": 14,
    "124": 3,
    "125": 8,
    "126": 4,
    "127": 4,
    "128": 3,
    "129": 5,
    "205": 467298,
    "206": 6929,
    "207": 263942,
    "216": 106399,
    "508": 5888,
    "509": 104830,
    "702": 51878,
    "853": 37148,
    "301": 4,
}


class LitCGCM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = config["student_model"](config)
        self.loss = config["student_model_loss"](config)
        self.lr = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.contrastive_loss = config["contrastive_loss"](config)
        self.batch_transform = BatchTransform(config)

    def forward(self, features):
        return self.model(features)

    def training_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)

        click_pred, conversion_pred, click_conversion_pred, imp_pred, embeddings = self.model(
            features
        )

        contrastive_loss = self.config["contrastive_loss_proportion"] * self.contrastive_loss(
            click_pred, click, conversion_pred, conversion, embeddings
        )

        loss = (
            self.loss(
                click_pred, conversion_pred, click_conversion_pred, click, conversion, imp_pred
            )
            + contrastive_loss
        )

        self.log("train_loss", loss, on_epoch=True, on_step=True)
        self.log("contrastive_loss", contrastive_loss, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, fea = self.model(features)
        # filter the conversion_pred where click is 0
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]

        val_loss = self.loss(
            click_pred, conversion_pred, click_conversion_pred, click, conversion, imp_pred
        )

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
        click_pred, conversion_pred, click_conversion_pred, imp_pred, fea = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]

        return {
            "click_pred_test": click_pred,
            "conversion_pred_test": conversion_pred,
            "conversion_pred_filter_test": conversion_pred_filter,
            "click_conversion_pred_test": click_conversion_pred,
            "click_label_test": click,
            "conversion_label_test": conversion,
            "conversion_label_filter_test": conversion_filter,
            "click_conversion_label_test": click * conversion,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.config["weight_decay"]
        )
        return optimizer


class LitCallback(Callback):
    def __init__(self, config):
        self.config = config
        super().__init__()

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
        self.click_label_test = torch.cat(self.click_label_test).cpu().detach().numpy()
        self.click_pred_test = torch.cat(self.click_pred_test).cpu().detach().numpy()
        self.conversion_pred_test = torch.cat(self.conversion_pred_test).cpu().detach().numpy()
        self.conversion_label_test = torch.cat(self.conversion_label_test).cpu().detach().numpy()
        propensity_scores = 1 / self.click_pred_test
        propensity_scores_click = 1 / self.click_pred_test[self.click_label_test == 1]
        propensity_scores_unclick = 1 / self.click_pred_test[self.click_label_test == 0]
        propensity_scores_click_expection = propensity_scores_click.mean()
        cvr_expection = self.conversion_label_test.mean()
        cvr_expection_prediction = self.conversion_pred_test.mean()
        cvr_expection_click = self.conversion_label_test[self.click_label_test == 1].mean()
        cvr_expection_click_prediction = self.conversion_pred_test[
            self.click_label_test == 1
        ].mean()
        cvr_expection_click_weighted = np.average(
            self.conversion_label_test[self.click_label_test == 1],
            weights=propensity_scores[self.click_label_test == 1],
        )
        print("propensity_scores_click_expection", propensity_scores_click_expection)
        print("propensity_scores_click_variance", propensity_scores_click.var())
        print("cvr_expection", cvr_expection)
        print("cvr_expection_prediction", cvr_expection_prediction)
        print("cvr_expection_prediction_variance", self.conversion_pred_test.var())
        print("cvr_expection_click", cvr_expection_click)
        print("cvr_expection_click_prediction", cvr_expection_click_prediction)
        print("cvr_expection_click_weighted", cvr_expection_click_weighted)

        # convert self.click_pred_test[self.click_label_test==1] to pandas
        propensity_scores_click_df = pd.DataFrame(propensity_scores_click)
        propensity_scores_click_df.to_csv(
            config["figure_dir"] + "/propensity_scores_click.csv", index=False
        )
        propensity_scores_unclick_df = pd.DataFrame(propensity_scores_unclick)
        propensity_scores_unclick_df = propensity_scores_unclick_df.sample(
            n=len(propensity_scores_click_df), random_state=1
        )
        propensity_scores_unclick_df.to_csv(
            config["figure_dir"] + "/propensity_scores_click_unclick.csv", index=False
        )
        k = int(len(propensity_scores_click_df) * 0.1)
        topk_of_propensity_scores_click = propensity_scores_click_df.sort_values(
            by=0, ascending=False
        ).head(k)
        topk_of_propensity_scores_click.to_csv(
            config["figure_dir"] + "/topk_of_propensity_scores_click.csv", index=False
        )
        topk_of_propensity_scores_unclick = propensity_scores_unclick_df.sort_values(
            by=0, ascending=True
        ).head(k)
        topk_of_propensity_scores_unclick.to_csv(
            config["figure_dir"] + "/topk_of_propensity_scores_unclick.csv", index=False
        )

        # convert self.conversion_pred_test to pandas
        conversion_pred_test_df = pd.DataFrame(self.conversion_pred_test)
        # random sample 400000 rows
        conversion_pred_test_df = conversion_pred_test_df.sample(n=1000000, random_state=1)
        conversion_pred_test_df.to_csv(
            config["figure_dir"] + "/conversion_pred_test.csv", index=False
        )

        plt.hist(self.click_pred_test, bins=100, edgecolor="black", alpha=0.5, color="orange")
        plt.savefig(
            config["figure_dir"]
            + "/click_prediction_in_exposure_space_"
            + config["png_name"]
            + ".png",
            format="png",
        )
        plt.clf()

        plt.hist(
            self.click_pred_test[self.click_label_test == 1],
            bins=100,
            edgecolor="black",
            alpha=0.5,
            color="green",
        )
        plt.savefig(
            config["figure_dir"]
            + "/click_prediction_in_click_space_"
            + config["png_name"]
            + ".png",
            format="png",
        )
        plt.clf()

        plt.hist(
            self.click_pred_test[self.click_label_test == 0],
            bins=100,
            edgecolor="black",
            alpha=0.5,
            color="blue",
        )
        plt.savefig(
            config["figure_dir"]
            + "/click_prediction_in_unclick_space_"
            + config["png_name"]
            + ".png",
            format="png",
        )
        plt.clf()

        plt.hist(
            1.0 / self.click_pred_test, bins=100, edgecolor="black", alpha=0.5, color="orange"
        )
        plt.xlim(0, 100)
        plt.savefig(
            config["figure_dir"]
            + "/inversed_click_prediction_in_exposure_space_"
            + config["png_name"]
            + ".png",
            format="png",
        )
        plt.clf()

        plt.hist(
            1.0 / self.click_pred_test[self.click_label_test == 1],
            bins=100,
            edgecolor="black",
            alpha=0.5,
            color="green",
        )
        plt.xlim(0, 100)
        plt.savefig(
            config["figure_dir"]
            + "/inversed_click_prediction_in_click_space_"
            + config["png_name"]
            + ".png",
            format="png",
        )
        plt.clf()

        plt.hist(
            1.0 / self.click_pred_test[self.click_label_test == 0],
            bins=100,
            edgecolor="black",
            alpha=0.5,
            color="blue",
        )
        plt.xlim(0, 100)
        plt.savefig(
            config["figure_dir"]
            + "/inversed_click_prediction_in_unclick_space_"
            + config["png_name"]
            + ".png",
            format="png",
        )
        plt.clf()

        plt.hist(
            self.conversion_pred_test, bins=1000, edgecolor="black", alpha=0.5, color="orange"
        )
        plt.xlim(0, 0.01)
        plt.axvline(x=cvr_expection, color="r", linestyle="--")
        plt.axvline(x=cvr_expection_prediction, color="g", linestyle="--")
        plt.savefig(
            config["figure_dir"]
            + "/conversion_prediction_in_exposure_space_"
            + config["png_name"]
            + ".png",
            format="png",
        )
        plt.clf()
        plt.hist(
            self.conversion_pred_test[self.click_label_test == 0],
            bins=1000,
            edgecolor="black",
            alpha=0.5,
            color="blue",
        )
        plt.xlim(0, 0.01)
        plt.axvline(x=cvr_expection, color="r", linestyle="--")
        plt.axvline(x=cvr_expection_prediction, color="g", linestyle="--")
        plt.savefig(
            config["figure_dir"]
            + "/conversion_prediction_in_unclick_space_"
            + config["png_name"]
            + ".png",
            format="png",
        )
        plt.clf()
        plt.hist(
            self.conversion_pred_test[self.click_label_test == 1],
            bins=1000,
            edgecolor="black",
            alpha=0.5,
            color="green",
        )
        plt.xlim(0, 0.01)
        plt.axvline(x=cvr_expection_click, color="r", linestyle="--")
        plt.axvline(x=cvr_expection_click_prediction, color="g", linestyle="--")
        plt.savefig(
            config["figure_dir"]
            + "/conversion_prediction_in_click_space_"
            + config["png_name"]
            + ".png",
            format="png",
        )
        plt.clf()

        plt.hist(
            self.conversion_pred_test[self.click_label_test == 1],
            weights=propensity_scores[self.click_label_test == 1],
            bins=1000,
            edgecolor="black",
            alpha=0.5,
            color="green",
        )
        plt.xlim(0, 0.01)
        plt.axvline(x=cvr_expection, color="r", linestyle="--")
        plt.axvline(x=cvr_expection_click_weighted, color="g", linestyle="--")
        plt.savefig(
            config["figure_dir"]
            + "/reweight_conversion_prediction_in_click_space_"
            + config["png_name"]
            + ".png",
            format="png",
        )


if __name__ == "__main__":
    config = dict(
        # basic
        global_seed=2020,
        vocabulary_size=vocabulary_size,
        single_embedding_vocabulary_size=737946,
        sigmoid=False,
        # dataloader
        num_workers=8,
        num_workers_test=8,
        persistent_workers=False,
        test_persistent_workers=False,
        shuffle=True,
        # early stop
        using_early_stop=False,
        monitor_metric="val_loss_epoch",
        mode="min",
        earlystop_epoch=1,
        min_delta=0.0002,
        # checkpoint
        using_model_checkpoint=False,
        checkpoint_monitor_metric="val_loss_epoch",
        checkpoint_monitor_mode="min",
        save_top_k=1,
        check_point_path="./out/",
        # model structure and hyperparameters
        total_epoch=1,
        batch_size=8000,
        learning_rate=0.001,
        embedding_size=5,
        weight_decay=1e-6,
        presion="32",
        expert_num=8,
        task_num=3,
        embed_output_dim=90,
        bottom_mlp_dims=[256],
        tower_mlp_dims=[128, 64, 32],
        dropout_expert=[0.3],
        dropout_tower=[0.1, 0.3, 0.3],
        # student hyperparameters
        student_model=CGAM_V3,
        student_model_loss=Loss_Student,
        batch_type="fr",
        counterfact_mode="IPW",
        # contrastive loss
        hard_positive_topk=50,
        hard_negative_topk=50,
        contrastive_loss_proportion=0.4,
        contrastive_loss=ContrastiveLoss,
        mining_target="mining_click_embedding",  # 'mining_click_embedding','mining_conversion', 'mining_click', 'mining_both'
        contrastive_type="CosineSimilarity_ntxent",  #'CosineSimilarity_ntxent','swap_positive_and_negative_triplet', 'basic_triplet', 'inverse_triplet', 'inverse_exp_triplet', 'LpDistance_ntxent', 'negative_set_to_1_LpDistance_ntxent'
        temperature=0.5,
        negative_setting_number=1,
        normalize_embeddings=False,
        # basic loss
        cvr_nonclick_trade_off=0,
        ctr_nonclick_trade_off=0.45,
        ctr_loss_proportion=1,
        cvr_loss_proportion=1,
        ctcvr_loss_proportion=0.1,
        weight_clicked=0.5,
        weight_conversion=0.1,
        p_ctr_down_clip=0.05,
        # wandb
        using_wandb=False,
        project="CVR prediction test",
        name="student-hard_mining_and_ctr_debias",
        group_name="CGAM-IPW-constractive",
        log_model=False,
        profiler=None,  # 'simple' , 'pytorch'
    )
    debug = False
    config["batch_type"] = "fr"
    config["using_wandb"] = False
    dataloaders, config["field_dims"], config["numerical_num"] = get_dataloaders(
        config, debug=debug
    )

    config["test_model_path"] = "./out/ctr_base_model.ckpt"
    config["png_name"] = "ctr_base_model"
    config["figure_dir"] = "./out/figure_train/ctr_base_model"
    # config['test_model_path'] = './out/contrastive_no_cliping_ctr.ckpt'
    # config['png_name'] = 'contrastive_no_cliping_ctr'
    # config['figure_dir'] = './out/figure_train/contrastive_no_cliping_ctr'
    if not os.path.exists(config["figure_dir"]):
        os.makedirs(config["figure_dir"])

    dataloaders["test"] = dataloaders["train"]
    litmodel = LitCGCM.load_from_checkpoint(config["test_model_path"], config=config)
    litcallback = LitCallback(config)
    single_train(config, litmodel, litcallback, dataloaders, train=False, test=True)
