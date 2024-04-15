import sys

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig
from torch import nn
from torchmetrics.classification import AUROC, Accuracy, BinaryAUROC

from src.models.common import BatchTransform, MultiLayerPerceptron


class ESMM(nn.Module):
    def __init__(
        self,
        embedding_layer,
        tower_mlp_dims,
        dropout_tower,
    ):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.embed_output_dim = self.embedding_layer.get_embed_output_dim()
        self.tower_ctr = MultiLayerPerceptron(self.embed_output_dim, tower_mlp_dims, dropout_tower)
        self.tower_cvr = MultiLayerPerceptron(self.embed_output_dim, tower_mlp_dims, dropout_tower)

    def forward(self, x):
        feature_embedding = self.embedding_layer(x)

        p_ctr_logits = self.tower_ctr(feature_embedding)
        p_ctr = torch.sigmoid(p_ctr_logits)
        p_cvr_logits = self.tower_cvr(feature_embedding)
        p_cvr = torch.sigmoid(p_cvr_logits)
        p_ctcvr = torch.mul(p_ctr, p_cvr)

        return p_ctr.squeeze(1), p_cvr.squeeze(1), p_ctcvr.squeeze(1)


class Loss_ESMM(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr):
        loss_ctr = self.loss(p_ctr, y_ctr)
        # loss_cvr = self.loss(p_cvr, y_cvr)
        loss_ctcvr = self.loss(p_ctcvr, y_ctr * y_cvr)

        return loss_ctr + loss_ctcvr


class ESMMLitModel(pl.LightningModule):
    def __init__(self, model, loss, lr, weight_decay, batch_type):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.ctr_auc = BinaryAUROC()
        self.cvr_auc = BinaryAUROC()
        self.ctcvr_auc = BinaryAUROC()
        self.batch_transform = BatchTransform(batch_type)

    def training_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred = self.model(features)
        # filter the conversion_pred where click is 0
        val_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.ctr_auc.update(click_pred, click)
        self.cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.ctcvr_auc.update(click_conversion_pred, click * conversion)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class ESMMCallback(Callback):
    def __init__(
        self,
    ):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("start training teacher")

    def on_test_epoch_end(self, trainer, pl_module):
        self.log("test/ctr_auc", pl_module.ctr_auc.compute())
        self.log("test/cvr_auc", pl_module.cvr_auc.compute())
        self.log("test/ctcvr_auc", pl_module.ctcvr_auc.compute())
        pl_module.ctr_auc.reset()
        pl_module.cvr_auc.reset()
        pl_module.ctcvr_auc.reset()


class MMoEModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_layer, self.embed_output_dim = read_embedding_layer(config)
        # expert, gates and towers
        self.task_num = config["task_num"]
        self.expert_num = config["expert_num"]

        self.expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.embed_output_dim,
                    config["bottom_mlp_dims"],
                    config["dropout_expert"],
                    output_layer=False,
                )
                for i in range(self.expert_num)
            ]
        )

        self.tower = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    config["bottom_mlp_dims"][-1],
                    config["tower_mlp_dims"],
                    config["dropout_tower"],
                )
                for i in range(self.task_num)
            ]
        )

        self.gate = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.embed_output_dim, self.expert_num),
                    torch.nn.Softmax(dim=1),
                )
                for i in range(self.task_num)
            ]
        )

    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        gate_value = [self.gate[i](feature_embedding).unsqueeze(1) for i in range(self.task_num)]
        fea = torch.cat(
            [self.expert[i](feature_embedding).unsqueeze(1) for i in range(self.expert_num)], dim=1
        )
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        results = [
            torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)
        ]

        return results[0], results[1], torch.mul(results[0], results[1])


class Loss_MMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr):
        loss_ctr = self.loss(p_ctr, y_ctr)
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction="none")
        loss_cvr = torch.mean(loss_cvr * y_ctr)
        # loss_ctcvr = self.loss(p_ctcvr, y_ctr * y_cvr)

        return loss_ctr + loss_cvr


