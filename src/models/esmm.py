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
        self.tower_mlp_dims = tower_mlp_dims
        self.dropout_tower = dropout_tower
        self.tower_ctr = MultiLayerPerceptron(self.embed_output_dim, tower_mlp_dims, dropout_tower)
        self.tower_cvr = MultiLayerPerceptron(self.embed_output_dim, tower_mlp_dims, dropout_tower)

    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        p_ctr_logits = self.tower_ctr(feature_embedding)
        p_ctr = torch.sigmoid(p_ctr_logits)
        p_cvr_logits = self.tower_cvr(feature_embedding)
        p_cvr = torch.sigmoid(p_cvr_logits)
        p_ctcvr = torch.mul(p_ctr, p_cvr)

        return p_ctr.squeeze(1), p_cvr.squeeze(1), p_ctcvr.squeeze(1), p_ctr, feature_embedding

class Loss_ESMM(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, **kwargs):
        loss_ctr = self.loss(p_ctr, y_ctr)
        # loss_cvr = self.loss(p_cvr, y_cvr)
        loss_ctcvr = self.loss(p_ctcvr, y_ctr * y_cvr)

        return loss_ctr + loss_ctcvr







