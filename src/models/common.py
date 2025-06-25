from abc import ABC, abstractmethod
import os 
import sys
from typing import Any, Dict, Tuple, List
import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig
from torch import nn
from torchmetrics.classification import AUROC, Accuracy, BinaryAUROC # type: ignore
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dims,
        dropout,
        drop_last_dropout=False,
        output_layer=True,
    ):
        super().__init__()
        layers = []
        for i, embed_dim in enumerate(embed_dims):
            layers.extend((torch.nn.Linear(input_dim, embed_dim), torch.nn.ReLU()))
            if dropout[i] > 0:
                layers.append(torch.nn.Dropout(p=dropout[i]))
            input_dim = embed_dim
        if drop_last_dropout == True:
            layers.pop()
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    
class MlpLayerFea(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dims,
        dropout,
        drop_last_dropout=False,
        output_layer=True,
    ):
        super().__init__()
        layers = []
        for i, embed_dim in enumerate(embed_dims):
            layers.extend((torch.nn.Linear(input_dim, embed_dim), torch.nn.ReLU()))
            if dropout[i] > 0:
                layers.append(torch.nn.Dropout(p=dropout[i]))
            input_dim = embed_dim
        if drop_last_dropout == True:
            layers.pop()
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        # forward x layer by layer
        layer_output = []
        for i in range(0, len(self.layers), 2):
            x = self.layers[i](x)
            x = self.layers[i+1](x)
            layer_output.append(x)
        if len(self.layers) % 2 == 1:
            layer_output.append(self.layers[i+2](x))
        return layer_output
    

class AlldataEmbeddingLayer(torch.nn.Module):
    def __init__(self, batch_type="ccp", embedding_size=5):
        super().__init__()
        self.batch_type = batch_type
        self.numerical_num = 63
        self.embedding_size = embedding_size


        if batch_type == "es":
            self.field_dims = [8, 4, 7, 2, 19, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2]
            self.embedding_layer = torch.nn.Embedding(sum(self.field_dims), embedding_size)
            self.numerical_layer = torch.nn.Linear(self.numerical_num, embedding_size)
            self.embed_output_dim = (len(self.field_dims) + 1) * embedding_size


        # self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight.data)

    def forward(self, x):

        categorical_x, numerical_x = x
        categorical_emb = self.embedding_layer(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        return torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)

    def get_embed_output_dim(self):
        return self.embed_output_dim


class BatchTransform:
    def __init__(self, batch_type="ccp"):
        self.batch_type = batch_type
        self.single_feature_len = 3

    def __call__(self, batch):
        click, conversion, features = batch

        return click, conversion, features

class MultiTaskCallback(Callback):
    def __init__(
        self,
    ):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("start training")
        
    def dim_zero_cat(self, x):
        """Concatenation along the zero dimension."""
        if isinstance(x, torch.Tensor):
            return x
        x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
        if not x:  # empty list
            raise ValueError("No samples to concatenate")
        return torch.cat(x, dim=0)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # judge if a attribute is not None

        if hasattr(pl_module, 'val_cvr_auc'):
            self.log("val/cvr_auc", pl_module.val_cvr_auc.compute())
            pval_cvr = self.dim_zero_cat(pl_module.val_cvr_auc.preds)
            pval_cvr_target =  self.dim_zero_cat(pl_module.val_cvr_auc.target)
            pl_module.val_cvr_auc.reset()

    
    def on_test_epoch_end(self, trainer, pl_module):

        if hasattr(pl_module, 'cvr_auc'):
            self.log("test/cvr_auc", pl_module.cvr_auc.compute())
            pcvr = self.dim_zero_cat(pl_module.cvr_auc.preds)
            pcvr_target =  self.dim_zero_cat(pl_module.cvr_auc.target)
            pl_module.cvr_auc.reset()
        
    