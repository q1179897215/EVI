from abc import ABC, abstractmethod
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

multi_embedding_vocabulary_size = {
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
using_feature_ids = [str(i) for i in range(330)]
using_feature_ids.pop(295)
in_feature_names = [i for i in range(0, len(using_feature_ids))]
cpp_feature_names = sorted(list(multi_embedding_vocabulary_size.keys()))
cpp_embedding_vocabulary_size = 737946
in_embedding_vocabulary_size = 10614790


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
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class AlldataEmbeddingLayer(torch.nn.Module):
    def __init__(self, batch_type="ccp", embedding_size=5):
        super().__init__()
        self.batch_type = batch_type
        self.numerical_num = 63
        self.embedding_size = embedding_size

        if batch_type == "ccp":
            self.feature_names = cpp_feature_names
            self.embedding_layer = torch.nn.Embedding(737946, embedding_size)
            self.embed_output_dim = 18 * embedding_size
        elif batch_type == "in":
            self.feature_names = in_feature_names
            self.embedding_layer = torch.nn.Embedding(10614790, embedding_size)
            self.embed_output_dim = len(self.feature_names) * embedding_size
        elif batch_type == "fr":
            self.field_dims = [9, 4, 7, 2, 20, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2]
            self.embedding_layer = torch.nn.Embedding(sum(self.field_dims), embedding_size)
            self.numerical_layer = torch.nn.Linear(self.numerical_num, embedding_size)
            self.embed_output_dim = (len(self.field_dims) + 1) * embedding_size
        elif batch_type == "nl":
            self.field_dims = [9, 4, 7, 2, 20, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2]
            self.embedding_layer = torch.nn.Embedding(sum(self.field_dims), embedding_size)
            self.numerical_layer = torch.nn.Linear(self.numerical_num, embedding_size)
            self.embed_output_dim = (len(self.field_dims) + 1) * embedding_size
        elif batch_type == "es":
            self.field_dims = [8, 4, 7, 2, 19, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2]
            self.embedding_layer = torch.nn.Embedding(sum(self.field_dims), embedding_size)
            self.numerical_layer = torch.nn.Linear(self.numerical_num, embedding_size)
            self.embed_output_dim = (len(self.field_dims) + 1) * embedding_size
        elif batch_type == "us":
            self.field_dims = [10, 4, 7, 2, 21, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2]
            self.embedding_layer = torch.nn.Embedding(sum(self.field_dims), embedding_size)
            self.numerical_layer = torch.nn.Linear(self.numerical_num, embedding_size)
            self.embed_output_dim = (len(self.field_dims) + 1) * embedding_size

        # self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight.data)

    def forward(self, x):
        if self.batch_type == "ccp":
            feature_embedding = []
            for name in self.feature_names:
                embed = self.embedding_layer(x[name])
                # print(embed.shape) --> [batch_size, embedding_size] = [2000, 5]
                feature_embedding.append(embed)
            return torch.cat(feature_embedding, 1)
        elif self.batch_type == "in":
            feature_embedding = []
            for name in range(len(self.feature_names)):
                embed = self.embedding_layer(x[:, name, :])
                embed_sum = torch.sum(embed, dim=1)
                # print(embed.shape) --> [batch_size, embedding_size] = [2000, 5]
                feature_embedding.append(embed_sum)
            return torch.cat(feature_embedding, 1)
        else:
            categorical_x, numerical_x = x
            categorical_emb = self.embedding_layer(categorical_x)
            numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
            return torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)

    def get_embed_output_dim(self):
        return self.embed_output_dim


class BatchTransform:
    def __init__(self, batch_type="ccp"):
        self.batch_type = batch_type
        self.feature_num = len(using_feature_ids)
        self.single_feature_len = 3

    def __call__(self, batch):
        if self.batch_type == "in":
            click, conversion, features = (
                batch["click"].squeeze(1).float(),
                batch["conversion"].squeeze(1).float(),
                batch["features"],
            )
            features = features.reshape(len(features), self.feature_num, self.single_feature_len)

        elif self.batch_type == "ccp":
            click, conversion, features = batch
            click = click.float()
            conversion = conversion.float()
        else:
            click, conversion, features = batch

        return click, conversion, features

class MultiTaskCallback(Callback):
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

class MultiTaskLitModel(pl.LightningModule):
    def __init__(self, model, loss, lr, weight_decay, batch_type):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
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
        click_pred, conversion_pred, click_conversion_pred, imp_pred, task_feature = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion, p_imp=imp_pred)
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, task_feature = self.model(features)
        # filter the conversion_pred where click is 0
        val_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion, p_imp=imp_pred)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, task_feature = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.ctr_auc.update(click_pred, click)
        self.cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.ctcvr_auc.update(click_conversion_pred, click * conversion)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class BasicMultiTaskLoss():
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        self.ctr_loss_proportion = ctr_loss_proportion
        self.cvr_loss_proportion = cvr_loss_proportion
        self.ctcvr_loss_proportion = ctcvr_loss_proportion
    # @abstractmethod
    def caculate_loss(self, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        pass
        
    def __call__(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, **kwargs):
        loss_ctr, loss_cvr, loss_ctcvr = self.caculate_loss(p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs)
        return self.ctr_loss_proportion*loss_ctr + self.cvr_loss_proportion*loss_cvr + self.ctcvr_loss_proportion*loss_ctcvr

class Impression_CTR_IPW_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        p_ctr_clamp = torch.clamp(p_ctr, 0.05, 1.0-1e-7)
        ips = 1. / p_ctr_clamp
        
        loss_ctr = torch.nn.functional.binary_cross_entropy(
            p_ctr, y_ctr, reduction='none'
        )
        loss_ctr = torch.mean(ips*loss_ctr)

        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(y_ctr*loss_cvr)

        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        return loss_ctr, loss_cvr, loss_ctcvr
class Unclick_CTR_IPW_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        p_ctr_clamp = torch.clamp(p_ctr, 0.05, 1.0-1e-7)
        
        unclick_ips = (1 - y_ctr) /  p_ctr_clamp
        click_ips = y_ctr / p_ctr_clamp
        loss_ctr = torch.nn.functional.binary_cross_entropy(
            p_ctr, y_ctr, reduction='none'
        )
        loss_ctr = torch.mean(y_ctr*loss_ctr + unclick_ips*loss_ctr)
        

        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(y_ctr*loss_cvr)

        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        return loss_ctr, loss_cvr, loss_ctcvr
    
class Click_CTR_IPW_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        p_ctr_clamp = torch.clamp(p_ctr, 0.05, 1.0-1e-7)
        ips = 1. / p_ctr_clamp
        
        unclick_ips = (1 - y_ctr) /  p_ctr_clamp
        click_ips = y_ctr / p_ctr_clamp
        loss_ctr = torch.nn.functional.binary_cross_entropy(
            p_ctr, y_ctr, reduction='none'
        )
        loss_ctr = torch.mean(click_ips*loss_ctr + (1  -y_ctr)*loss_ctr)

        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(y_ctr*loss_cvr)

        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        return loss_ctr, loss_cvr, loss_ctcvr
class Impression_CTR_IPW_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        p_ctr_clamp = torch.clamp(p_ctr, 1e-7, 1.0-1e-7)
        ips = 1. / p_ctr_clamp
        
        loss_ctr = torch.nn.functional.binary_cross_entropy(
            p_ctr, y_ctr, reduction='none'
        )
        loss_ctr = torch.mean(ips*loss_ctr)

        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(y_ctr*loss_cvr)

        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        return loss_ctr, loss_cvr, loss_ctcvr

class CTR_IPW_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        p_ctr_clamp = torch.clamp(p_ctr, 1e-7, 1.0-1e-7)
        ips = 1. / p_ctr_clamp
        
        loss_ctr = torch.nn.functional.binary_cross_entropy(
            p_ctr, y_ctr, reduction='none'
        )
        loss_ctr = torch.mean(ips*loss_cvr)

        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(y_ctr*loss_cvr)

        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        return loss_ctr, loss_cvr, loss_ctcvr

class IPW_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        
        loss_ctr = torch.nn.functional.binary_cross_entropy(
            p_ctr, y_ctr, reduction='mean'
        )

        p_ctr_clamp = torch.clamp(p_ctr, 1e-7, 1.0-1e-7)
        ips = y_ctr / p_ctr_clamp
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(ips*loss_cvr)

        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        return loss_ctr, loss_cvr, loss_ctcvr


class DR_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        p_imp = kwargs['p_imp']
        p_ctr_clamp = torch.clamp(p_ctr, 1e-7, 1.0-1e-7)
        ips = y_ctr / p_ctr_clamp
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        imp_error = torch.abs(p_imp-loss_cvr) 
        imp_error_2 = imp_error * imp_error
        loss_cvr = torch.mean(p_imp+ips*imp_error+ips*imp_error_2)


        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')

        return loss_ctr, loss_cvr, loss_ctcvr
    
class Basic_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):

        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(loss_cvr*y_ctr)
        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')

        return loss_ctr, loss_cvr, loss_ctcvr
    
class Entire_Space_Basic_Loss(BasicMultiTaskLoss):
    '''
    CVR loss is caculated on the exposure space
    '''
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        # CVR loss is caculated on the exposure space
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(loss_cvr)
        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')

        return loss_ctr, loss_cvr, loss_ctcvr