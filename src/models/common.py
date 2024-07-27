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

        if hasattr(pl_module, 'val_ctr_auc'):
            self.log("val/ctr_auc", pl_module.val_ctr_auc.compute())
            pval_ctr = self.dim_zero_cat(pl_module.val_ctr_auc.preds)
            pval_ctr_target =  self.dim_zero_cat(pl_module.val_ctr_auc.target)
            self.log("val/ctr_logloss", torch.nn.functional.binary_cross_entropy(pval_ctr, pval_ctr_target))
            pl_module.val_ctr_auc.reset()
        if hasattr(pl_module, 'val_cvr_auc'):
            self.log("val/cvr_auc", pl_module.val_cvr_auc.compute())
            pval_cvr = self.dim_zero_cat(pl_module.val_cvr_auc.preds)
            pval_cvr_target =  self.dim_zero_cat(pl_module.val_cvr_auc.target)
            self.log("val/cvr_logloss", torch.nn.functional.binary_cross_entropy(pval_cvr, pval_cvr_target))
            pl_module.val_cvr_auc.reset()
        if hasattr(pl_module, 'val_ctcvr_auc'):
            self.log("val/ctcvr_auc", pl_module.val_ctcvr_auc.compute())
            pval_ctcvr = self.dim_zero_cat(pl_module.val_ctcvr_auc.preds)
            pval_ctcvr_target =  self.dim_zero_cat(pl_module.val_ctcvr_auc.target)
            self.log("val/ctcvr_logloss", torch.nn.functional.binary_cross_entropy(pval_ctcvr, pval_ctcvr_target))
            pl_module.val_ctcvr_auc.reset()
        if hasattr(pl_module, 'val_cvr_unclick_auc'):
            self.log("val/cvr_unclick_auc", pl_module.val_cvr_unclick_auc.compute())
            pval_cvr_unclick = self.dim_zero_cat(pl_module.val_cvr_unclick_auc.preds)
            pval_cvr_unclick_target =  self.dim_zero_cat(pl_module.val_cvr_unclick_auc.target)
            self.log("val/cvr_unclick_logloss", torch.nn.functional.binary_cross_entropy(pval_cvr_unclick, pval_cvr_unclick_target))
            pl_module.cvr_unclick_auc.reset()
        if hasattr(pl_module, 'val_cvr_exposure_auc'):
            self.log("val/cvr_exposure_auc", pl_module.val_cvr_exposure_auc.compute())
            pval_cvr_exposure = self.dim_zero_cat(pl_module.val_cvr_exposure_auc.preds)
            pval_cvr_exposure_target =  self.dim_zero_cat(pl_module.val_cvr_exposure_auc.target)
            self.log("val/cvr_exposure_logloss", torch.nn.functional.binary_cross_entropy(pval_cvr_exposure, pval_cvr_exposure_target))
            pl_module.val_cvr_exposure_auc.reset()
    
    def on_test_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module, 'ctr_auc'):
            self.log("test/ctr_auc", pl_module.ctr_auc.compute())
            pctr = self.dim_zero_cat(pl_module.ctr_auc.preds)
            pctr_target =  self.dim_zero_cat(pl_module.ctr_auc.target)
            ctr_logloss = torch.nn.functional.binary_cross_entropy(pctr, pctr_target)
            self.log("test/ctr_logloss", ctr_logloss)
            pl_module.ctr_auc.reset()
        if hasattr(pl_module, 'cvr_auc'):
            self.log("test/cvr_auc", pl_module.cvr_auc.compute())
            pcvr = self.dim_zero_cat(pl_module.cvr_auc.preds)
            pcvr_target =  self.dim_zero_cat(pl_module.cvr_auc.target)
            self.log("test/cvr_logloss", torch.nn.functional.binary_cross_entropy(pcvr, pcvr_target))
            pl_module.cvr_auc.reset()
        if hasattr(pl_module, 'ctcvr_auc'):
            self.log("test/ctcvr_auc", pl_module.ctcvr_auc.compute())
            pctcvr = self.dim_zero_cat(pl_module.ctcvr_auc.preds)
            pctcvr_target =  self.dim_zero_cat(pl_module.ctcvr_auc.target)
            self.log("test/ctcvr_logloss", torch.nn.functional.binary_cross_entropy(pctcvr, pctcvr_target))
            pl_module.ctcvr_auc.reset()        
        if hasattr(pl_module, 'cvr_unclick_auc'):
            self.log("test/cvr_unclick_auc", pl_module.cvr_unclick_auc.compute())
            pcvr_unclick = self.dim_zero_cat(pl_module.cvr_unclick_auc.preds)
            pcvr_unclick_target =  self.dim_zero_cat(pl_module.cvr_unclick_auc.target)
            self.log("test/cvr_unclick_logloss", torch.nn.functional.binary_cross_entropy(pcvr_unclick, pcvr_unclick_target))
            pl_module.cvr_unclick_auc.reset()
        if hasattr(pl_module, 'cvr_exposure_auc'):
            self.log("test/cvr_exposure_auc", pl_module.cvr_exposure_auc.compute())
            pcvr_exposure = self.dim_zero_cat(pl_module.cvr_exposure_auc.preds)
            pcvr_exposure_target =  self.dim_zero_cat(pl_module.cvr_exposure_auc.target)
            self.log("test/cvr_exposure_logloss", torch.nn.functional.binary_cross_entropy(pcvr_exposure, pcvr_exposure_target))
            pl_module.cvr_exposure_auc.reset()
        
        # pctr = self.dim_zero_cat(pl_module.ctr_auc.preds)
        # target =  self.dim_zero_cat(pl_module.ctr_auc.target)
        # m = target.sum()
        # n = len(target) - target.sum()
        # p = m / (m + n)
        # q = m / (m + 2*n)
        # self.log("pctr_mean", pctr.mean())
        # self.log("m", m)
        # self.log('n', n)
        # self.log('p', p)
        # self.log('q', q)
        
class MultiTaskCallback_Plot(Callback):
    def __init__(
        self, 
        fig_dir: str = './outputs'
    ):
        super().__init__()
        self.fig_dir = fig_dir 

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
        self.click_pred_test.append(outputs['click_pred_test'])
        self.conversion_pred_test.append(outputs['conversion_pred_test'])
        self.conversion_pred_filter_test.append(outputs['conversion_pred_filter_test'])
        self.click_conversion_pred_test.append(outputs['click_conversion_pred_test'])
        self.click_label_test.append(outputs['click_label_test'])
        self.conversion_label_test.append(outputs['conversion_label_test'])
        self.conversion_label_filter_test.append(outputs['conversion_label_filter_test'])
        self.click_conversion_label_test.append(outputs['click_conversion_label_test'])

    def on_test_epoch_end(self, trainer, pl_module):
        self.log("test/ctr_auc", pl_module.ctr_auc.compute())
        self.log("test/cvr_auc", pl_module.cvr_auc.compute())
        self.log("test/ctcvr_auc", pl_module.ctcvr_auc.compute())
        pl_module.ctr_auc.reset()
        pl_module.cvr_auc.reset()
        pl_module.ctcvr_auc.reset()
        
        self.click_label_test = torch.cat(self.click_label_test).cpu().detach().numpy()
        self.click_pred_test = torch.cat(self.click_pred_test).cpu().detach().numpy()
        self.conversion_pred_test = torch.cat(self.conversion_pred_test).cpu().detach().numpy()
        self.conversion_label_test = torch.cat(self.conversion_label_test).cpu().detach().numpy()
        self.conversion_pred_filter_test = torch.cat(self.conversion_pred_filter_test).cpu().detach().numpy()
        self.conversion_label_filter_test = torch.cat(self.conversion_label_filter_test).cpu().detach().numpy()
        
        propensity_scores = 1 / self.click_pred_test
        propensity_scores_click = 1 / self.click_pred_test[self.click_label_test==1]
        propensity_scores_unclick = 1 / self.click_pred_test[self.click_label_test==0]
        propensity_scores_click_expection = propensity_scores_click.mean()
        
        print('propensity_scores_click_expection', propensity_scores_click_expection)
        print('propensity_scores_click_variance', propensity_scores_click.var())

        
        print('conversion_label_mean', self.conversion_label_test.mean())
        print('conversion_pred_mean', self.conversion_pred_test.mean())
        print('conversion_pred_variance', self.conversion_pred_test.var())
        
        print('conversion_label_filter_mean', self.conversion_label_filter_test.mean())
        print('conversion_pred_filter_mean', self.conversion_pred_filter_test.mean())
        print('conversion_pred_filter_variance', self.conversion_pred_filter_test.var()) 
        
        if os.path.exists(self.fig_dir) == False:
            os.makedirs(self.fig_dir)
            
        
        plt.figure(figsize=(12,9), dpi=200)
        plt.hist(self.conversion_pred_filter_test, bins=100, color='whitesmoke', alpha=1, edgecolor='black', density=True, label='Count')
        # Add a vertical line at the mean
        pred_mean = self.conversion_pred_filter_test.mean()
        pred_var = self.conversion_pred_filter_test.var()
        # var show in scientific notation
        pred_var = '%.2E' % pred_var
        label_mean = self.conversion_label_filter_test.mean()
        increase = (pred_mean - label_mean) / label_mean
        # get round of increase
        if increase > 0:
            increase = '+' + str(round(increase*100, 2))
        else:
            increase = str(round(increase*100, 2))
        increase = increase + '%'
        

        plt.axvline(pred_mean, color='r', linestyle='dashed', linewidth=2, label='Prediction Mean:{:.4f}({}), Variance:{}'.format(pred_mean, increase, pred_var))
        plt.axvline(label_mean, color='g', linestyle='dashed', linewidth=2, label='Label Mean:{:.4f}'.format(label_mean))

        # Display the mean value on the plot
        # plt.text(pred_mean+1, 0.03, 'Mean: {:.2f}'.format(pred_mean), fontsize=20, color='r')

        plt.xlabel('CVR prediction values', fontsize=28, color='black')
        plt.ylabel('Count', fontsize=28, color='black')
        plt.xticks(fontsize=20, color='black')
        plt.yticks(fontsize=20, color='black')
        plt.style.use('fast')
        plt.legend(fontsize=20)
        plt.savefig(self.fig_dir +'/conversion_pred_filter_test' + '.pdf', format='pdf')
        plt.clf()
        
        plt.hist(self.conversion_pred_test, bins=100,  edgecolor='black', alpha=0.5,  color='orange')
        plt.savefig(self.fig_dir+'/conversion_pred_test' + '.png', format='png', dpi=300)
        plt.clf()
        
        plt.hist(self.click_pred_test, bins=100,  edgecolor='black', alpha=0.5,  color='orange')
        plt.savefig(self.fig_dir+'/click_pred_test' + '.png', format='png', dpi=300)
        plt.clf()
        
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
        self.preds = []
        self.target = []

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
        
        return {'click_pred_test': click_pred, 
                'conversion_pred_test': conversion_pred,
                'conversion_pred_filter_test': conversion_pred_filter,
                'click_conversion_pred_test': click_conversion_pred,
                'click_label_test': click,  
                'conversion_label_test': conversion,  
                'conversion_label_filter_test': conversion_filter,
                'click_conversion_label_test': click*conversion} 

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
    
class BasicLoss(BasicMultiTaskLoss):
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
    
class MMoE_Single_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):

        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(loss_cvr*y_ctr)
        loss_ctr = 0
        loss_ctcvr = 0

        return loss_ctr, loss_cvr, loss_ctcvr
    
class MMoE_Multi_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):

        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(loss_cvr*y_ctr)
        loss_ctr =  torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')

        loss_ctcvr = 0

        return loss_ctr, loss_cvr, loss_ctcvr
    
class IpwLoss(BasicMultiTaskLoss):
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
    
class MRDR_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        p_imp = kwargs['p_imp']
        loss_ctr = torch.nn.functional.binary_cross_entropy(
            p_ctr, y_ctr, reduction='mean'
        )

        p_ctr_clamp = torch.clamp(p_ctr, 1e-7, 1.0-1e-7)
        ips = y_ctr / p_ctr_clamp
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        imp_error = torch.abs(p_imp-loss_cvr) 
        imp_error_2 = imp_error * imp_error * ((1 - p_ctr_clamp) / p_ctr_clamp)
        loss_cvr = torch.mean(p_imp+ips*imp_error+ips*imp_error_2)

        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        return loss_ctr, loss_cvr, loss_ctcvr
class DR_BIAS_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        p_imp = kwargs['p_imp']
        loss_ctr = torch.nn.functional.binary_cross_entropy(
            p_ctr, y_ctr, reduction='mean'
        )

        p_ctr_clamp = torch.clamp(p_ctr, 1e-7, 1.0-1e-7)
        ips = y_ctr / p_ctr_clamp
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        imp_error = torch.abs(p_imp-loss_cvr) 
        imp_error_2 = imp_error * imp_error * ((y_ctr - p_ctr_clamp) / p_ctr_clamp)**2
        loss_cvr = torch.mean(p_imp+ips*imp_error+ips*imp_error_2)

        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        return loss_ctr, loss_cvr, loss_ctcvr
class DR_MSE_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        p_imp = kwargs['p_imp']
        loss_ctr = torch.nn.functional.binary_cross_entropy(
            p_ctr, y_ctr, reduction='mean'
        )

        p_ctr_clamp = torch.clamp(p_ctr, 1e-7, 1.0-1e-7)
        ips = y_ctr / p_ctr_clamp
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        imp_error = torch.abs(p_imp-loss_cvr) 
        imp_error_2 = 0.5 * imp_error * imp_error * ((y_ctr - p_ctr_clamp) / p_ctr_clamp)**2 + 0.5 * imp_error * imp_error * ((1 - p_ctr_clamp) / p_ctr_clamp)
        loss_cvr = torch.mean(p_imp+ips*imp_error+ips*imp_error_2)

        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        return loss_ctr, loss_cvr, loss_ctcvr

    
    

class IPS_GPL_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 bias_var_trade_off: float = 0.5,
                 strength_trade_off: float = 1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
        self.bias_var_trade_off = bias_var_trade_off
        self.strength_trade_off = strength_trade_off
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        p_imp = kwargs['p_imp']
        loss_ctr = torch.nn.functional.binary_cross_entropy(
            p_ctr, y_ctr, reduction='mean'
        )
        
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        ips = y_ctr / p_ctr
        bias_term = ((y_ctr-2*y_ctr*p_ctr+p_ctr**2) / p_ctr**2) * loss_cvr**2
        bias_term = torch.sum(bias_term) / len(p_ctr)**2
        variance_term = (y_ctr / p_ctr**2) * loss_cvr**2
        variance_term = torch.sum(variance_term) / len(p_ctr)**2
        
        loss_ctr = loss_ctr + self.strength_trade_off*(self.bias_var_trade_off*bias_term + (1-self.bias_var_trade_off)*variance_term)
        loss_cvr = torch.mean(ips*loss_cvr)
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        
        return loss_ctr, loss_cvr, loss_ctcvr
    
class DR_GPL_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 bias_var_trade_off: float = 0.5,
                 strength_trade_off: float = 1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
        self.bias_var_trade_off = bias_var_trade_off
        self.strength_trade_off = strength_trade_off
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        p_imp = kwargs['p_imp']
        loss_ctr = torch.nn.functional.binary_cross_entropy(
            p_ctr, y_ctr, reduction='mean'
        )
        
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        ips = y_ctr / p_ctr
        imp_error = torch.abs(p_imp-loss_cvr) 

        bias_term = ((y_ctr-2*y_ctr*p_ctr+p_ctr**2) / p_ctr**2) * imp_error**2
        bias_term = torch.sum(bias_term) / len(p_ctr)**2
        variance_term = (y_ctr / p_ctr**2) * imp_error**2
        variance_term = torch.sum(variance_term) / len(p_ctr)**2
        
        loss_ctr = loss_ctr + self.strength_trade_off*(self.bias_var_trade_off*bias_term + (1-self.bias_var_trade_off)*variance_term)
        imp_error_2 = imp_error * imp_error
        loss_cvr = torch.mean(p_imp+ips*imp_error+ips*imp_error_2)
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        
        return loss_ctr, loss_cvr, loss_ctcvr
    
class MRDR_GPL_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 bias_var_trade_off: float = 0.5,
                 strength_trade_off: float = 1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
        self.bias_var_trade_off = bias_var_trade_off
        self.strength_trade_off = strength_trade_off
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        p_imp = kwargs['p_imp']
        loss_ctr = torch.nn.functional.binary_cross_entropy(
            p_ctr, y_ctr, reduction='mean'
        )
        
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        ips = y_ctr / p_ctr
        imp_error = torch.abs(p_imp-loss_cvr) 
        
        bias_term = ((y_ctr-2*y_ctr*p_ctr+p_ctr**2) / p_ctr**2) * imp_error**2
        bias_term = torch.sum(bias_term) / len(p_ctr)**2
        variance_term = (y_ctr / p_ctr**2) * imp_error**2
        variance_term = torch.sum(variance_term) / len(p_ctr)**2
        
        loss_ctr = loss_ctr + self.strength_trade_off*(self.bias_var_trade_off*bias_term + (1-self.bias_var_trade_off)*variance_term)
        imp_error_2 = imp_error * imp_error * ((1 - p_ctr) / p_ctr)
        loss_cvr = torch.mean(p_imp+ips*imp_error+ips*imp_error_2)
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        
        return loss_ctr, loss_cvr, loss_ctcvr
    
    
class DRMSE_GPL_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 bias_var_trade_off: float = 0.5,
                 strength_trade_off: float = 1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
        self.bias_var_trade_off = bias_var_trade_off
        self.strength_trade_off = strength_trade_off
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        p_imp = kwargs['p_imp']
        loss_ctr = torch.nn.functional.binary_cross_entropy(
            p_ctr, y_ctr, reduction='mean'
        )
        
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        ips = y_ctr / p_ctr
        imp_error = torch.abs(p_imp-loss_cvr) 
        
        bias_term = ((y_ctr-2*y_ctr*p_ctr+p_ctr**2) / p_ctr**2) * imp_error**2
        bias_term = torch.sum(bias_term) / len(p_ctr)**2
        variance_term = (y_ctr / p_ctr**2) * imp_error**2
        variance_term = torch.sum(variance_term) / len(p_ctr)**2
        
        loss_ctr = loss_ctr + self.strength_trade_off*(self.bias_var_trade_off*bias_term + (1-self.bias_var_trade_off)*variance_term)
        imp_error_2 = 0.5 * imp_error * imp_error * ((y_ctr - p_ctr) / p_ctr)**2 + 0.5 * imp_error * imp_error * ((1 - p_ctr) / p_ctr)
        loss_cvr = torch.mean(p_imp+ips*imp_error+ips*imp_error_2)
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        
        return loss_ctr, loss_cvr, loss_ctcvr
    
    
class DR_VR_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        p_imp = kwargs['p_imp']
        ips = y_ctr / p_ctr
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        imp_error = torch.abs(p_imp-loss_cvr) 
        imp_error_2 = imp_error * imp_error
        bmse = (ips - (1-y_ctr) / (1-p_ctr))*p_cvr
        bmse = torch.mean(bmse)
        bmse = torch.sqrt(bmse * bmse)
        
        loss_cvr = torch.mean(p_imp+ips*imp_error+ips*imp_error_2) + 0.5*bmse


        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')

        return loss_ctr, loss_cvr, loss_ctcvr
    
class IPS_VR_Loss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        ips = y_ctr / p_ctr
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        bmse = (ips - (1-y_ctr) / (1-p_ctr))*p_cvr
        bmse = torch.mean(bmse)
        bmse = torch.sqrt(bmse * bmse)
        
        loss_cvr = torch.mean(loss_cvr) + 0.5*bmse


        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')

        return loss_ctr, loss_cvr, loss_ctcvr
    

    

    




    
    
    
class SNIPS_Loss(BasicMultiTaskLoss):
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
        # normalize ips
        ips = ips / torch.sum(ips)
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(ips*loss_cvr)

        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        return loss_ctr, loss_cvr, loss_ctcvr
