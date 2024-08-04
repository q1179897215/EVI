from typing import Any, Dict, Tuple, List
from abc import abstractmethod, ABC
from src.models.common import BatchTransform, MultiLayerPerceptron, MlpLayerFea
import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig
from torch import nn
from torchmetrics.classification import AUROC, Accuracy, BinaryAUROC
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss
from src.models.common import MultiLayerPerceptron, AlldataEmbeddingLayer, BasicMultiTaskLoss
from src.models.descm_cycada import MMOE
from src.losses.mmd_loss import MMDLoss, LinearMMDLoss, BatchwiseMMDLoss
import itertools

class MmoeFea(torch.nn.Module):
    def __init__(self, input_dim, expert_num, task_num, expert_dims, tower_dims, expert_dropout, tower_dropout):
        super().__init__()
        self.input_dim = input_dim    
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_dims = expert_dims
        self.expert_dropout = expert_dropout
        self.tower_dims = tower_dims
        self.tower_dropout = tower_dropout
        self.expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.input_dim,
                    self.expert_dims,
                    self.expert_dropout,
                    output_layer=False,
                ).cuda()  # Add .cuda() to move the model to GPU
                for _ in range(self.expert_num)
            ]
        )

        self.gate = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.input_dim, self.expert_num),
                    torch.nn.Softmax(dim=1),
                )  # Add .cuda() to move the model to GPU
                for _ in range(self.task_num)
            ]
        )
        self.tower_fea = torch.nn.ModuleList([
            MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                    output_layer=False,
                )  # Add .cuda() to move the model to GPU
                for _ in range(self.task_num)
        ])
        self.tower_pred = torch.nn.ModuleList([
                nn.Linear(self.tower_dims[-1], 1) # Add .cuda() to move the model to GPU
                for _ in range(self.task_num)
        ])


    def forward(self, x):
        fea = torch.cat([self.expert[i](x).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        gate_value = [self.gate[i](x).unsqueeze(1) for i in range(self.task_num)]
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        tower_fea = [self.tower_fea[i](task_fea[i]) for i in range(self.task_num)]
        results = [torch.sigmoid(self.tower_pred[i](tower_fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results, task_fea, tower_fea

class EsmmFea(torch.nn.Module):
    def __init__(self, input_dim, expert_num, task_num, expert_dims, tower_dims, expert_dropout, tower_dropout):
        super().__init__()
        self.input_dim = input_dim    
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_dims = expert_dims
        self.expert_dropout = expert_dropout
        self.tower_dims = tower_dims
        self.tower_dropout = tower_dropout

        self.tower_fea = torch.nn.ModuleList([
            MultiLayerPerceptron(
                    self.input_dim,
                    self.tower_dims,
                    self.tower_dropout,
                    output_layer=False,
                )  # Add .cuda() to move the model to GPU
                for _ in range(self.task_num)
        ])
        self.tower_pred = torch.nn.ModuleList([
                nn.Linear(self.tower_dims[-1], 1) # Add .cuda() to move the model to GPU
                for _ in range(self.task_num)
        ])


    def forward(self, x):
        tower_fea = [self.tower_fea[i](x) for i in range(self.task_num)]
        results = [torch.sigmoid(self.tower_pred[i](tower_fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results, tower_fea, tower_fea

class Dcmt(torch.nn.Module):
    def __init__(
        self,
        embedding_layer: AlldataEmbeddingLayer,
        task_num: int = 3,
        expert_num: int = 8,
        expert_dims: List[int] = [256],
        expert_dropout: List[float] = [0.3],
        tower_dims: List[int] = [128, 64, 32],
        tower_dropout: List[float] = [0.1, 0.3, 0.3],
        A_embed_output_dim: int = 0,
    ):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.embed_output_dim = self.embedding_layer.get_embed_output_dim()
        # expert, gates and towers
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_dims = expert_dims
        self.expert_dropout = expert_dropout
        self.tower_dims = tower_dims
        self.tower_dropout = tower_dropout

        self.extractor_and_predictor = EsmmFea(
            input_dim=self.embed_output_dim,
            expert_num=self.expert_num,
            task_num=self.task_num,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
    
        
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        results, task_fea, tower_fea = self.extractor_and_predictor(feature_embedding)
        return results, feature_embedding, task_fea, tower_fea

class DcmtLitModel(pl.LightningModule):
    def __init__(self, 
                 model:torch.nn.Module,
                 loss:torch.nn.Module, 
                 lr:float, 
                 weight_decay:float=1, 
                 batch_type:str='fr'):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        # self.automatic_optimization = False
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_transform = BatchTransform(batch_type)
        self.ctr_auc = BinaryAUROC()
        self.cvr_auc = BinaryAUROC()
        self.cvr_unclick_auc = BinaryAUROC()
        self.cvr_exposure_auc = BinaryAUROC()
        self.ctcvr_auc = BinaryAUROC()
        self.val_ctr_auc = BinaryAUROC()
        self.val_cvr_auc = BinaryAUROC()
        self.val_cvr_unclick_auc = BinaryAUROC()
        self.val_cvr_exposure_auc = BinaryAUROC()
        self.val_ctcvr_auc = BinaryAUROC()
        
    
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        self.toggle_optimizer(optimizer)
        optimizer.zero_grad()
        
        click, conversion, features = self.batch_transform(batch)
        results, feature_embedding, task_fea, tower_fea = self.model(features)
        click_pred, conversion_pred, click_conversion_pred = results[0], results[1], results[2]
        
        
        # caculate normal loss
        loss = self.loss.caculate_loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        return loss
                
        
        
    def validation_step(self, batch, batch_idx):        
        click, conversion, features = self.batch_transform(batch)
        results, feature_embedding, task_fea, tower_fea = self.model(features)
        click_pred, conversion_pred, click_conversion_pred = results[0], results[1], results[2]
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.val_ctr_auc.update(click_pred, click)
        self.val_cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.val_ctcvr_auc.update(click_conversion_pred, click * conversion)
        self.val_cvr_unclick_auc.update(conversion_pred[click == 0], conversion[click == 0])
        self.val_cvr_exposure_auc.update(conversion_pred, conversion)
        val_loss = self.loss.caculate_loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        results, feature_embedding, task_fea, tower_fea = self.model(features)
        click_pred, conversion_pred, click_conversion_pred = results[0], results[1], results[2]
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.ctr_auc.update(click_pred, click)
        self.cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.ctcvr_auc.update(click_conversion_pred, click * conversion)
        self.cvr_unclick_auc.update(conversion_pred[click == 0], conversion[click == 0])
        self.cvr_exposure_auc.update(conversion_pred, conversion)

    def configure_optimizers(self):    
        # define optimizer and lr scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class DcmtLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr):
        pctr_clamp = torch.clamp(p_ctr.detach(), 0.001, 1-0.001)
        ips = 1 / pctr_clamp
        non_ips = 1 / (1 - pctr_clamp)
        # self-normalize the ips and non_ips
        ips = ips / torch.sum(ips)
        non_ips = non_ips / torch.sum(non_ips)
        # calculate the loss
        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(ips*y_ctr*loss_cvr)
        loss_noncvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_cvr, reduction='none')
        loss_noncvr = torch.mean(non_ips*(1-y_ctr)*loss_noncvr)
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_cvr*p_ctr, y_cvr, reduction='mean')
        constraint_loss = torch.mean(torch.abs(1 - p_cvr - p_ctcvr))
        loss = loss_ctr + loss_cvr + loss_noncvr + loss_ctcvr + 0.001*constraint_loss
        return loss