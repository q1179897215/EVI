from typing import Any, Dict, Tuple, List
from abc import abstractmethod, ABC
from src.models.common import BatchTransform, MultiLayerPerceptron
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

class CvrMultiTask(torch.nn.Module):
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

        self.mmoe = MmoeFea(
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
        results, task_fea, tower_fea = self.mmoe(feature_embedding)
        return results, feature_embedding, task_fea, tower_fea
    
class CvrMultiTaskLitModel(pl.LightningModule):
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
        self.val_ctcvr_auc = BinaryAUROC()
        
    
    def training_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        results, feature_embedding, task_fea, tower_fea = self.model(features)
        click_pred, conversion_pred, imputation_pred = results
        
        # caculate normal loss
        loss = self.loss.caculate_loss(click_pred, conversion_pred, imputation_pred, click, conversion)

        self.log("train/loss", loss, on_epoch=True, on_step=True)
        
        return loss
        
        
    def validation_step(self, batch, batch_idx):        
        click, conversion, features = self.batch_transform(batch)
        results, feature_embedding, task_fea, tower_fea = self.model(features)
        click_pred, conversion_pred, imputation_pred = results
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.val_ctr_auc.update(click_pred, click)
        self.val_cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.val_ctcvr_auc.update(click_pred*click_pred, click * conversion)
        val_loss = self.loss.caculate_loss(click_pred, conversion_pred, imputation_pred, click, conversion)


        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        results, feature_embedding, task_fea, tower_fea = self.model(features)
        click_pred, conversion_pred, imputation_pred = results
        
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.ctr_auc.update(click_pred, click)
        self.cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.ctcvr_auc.update(click_pred*conversion_pred, click * conversion)
        self.cvr_unclick_auc.update(conversion_pred[click == 0], conversion[click == 0])
        self.cvr_exposure_auc.update(conversion_pred, conversion)

    def configure_optimizers(self):    
        # define optimizer and lr scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class EsmmLoss(nn.Module):
    def __init__(self, 
                 ctr_loss_proportion: float = 1.0, 
                 cvr_loss_proportion: float = 1.0, 
                 ctcvr_loss_proportion: float = 0.2,
                 ):
        super().__init__()
        self.ctr_loss_proportion = ctr_loss_proportion
        self.cvr_loss_proportion = cvr_loss_proportion
        self.ctcvr_loss_proportion = 1.0
    
    
    def caculate_loss(self, p_ctr, p_cvr, p_imp, y_ctr, y_cvr):
        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctr*p_cvr, y_ctr*y_cvr, reduction='mean')
        loss = self.ctr_loss_proportion*loss_ctr + self.ctcvr_loss_proportion*loss_ctcvr
        return loss
    
class IpwLoss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1.0, 
                 cvr_loss_proportion: float = 1.0, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__()
        self.ctr_loss_proportion = ctr_loss_proportion
        self.cvr_loss_proportion = cvr_loss_proportion
        self.ctcvr_loss_proportion = ctcvr_loss_proportion
    def caculate_loss(self, p_ctr, p_cvr, p_imp, y_ctr, y_cvr):
        
        loss_ctr = torch.nn.functional.binary_cross_entropy(
            p_ctr, y_ctr, reduction='mean'
        )

        p_ctr_clamp = torch.clamp(p_ctr, 1e-7, 1.0-1e-7)
        ips = 1.0 / p_ctr_clamp
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(y_ctr*ips*loss_cvr)

        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctr*p_cvr, y_ctr*y_cvr, reduction='mean')
        loss = self.ctr_loss_proportion*loss_ctr + self.cvr_loss_proportion*loss_cvr + self.ctcvr_loss_proportion*loss_ctcvr
        return loss
    
class DrLoss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1.0, 
                 cvr_loss_proportion: float = 1.0, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__()
        self.ctr_loss_proportion = ctr_loss_proportion
        self.cvr_loss_proportion = cvr_loss_proportion
        self.ctcvr_loss_proportion = ctcvr_loss_proportion
    def caculate_loss(self, p_ctr, p_cvr, p_imp, y_ctr, y_cvr):
        p_ctr_clamp = torch.clamp(p_ctr, 1e-7, 1.0-1e-7)
        ips = y_ctr / p_ctr_clamp
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        imp_error = torch.abs(p_imp-loss_cvr) 
        imp_error_2 = imp_error * imp_error
        loss_cvr = torch.mean(p_imp+ips*imp_error+ips*imp_error_2)


        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctr*p_cvr, y_ctr*y_cvr, reduction='mean')
        
        loss = self.ctr_loss_proportion*loss_ctr + self.cvr_loss_proportion*loss_cvr + self.ctcvr_loss_proportion*loss_ctcvr
        return loss
   
    
class MrdrGplLoss(nn.Module):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 bias_var_trade_off: float = 0.5,
                 strength_trade_off: float = 1,
                 ):
        super().__init__()
        self.ctr_loss_proportion = ctr_loss_proportion
        self.cvr_loss_proportion = cvr_loss_proportion
        self.bias_var_trade_off = bias_var_trade_off
        self.strength_trade_off = strength_trade_off
    def caculate_loss(self, p_ctr, p_cvr, p_imp, y_ctr, y_cvr):
        
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        p_ctr_clip = torch.clamp(p_ctr, 1e-7, 1-1e-7)
        ips = y_ctr / p_ctr_clip
        imp_error = torch.abs(p_imp-loss_cvr) 
        
        bias_term = ((y_ctr-2*y_ctr*p_ctr+p_ctr**2) / p_ctr**2) * imp_error**2
        bias_term = torch.sum(bias_term) / len(p_ctr)**2
        variance_term = (y_ctr / p_ctr**2) * imp_error**2
        variance_term = torch.sum(variance_term) / len(p_ctr)**2
        
        loss_ctr = torch.nn.functional.binary_cross_entropy(
            p_ctr, y_ctr, reduction='mean')
        loss_ctr = loss_ctr + self.strength_trade_off*(self.bias_var_trade_off*bias_term + (1-self.bias_var_trade_off)*variance_term)
        imp_error_2 = imp_error * imp_error * ((1 - p_ctr) / p_ctr)
        loss_cvr = torch.mean(p_imp+ips*imp_error+ips*imp_error_2)
        loss = self.ctr_loss_proportion*loss_ctr + self.cvr_loss_proportion*loss_cvr
        return loss
    

    
    
class DrV2Loss(nn.Module):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__()
        self.ctr_loss_proportion = ctr_loss_proportion
        self.cvr_loss_proportion = cvr_loss_proportion
    def caculate_loss(self, p_ctr, p_cvr, p_imp, y_ctr, y_cvr):
        p_ctr_clip = torch.clamp(p_ctr, 1e-7, 1-1e-7)
        ips = y_ctr / p_ctr_clip
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        imp_error = torch.abs(p_imp-loss_cvr) 
        imp_error_2 = imp_error * imp_error
        bmse = (ips - (1-y_ctr) / (1 - p_ctr_clip))*p_cvr
        bmse = torch.mean(bmse)
        bmse = torch.sqrt(bmse * bmse)
        
        loss_cvr = torch.mean(p_imp+ips*imp_error+ips*imp_error_2) + 0.5*bmse


        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        
        loss = self.ctr_loss_proportion*loss_ctr + self.cvr_loss_proportion*loss_cvr
        return loss
    
class BasicLoss(nn.Module):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__()
        self.ctr_loss_proportion = ctr_loss_proportion
        self.cvr_loss_proportion = cvr_loss_proportion
    def caculate_loss(self, p_ctr, p_cvr, p_imp, y_ctr, y_cvr):
        p_ctr_clip = torch.clamp(p_ctr, 1e-7, 1-1e-7)
        ips = y_ctr / p_ctr_clip
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        imp_error = torch.abs(p_imp-loss_cvr) 
        imp_error_2 = imp_error * imp_error
        bmse = (ips - (1-y_ctr) / (1 - p_ctr_clip))*p_cvr
        bmse = torch.mean(bmse)
        bmse = torch.sqrt(bmse * bmse)
        
        loss_cvr = torch.mean(p_imp+ips*imp_error+ips*imp_error_2) + 0.5*bmse


        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        
        loss = self.ctr_loss_proportion*loss_ctr + self.cvr_loss_proportion*loss_cvr
        return loss