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



class DESCM_Embedding_Res(torch.nn.Module):
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
        if A_embed_output_dim == 0:
            self.A_embed_output_dim  = self.embedding_layer.get_embed_output_dim()
        else:
            self.A_embed_output_dim = A_embed_output_dim
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_dims = expert_dims
        self.expert_dropout = expert_dropout
        self.tower_dims = tower_dims
        self.tower_dropout = tower_dropout
        self.task_feature_dim = self.A_embed_output_dim
            
        self.mmoe = MMOE(
            input_dim=self.embed_output_dim + self.embedding_layer.embedding_size*2, 
            expert_num=self.expert_num, 
            task_num=self.task_num-1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.ctr_mmoe_0 = MMOE(
            input_dim=self.embed_output_dim, 
            expert_num=self.expert_num, 
            task_num=1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.ctr_mmoe_1 = MMOE(
            input_dim=self.embed_output_dim+self.embedding_layer.embedding_size, 
            expert_num=self.expert_num, 
            task_num=1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        # self.ratio = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.confounder_dense_0 = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense_0.weight.data)
        self.layer_0 = MultiLayerPerceptron(
                    self.embed_output_dim,
                    [512, self.A_embed_output_dim],
                    [0.3, 0.3],
                    output_layer=False,
                )
        self.confounder_dense_1 = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense_1.weight.data)
        self.layer_1 = MultiLayerPerceptron(
                    self.embed_output_dim+self.embedding_layer.embedding_size,
                    [512, self.embed_output_dim+self.embedding_layer.embedding_size],
                    [0.3, 0.3],
                    output_layer=False,
                )
        
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        pctr_0 = self.ctr_mmoe_0(feature_embedding)[0]
        pctr_0 = pctr_0.reshape(-1, 1)
        pctr_0_embedding = self.confounder_dense_0(pctr_0.detach())
        new_embedding_0 = torch.cat((self.layer_0(feature_embedding), pctr_0_embedding), 1)
        pctr_1 = self.ctr_mmoe_1(new_embedding_0)[0]
        pctr_1 = pctr_1.reshape(-1, 1)
        pctr_1_embedding = self.confounder_dense_1(pctr_1.detach())
        new_embedding_1 = torch.cat((self.layer_1(new_embedding_0), pctr_1_embedding), 1)
        results = self.mmoe(new_embedding_1)
        return pctr_1.squeeze(1), results[0], torch.mul(pctr_1.squeeze(1), results[0]), results[1], pctr_0.squeeze(1)
    
class DESCM_Embedding_Res_Simple(torch.nn.Module):
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
        if A_embed_output_dim == 0:
            self.A_embed_output_dim  = self.embedding_layer.get_embed_output_dim()
        else:
            self.A_embed_output_dim = A_embed_output_dim
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_dims = expert_dims
        self.expert_dropout = expert_dropout
        self.tower_dims = tower_dims
        self.tower_dropout = tower_dropout
        self.task_feature_dim = self.A_embed_output_dim
            
        self.mmoe = MMOE(
            input_dim=self.embed_output_dim + self.embedding_layer.embedding_size*2, 
            expert_num=self.expert_num, 
            task_num=self.task_num-1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.ctr_mmoe_0 = MMOE(
            input_dim=self.embed_output_dim, 
            expert_num=self.expert_num, 
            task_num=1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.ctr_mmoe_1 = MMOE(
            input_dim=self.embed_output_dim+self.embedding_layer.embedding_size, 
            expert_num=self.expert_num, 
            task_num=1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        # self.ratio = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.confounder_dense_0 = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense_0.weight.data)
        self.confounder_dense_1 = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense_1.weight.data)

        
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        pctr_0 = self.ctr_mmoe_0(feature_embedding)[0]
        pctr_0 = pctr_0.reshape(-1, 1)
        pctr_0_embedding = self.confounder_dense_0(pctr_0.detach())
        new_embedding_0 = torch.cat((feature_embedding, pctr_0_embedding), 1)
        pctr_1 = self.ctr_mmoe_1(new_embedding_0)[0]
        pctr_1 = pctr_1.reshape(-1, 1)
        pctr_1_embedding = self.confounder_dense_1(pctr_1.detach())
        new_embedding_1 = torch.cat((new_embedding_0, pctr_1_embedding), 1)
        results = self.mmoe(new_embedding_1)
        return pctr_1.squeeze(1), results[0], torch.mul(pctr_1.squeeze(1), results[0]), results[1], pctr_0.squeeze(1)
    
class CrossNetwork(nn.Module):
    def __init__(self, input_dim, layer_num=2):
        super(CrossNetwork, self).__init__()
        self.cross_layers = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(layer_num)])

    def forward(self, x, x0):
        for layer in self.cross_layers:
            x = x0 * layer(x) + x
        return x
class DESCM_Embedding_Res_Cross(torch.nn.Module):
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
        if A_embed_output_dim == 0:
            self.A_embed_output_dim  = self.embedding_layer.get_embed_output_dim()
        else:
            self.A_embed_output_dim = A_embed_output_dim
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_dims = expert_dims
        self.expert_dropout = expert_dropout
        self.tower_dims = tower_dims
        self.tower_dropout = tower_dropout
        self.task_feature_dim = self.A_embed_output_dim
            
        self.mmoe = MMOE(
            input_dim=self.embed_output_dim + self.embedding_layer.embedding_size, 
            expert_num=self.expert_num, 
            task_num=self.task_num-1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.ctr_mmoe_0 = MMOE(
            input_dim=self.embed_output_dim, 
            expert_num=self.expert_num, 
            task_num=1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.ctr_mmoe_1 = MMOE(
            input_dim=self.embed_output_dim, 
            expert_num=self.expert_num, 
            task_num=1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        # self.ratio = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.confounder_dense_0 = torch.nn.Linear(1, self.embed_output_dim)
        torch.nn.init.xavier_uniform_(self.confounder_dense_0.weight.data)
        self.confounder_dense_1 = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense_1.weight.data)
        
        self.cross = CrossNetwork(self.embed_output_dim, layer_num=2)

        
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        pctr_0 = self.ctr_mmoe_0(feature_embedding)[0]
        pctr_0 = pctr_0.reshape(-1, 1)
        pctr_0_embedding = self.confounder_dense_0(pctr_0.detach())
        new_embedding_0 = self.cross(feature_embedding, pctr_0_embedding)
        pctr_1 = self.ctr_mmoe_1(new_embedding_0)[0]
        pctr_1 = pctr_1.reshape(-1, 1)
        pctr_1_embedding = self.confounder_dense_1(pctr_1.detach())
        new_embedding_1 = torch.cat((new_embedding_0, pctr_1_embedding), 1)
        results = self.mmoe(new_embedding_1)
        return pctr_1.squeeze(1), results[0], torch.mul(pctr_1.squeeze(1), results[0]), results[1], pctr_0.squeeze(1)

class DESCM_Embedding_Res_M1(torch.nn.Module):
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
        if A_embed_output_dim == 0:
            self.A_embed_output_dim  = self.embedding_layer.get_embed_output_dim()
        else:
            self.A_embed_output_dim = A_embed_output_dim
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_dims = expert_dims
        self.expert_dropout = expert_dropout
        self.tower_dims = tower_dims
        self.tower_dropout = tower_dropout
        self.task_feature_dim = self.A_embed_output_dim
            
        self.mmoe = MMOE(
            input_dim=self.embed_output_dim + self.embedding_layer.embedding_size*2, 
            expert_num=self.expert_num, 
            task_num=self.task_num-1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.ctr_mmoe_0 = MMOE(
            input_dim=self.embed_output_dim, 
            expert_num=self.expert_num, 
            task_num=1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.ctr_mmoe_1 = MMOE(
            input_dim=self.embed_output_dim+self.embedding_layer.embedding_size, 
            expert_num=self.expert_num, 
            task_num=1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        # self.ratio = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.confounder_dense_0 = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense_0.weight.data)
        
        self.confounder_dense_1 = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense_1.weight.data)
        
        self.layer_1 = MultiLayerPerceptron(    
                    self.embed_output_dim+self.embedding_layer.embedding_size,
                    [512, self.A_embed_output_dim+self.embedding_layer.embedding_size],
                    [0.3, 0.3],
                    output_layer=False,
                )
        

        
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        pctr_0 = self.ctr_mmoe_0(feature_embedding)[0]
        pctr_0 = pctr_0.reshape(-1, 1)
        pctr_0_embedding = self.confounder_dense_0(pctr_0.detach())
        new_embedding_0 = torch.cat((feature_embedding, pctr_0_embedding), 1)
        pctr_1 = self.ctr_mmoe_1(new_embedding_0)[0]
        pctr_1 = pctr_1.reshape(-1, 1)
        pctr_1_embedding = self.confounder_dense_1(pctr_1.detach())
        new_embedding_1 = torch.cat((self.layer_1(new_embedding_0), pctr_1_embedding), 1)
        results = self.mmoe(new_embedding_1)
        return pctr_1.squeeze(1), results[0], torch.mul(pctr_1.squeeze(1), results[0]), results[1], pctr_0.squeeze(1)
    
class DESCM_Embedding_Res_M0(torch.nn.Module):
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
        if A_embed_output_dim == 0:
            self.A_embed_output_dim  = self.embedding_layer.get_embed_output_dim()
        else:
            self.A_embed_output_dim = A_embed_output_dim
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_dims = expert_dims
        self.expert_dropout = expert_dropout
        self.tower_dims = tower_dims
        self.tower_dropout = tower_dropout
        self.task_feature_dim = self.A_embed_output_dim
            
        self.mmoe = MMOE(
            input_dim=self.embed_output_dim + self.embedding_layer.embedding_size*2, 
            expert_num=self.expert_num, 
            task_num=self.task_num-1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.ctr_mmoe_0 = MMOE(
            input_dim=self.embed_output_dim, 
            expert_num=self.expert_num, 
            task_num=1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.ctr_mmoe_1 = MMOE(
            input_dim=self.embed_output_dim+self.embedding_layer.embedding_size, 
            expert_num=self.expert_num, 
            task_num=1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        # self.ratio = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.confounder_dense_0 = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense_0.weight.data)
        self.layer_0 = MultiLayerPerceptron(
                    self.embed_output_dim,
                    [512, self.A_embed_output_dim],
                    [0.3, 0.3],
                    output_layer=False,
                )
        
        self.confounder_dense_1 = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense_1.weight.data)
        
        

        
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        pctr_0 = self.ctr_mmoe_0(feature_embedding)[0]
        pctr_0 = pctr_0.reshape(-1, 1)
        pctr_0_embedding = self.confounder_dense_0(pctr_0.detach())
        new_embedding_0 = torch.cat((self.layer_0(feature_embedding), pctr_0_embedding), 1)
        pctr_1 = self.ctr_mmoe_1(new_embedding_0)[0]
        pctr_1 = pctr_1.reshape(-1, 1)
        pctr_1_embedding = self.confounder_dense_1(pctr_1.detach())
        new_embedding_1 = torch.cat((new_embedding_0, pctr_1_embedding), 1)
        results = self.mmoe(new_embedding_1)
        return pctr_1.squeeze(1), results[0], torch.mul(pctr_1.squeeze(1), results[0]), results[1], pctr_0.squeeze(1)
    
class DESCM_Embedding_Res_All(torch.nn.Module):
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
        if A_embed_output_dim == 0:
            self.A_embed_output_dim  = self.embedding_layer.get_embed_output_dim()
        else:
            self.A_embed_output_dim = A_embed_output_dim
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_dims = expert_dims
        self.expert_dropout = expert_dropout
        self.tower_dims = tower_dims
        self.tower_dropout = tower_dropout
        self.task_feature_dim = self.A_embed_output_dim
            

        self.mmoe_0 = MMOE(
            input_dim=self.embed_output_dim, 
            expert_num=self.expert_num, 
            task_num=self.task_num,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.mmoe_1 = MMOE(
            input_dim=self.embed_output_dim+self.embedding_layer.embedding_size*2, 
            expert_num=self.expert_num, 
            task_num=self.task_num,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.mmoe = MMOE(
            input_dim=self.embed_output_dim + self.embedding_layer.embedding_size*4, 
            expert_num=self.expert_num, 
            task_num=self.task_num,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        # self.ratio = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.confounder_dense_0 = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense_0.weight.data)
        self.confounder_dense_1 = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense_0.weight.data)
        self.confounder_dense_2 = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense_1.weight.data)
        self.confounder_dense_3 = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense_1.weight.data)

        
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        results_0 = self.mmoe_0(feature_embedding)
        pctr_0 = results_0[0]
        pctr_0 = pctr_0.reshape(-1, 1)
        pcvr_0 = results_0[1]
        pcvr_0 = pcvr_0.reshape(-1, 1)
        pctr_0_embedding = self.confounder_dense_0(pctr_0.detach())
        pcvr_0_embedding = self.confounder_dense_1(pcvr_0.detach())
        new_embedding_0 = torch.cat((feature_embedding, pctr_0_embedding, pcvr_0_embedding), 1)
        results_1 = self.mmoe_1(new_embedding_0)
        pctr_1 = results_1[0]
        pctr_1 = pctr_1.reshape(-1, 1)
        p_cvr_1 = results_1[1]
        p_cvr_1 = p_cvr_1.reshape(-1, 1)
        pctr_1_embedding = self.confounder_dense_2(pctr_1.detach())
        pcvr_1_embedding = self.confounder_dense_3(p_cvr_1.detach())
        new_embedding_1 = torch.cat((new_embedding_0, pctr_1_embedding, pcvr_1_embedding), 1)
        results = self.mmoe(new_embedding_1)
        return results[0], results[1], torch.mul(results[0], results[1]), results[2], results_0, results_1

class MultiTaskLitModel_Res(pl.LightningModule):
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
        click_pred, conversion_pred, click_conversion_pred, imp_pred, click_pred_0 = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion, p_imp=imp_pred, p_ctr_0=click_pred_0)
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, click_pred_0 = self.model(features)
        # filter the conversion_pred where click is 0
        val_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion, p_ctr_0=click_pred_0)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, click_pred_0 = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.ctr_auc.update(click_pred, click)
        self.cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.ctcvr_auc.update(click_conversion_pred, click * conversion)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
class MultiTaskLitModel_Res_All(pl.LightningModule):
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
        click_pred, conversion_pred, click_conversion_pred, imp_pred, results_0, results_1 = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion, p_imp=imp_pred, results_0=results_0, results_1=results_1)
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, results_0, results_1 = self.model(features)
        # filter the conversion_pred where click is 0
        val_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion, p_imp=imp_pred, results_0=results_0, results_1=results_1)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, results_0, results_1 = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.ctr_auc.update(click_pred, click)
        self.cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.ctcvr_auc.update(click_conversion_pred, click * conversion)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class Basic_Loss_Res(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):

        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(loss_cvr*y_ctr)
        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean') + torch.nn.functional.binary_cross_entropy(kwargs['p_ctr_0'], y_ctr, reduction='mean')
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')

        return loss_ctr, loss_cvr, loss_ctcvr
    
class Basic_Loss_Res_Ratio(BasicMultiTaskLoss):
    def __init__(self, 
                ctr_loss_proportion: float = 1, 
                cvr_loss_proportion: float = 1, 
                ctcvr_loss_proportion: float = 0.1,
                trade_off_ctr_loss_0: float = 0.5,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
        self.trade_off_ctr_loss_0 = trade_off_ctr_loss_0
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):

        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(loss_cvr*y_ctr)
        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        loss_ctr += torch.nn.functional.binary_cross_entropy(kwargs['p_ctr_0'], y_ctr, reduction='mean') * self.trade_off_ctr_loss_0
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        return loss_ctr, loss_cvr, loss_ctcvr
    
class Basic_Loss_Res_All(BasicMultiTaskLoss):
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
        
        p_ctr_0 = kwargs['results_0'][0]
        p_ctr_1 = kwargs['results_1'][0]
        loss_ctr += torch.nn.functional.binary_cross_entropy(p_ctr_0, y_ctr, reduction='mean')
        loss_ctr += torch.nn.functional.binary_cross_entropy(p_ctr_1, y_ctr, reduction='mean')
        p_cvr_0 = kwargs['results_0'][1]
        p_cvr_1 = kwargs['results_1'][1]
        loss_cvr += torch.nn.functional.binary_cross_entropy(p_cvr_0, y_cvr, reduction='mean')
        loss_cvr += torch.nn.functional.binary_cross_entropy(p_cvr_1, y_cvr, reduction='mean')
        p_ctcvr_0 = kwargs['results_0'][0] * kwargs['results_0'][1]
        p_ctcvr_1 = kwargs['results_1'][0] * kwargs['results_1'][1]
        loss_ctcvr += torch.nn.functional.binary_cross_entropy(p_ctcvr_0, y_ctr * y_cvr, reduction='mean')
        loss_ctcvr += torch.nn.functional.binary_cross_entropy(p_ctcvr_1, y_ctr * y_cvr, reduction='mean')

        return loss_ctr, loss_cvr, loss_ctcvr
    
class Basic_Loss_Res_1(BasicMultiTaskLoss):
    # 在最后一层添加CTR Loss，但是得使用ALL的结构
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
        
        p_ctr_0 = kwargs['results_0'][0]
        p_ctr_1 = kwargs['results_1'][0]
        loss_ctr += torch.nn.functional.binary_cross_entropy(p_ctr_0, y_ctr, reduction='mean')
        loss_ctr += torch.nn.functional.binary_cross_entropy(p_ctr_1, y_ctr, reduction='mean')
        
        return loss_ctr, loss_cvr, loss_ctcvr
    
    
class Basic_Loss_Res_Indentity(BasicMultiTaskLoss):
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
        loss_ctr += torch.nn.functional.binary_cross_entropy(kwargs['p_ctr_0'], y_ctr, reduction='mean')
        loss_ctr += torch.nn.functional.binary_cross_entropy(p_ctr, kwargs['p_ctr_0'], reduction='mean')
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')

        return loss_ctr, loss_cvr, loss_ctcvr
    
