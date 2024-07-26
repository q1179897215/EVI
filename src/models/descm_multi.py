


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

class DescmEmbeddingBase(torch.nn.Module):
    def __init__(
        self,
        embedding_layer: AlldataEmbeddingLayer,
        task_num: int = 2,
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
        return results[0], results[1], torch.mul(results[0], results[1]),feature_embedding, task_fea, tower_fea

class DescmEmbeddingConcatenation(torch.nn.Module):
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
            input_dim=self.embed_output_dim+self.embedding_layer.embedding_size,
            expert_num=self.expert_num, 
            task_num=self.task_num,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        
        self.ctr_mmoe= MmoeFea(
            input_dim=self.embed_output_dim,
            expert_num=self.expert_num, 
            task_num=1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.ratio = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.confounder_dense = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense.weight.data)
    
        
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        pctr_results, _, _  = self.ctr_mmoe(feature_embedding)
        pctr = pctr_results[0].reshape(-1, 1)
        pctr_embedding = self.confounder_dense(pctr.detach())
        feature_embedding = torch.cat((feature_embedding, pctr_embedding), 1)
        results, task_fea, tower_fea  = self.mmoe(feature_embedding)
        results[0] = pctr.squeeze(1)
        return results[0], results[1], torch.mul(results[0], results[1]),feature_embedding, task_fea, tower_fea                  

class DescmEmbeddingOuterProduct(torch.nn.Module):
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
            input_dim=self.embed_output_dim*self.embedding_layer.embedding_size,
            expert_num=self.expert_num, 
            task_num=self.task_num,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        
        self.ctr_mmoe= MmoeFea(
            input_dim=self.embed_output_dim,
            expert_num=self.expert_num, 
            task_num=1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.ratio = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.confounder_dense = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense.weight.data)
    
    def outer_product(self, x, y):
        """
        x: [batch_size, x_dim]
        y: [batch_size, y_dim]
        """
        x = x.unsqueeze(2)
        y = y.unsqueeze(1)
        out = x * y
        out = out.reshape(x.size(0), -1)
        return out
        
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        pctr_results, pctr_task_fea, pctr_tower_fea  = self.ctr_mmoe(feature_embedding)
        pctr = pctr_results[0].reshape(-1, 1)
        pctr_embedding = self.confounder_dense(pctr.detach())
        # pctr_embedding = self.confounder_dense(pctr_tower_fea[0].detach())
        # cacualte the outer product of feature_embedding and pctr_embedding
        feature_embedding = self.outer_product(feature_embedding, pctr_embedding)
        results, task_fea, tower_fea  = self.mmoe(feature_embedding)
        results[0] = pctr.squeeze(1)
        return results[0], results[1], torch.mul(results[0], results[1]),feature_embedding, task_fea, tower_fea
    
class DescmEmbeddingResOuterProduct(torch.nn.Module):
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
            input_dim=self.embed_output_dim*self.embedding_layer.embedding_size*2, 
            expert_num=self.expert_num, 
            task_num=2,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.ctr_mmoe_0 = MmoeFea(
            input_dim=self.embed_output_dim, 
            expert_num=self.expert_num, 
            task_num=1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.ctr_mmoe_1 = MmoeFea(
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
    
    def outer_product(self, x, y):
        """
        x: [batch_size, x_dim]
        y: [batch_size, y_dim]
        """
        x = x.unsqueeze(2)
        y = y.unsqueeze(1)
        out = x * y
        out = out.reshape(x.size(0), -1)
        return out
        
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        pctr_results_0, _, _  = self.ctr_mmoe_0(feature_embedding)
        pctr_0 = pctr_results_0[0].reshape(-1, 1)
        pctr_0_embedding = self.confounder_dense_0(pctr_0.detach())
        new_embedding_0 = torch.cat((feature_embedding, pctr_0_embedding), 1)
        pctr_results_1, _, _  = self.ctr_mmoe_1(new_embedding_0)
        pctr_1 = pctr_results_1[0].reshape(-1, 1)
        pctr_1_embedding = self.confounder_dense_1(pctr_1.detach())
        # cacualte the outer product of feature_embedding and pctr_embedding
        pctr_embedding = torch.cat((pctr_0_embedding, pctr_1_embedding), 1)
        feature_embedding = self.outer_product(feature_embedding, pctr_embedding)
        results, task_fea, tower_fea  = self.mmoe(feature_embedding)
        return (pctr_0.squeeze(1), pctr_1.squeeze(1)), results[1], torch.mul(pctr_1.squeeze(1), results[1]), feature_embedding, task_fea, tower_fea

class DescmEmbeddingLitModel(pl.LightningModule):
    '''
    ESCM 加 Pctr embedding 最基础的model
    '''
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
        self.ctcvr_auc = BinaryAUROC()
        self.val_ctr_auc = BinaryAUROC()
        self.val_cvr_auc = BinaryAUROC()
        self.val_ctcvr_auc = BinaryAUROC()
    
    def training_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        # caculate normal loss
        loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        
        return loss
        
        
    def validation_step(self, batch, batch_idx):        
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        if len(click_pred) == 2:
            click_pred = click_pred[1]
        self.val_ctr_auc.update(click_pred, click)
        self.val_cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.val_ctcvr_auc.update(click_conversion_pred, click * conversion)
        val_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        if len(click_pred) == 2:
            click_pred = click_pred[1]
        self.ctr_auc.update(click_pred, click)
        self.cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.ctcvr_auc.update(click_conversion_pred, click * conversion)

    def configure_optimizers(self):    
        # define optimizer and lr scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
class DescmEmbeddingDaLitModel(pl.LightningModule):
    '''
    DESCM Multilinear + DA 
    '''
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
        self.da_loss = DomainAdversarialLoss(DomainDiscriminator(in_feature=self.model.embed_output_dim*5, hidden_size=64))
        self.batch_transform = BatchTransform(batch_type)
        self.ctr_auc = BinaryAUROC()
        self.cvr_auc = BinaryAUROC()
        self.ctcvr_auc = BinaryAUROC()
    
    def training_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, feature_embedding, task_fea,tower_fea = self.model(features)
       # oversampling and caculate da loss
        source_representations = feature_embedding[click==1]
        target_representations = feature_embedding[click==0]
        source_sampling_weight = torch.ones(len(source_representations))
        indices = torch.multinomial(source_sampling_weight, len(click[click==0]), replacement=True)
        real_S = source_representations[indices]
        real_T = target_representations
        da_loss = self.da_loss(real_S, real_T)
        da_acc = self.da_loss.domain_discriminator_accuracy
        
        
        # caculate normal loss
        classification_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)
        loss = classification_loss + da_loss*0.1
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        self.log("train/da_loss", da_loss, on_epoch=True, on_step=True)
        self.log("train/classification_loss", classification_loss, on_epoch=True, on_step=True)
        self.log("train/da_acc", da_acc, on_epoch=True, on_step=True)
        
        return loss
        
        
        
    def validation_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        # filter the conversion_pred where click is 0
        val_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.ctr_auc.update(click_pred, click)
        self.cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.ctcvr_auc.update(click_conversion_pred, click * conversion)

    def configure_optimizers(self):    
        # define optimizer and lr scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class DescmEmbeddingMmdLitModel(pl.LightningModule):
    '''
    DESCM Multilinear DA model
    '''
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
        # self.da_loss = BatchwiseMMDLoss(kernel_mul=2.0, kernel_num=5, fix_sigma=1.0, sub_batch_size=50)
        self.da_loss = LinearMMDLoss(kernel_mul=2.0, kernel_num=5, fix_sigma=1.0)
        self.batch_transform = BatchTransform(batch_type)
        self.ctr_auc = BinaryAUROC()
        self.cvr_auc = BinaryAUROC()
        self.ctcvr_auc = BinaryAUROC()
        
    
    def training_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, feature_embedding, task_fea,tower_fea = self.model(features)
       # oversampling and caculate da loss
        source_representations = feature_embedding[click==1]
        target_representations = feature_embedding[click==0]
        source_sampling_weight = torch.ones(len(source_representations))
        indices = torch.multinomial(source_sampling_weight, len(click[click==0]), replacement=True)
        real_S = source_representations[indices]
        real_T = target_representations
        da_loss = self.da_loss(real_S, real_T)
        
        
        # caculate normal loss
        classification_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)
        loss = classification_loss + da_loss*0.1
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        self.log("train/da_loss", da_loss, on_epoch=True, on_step=True)
        self.log("train/classification_loss", classification_loss, on_epoch=True, on_step=True)
        
        return loss
        
        
        
    def validation_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        # filter the conversion_pred where click is 0
        val_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.ctr_auc.update(click_pred, click)
        self.cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.ctcvr_auc.update(click_conversion_pred, click * conversion)

    def configure_optimizers(self):    
        # define optimizer and lr scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class BasicResLoss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):

        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(loss_cvr*y_ctr)
        pctr_0, pctr_1 = p_ctr
        loss_ctr_0 = torch.nn.functional.binary_cross_entropy(pctr_0, y_ctr, reduction='mean')
        loss_ctr_1 = torch.nn.functional.binary_cross_entropy(pctr_1, y_ctr, reduction='mean')
        loss_ctr = (loss_ctr_0 + loss_ctr_1) / 2.0
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')

        return loss_ctr, loss_cvr, loss_ctcvr
    
class BasicResEntropyLoss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    
    def caculate_entropy(self, p):
        entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
        entropy_weight = 1. + torch.exp(-entropy)
        return entropy_weight
    
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        pctr_0, pctr_1 = p_ctr
        loss_ctr_0 = torch.nn.functional.binary_cross_entropy(pctr_0, y_ctr, reduction='mean')
        loss_ctr_1 = torch.nn.functional.binary_cross_entropy(pctr_1, y_ctr, reduction='mean')
        loss_ctr = (loss_ctr_0 + loss_ctr_1) / 2.0
        entropy_weight = (self.caculate_entropy(pctr_0) + self.caculate_entropy(pctr_1)) / 2.0
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(entropy_weight*loss_cvr*y_ctr)

        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='none')
        loss_ctcvr = torch.mean(entropy_weight*loss_ctcvr)

        return loss_ctr, loss_cvr, loss_ctcvr
    
class BasicEntropyLoss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    
    def caculate_entropy(self, p):
        entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
        entropy_weight = 1. + torch.exp(-entropy)
        return entropy_weight
    
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        entropy_weight = self.caculate_entropy(p_ctr)
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(entropy_weight*loss_cvr*y_ctr)

        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='none')
        loss_ctcvr = torch.mean(entropy_weight*loss_ctcvr)

        return loss_ctr, loss_cvr, loss_ctcvr
    
class BasicLoss(BasicMultiTaskLoss):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)
    
    
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, kwargs):
        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(loss_cvr*y_ctr)
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='none')
        loss_ctcvr = torch.mean(loss_ctcvr)

        return loss_ctr, loss_cvr, loss_ctcvr