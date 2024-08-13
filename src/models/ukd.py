
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

class Attention(nn.Module):
  def __init__(self, dim=32):
    super(Attention, self).__init__()
    self.dim = dim
    self.q_layer = nn.Linear(dim, dim, bias=False)
    self.k_layer = nn.Linear(dim, dim, bias=False)
    self.v_layer = nn.Linear(dim, dim, bias=False)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, inputs):
    Q = self.q_layer(inputs)
    K = self.k_layer(inputs)
    V = self.v_layer(inputs)
    a = torch.sum(torch.mul(Q, K), -1) / torch.sqrt(torch.tensor(self.dim))
    a = self.softmax(a)
    outputs = torch.sum(torch.mul(torch.unsqueeze(a, -1), V), dim=1)
    return outputs

class EsmmAttFea(torch.nn.Module):
    def __init__(self, input_dim, expert_num, task_num, expert_dims, tower_dims, expert_dropout, tower_dropout):
        super().__init__()
        self.input_dim = input_dim    
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_dims = expert_dims
        self.expert_dropout = expert_dropout
        self.tower_dims = tower_dims
        self.tower_dropout = tower_dropout
        self.info_layer = nn.Sequential(
                            nn.Linear(self.tower_dims[-1], 32), 
                            nn.ReLU(),
                            nn.Dropout(self.tower_dropout[-1]))
        self.attention_layer = Attention(self.tower_dims[-1])
        

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
        cvr_tower_fea = torch.unsqueeze(tower_fea[1], 1)
        info = torch.unsqueeze(self.info_layer(tower_fea[0]), 1)
        ait = self.attention_layer(torch.cat([cvr_tower_fea, info], 1))
        pctr = torch.sigmoid(self.tower_pred[0](tower_fea[0]).squeeze(1))
        pcvr = torch.sigmoid(self.tower_pred[1](ait).squeeze(1))
        results = [pctr, pcvr]
        return results, tower_fea, tower_fea
    
class EsmmOuterProductFea(torch.nn.Module):
    def __init__(self, input_dim, expert_num, task_num, expert_dims, tower_dims, expert_dropout, tower_dropout):
        super().__init__()
        self.input_dim = input_dim    
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_dims = expert_dims
        self.expert_dropout = expert_dropout
        self.tower_dims = tower_dims
        self.tower_dropout = tower_dropout
        self.pctr_embedding = nn.Linear(1, 5)
        

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
        self.tower_pred[1] = nn.Linear(self.tower_dims[-1]*5, 1)

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
        tower_fea = [self.tower_fea[i](x) for i in range(self.task_num)]
        pctr = torch.sigmoid(self.tower_pred[0](tower_fea[0]).squeeze(1))
        pctr_embedding = self.pctr_embedding(pctr.detach().unsqueeze(1))
        ait = self.outer_product(tower_fea[1], pctr_embedding)
        
        pcvr = torch.sigmoid(self.tower_pred[1](ait).squeeze(1))
        results = [pctr, pcvr]
        return results, tower_fea, tower_fea
    
class CvrTeacherMultiTask(torch.nn.Module):
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

        self.mmoe = EsmmFea(
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
    
class CvrTeacherMultiTaskLitModel(pl.LightningModule):
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
        
        self.da_loss = DomainAdversarialLoss(DomainDiscriminator(in_feature=32, hidden_size=64))
    
    def training_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        
       # oversampling and caculate da loss
        source_representations = tower_fea[1][click==1]
        target_representations = tower_fea[1][click==0]
        source_sampling_weight = torch.ones(len(source_representations))
        indices = torch.multinomial(source_sampling_weight, len(click[click==0]), replacement=True)
        real_S = source_representations[indices]
        real_T = target_representations
        da_loss = self.da_loss(real_S, real_T)
        da_acc = self.da_loss.domain_discriminator_accuracy
        
        
        # caculate normal loss
        classification_loss = self.loss.caculate_loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)
        loss = classification_loss + 0.5*da_loss
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        self.log("train/da_loss", da_loss, on_epoch=True, on_step=True)
        self.log("train/classification_loss", classification_loss, on_epoch=True, on_step=True)
        self.log("train/da_acc", da_acc, on_epoch=True, on_step=True)
        
        return loss
        
        
    def validation_step(self, batch, batch_idx):        
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.val_ctr_auc.update(click_pred, click)
        self.val_cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.val_ctcvr_auc.update(click_conversion_pred, click * conversion)
        val_loss = self.loss.caculate_loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.ctr_auc.update(click_pred, click)
        self.cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.ctcvr_auc.update(click_pred*conversion_pred, click * conversion)
        self.cvr_unclick_auc.update(conversion_pred[click == 0], conversion[click == 0])
        self.cvr_exposure_auc.update(conversion_pred, conversion)

    def configure_optimizers(self):    
        # define optimizer and lr scheduler
        optimizer = torch.optim.Adam(itertools.chain(self.model.parameters(), self.da_loss.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class CvrTeacherMultiTaskLoss(nn.Module):
    def __init__(self, 
                 ctr_loss_proportion: float = 0.0, 
                 cvr_loss_proportion: float = 1, 
                 ):
        super().__init__()
        self.ctr_loss_proportion = 0.0
        self.cvr_loss_proportion = cvr_loss_proportion
    
    
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr):
        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(loss_cvr*y_ctr)
        loss = self.ctr_loss_proportion*loss_ctr + self.cvr_loss_proportion*loss_cvr
        return loss

class CvrTeacherSingleTaskLoss(nn.Module):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__()
    
    def caculate_loss(self, p_cvr, y_ctr, y_cvr):

        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(loss_cvr*y_ctr)

        return loss_cvr
    
class CvrTeacherSingleTask(torch.nn.Module):
    def __init__(
        self,
        embedding_layer: AlldataEmbeddingLayer,
        task_num: int = 1,
        expert_num: int = 8,
        expert_dims: List[int] = [256],
        expert_dropout: List[float] = [0.3],
        tower_dims: List[int] = [128, 64, 32],
        tower_dropout: List[float] = [0.1, 0.3, 0.3],
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

        self.mmoe = EsmmFea(
            input_dim=self.embed_output_dim,
            expert_num=self.expert_num, 
            task_num=1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
    
        
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        results, task_fea, tower_fea = self.mmoe(feature_embedding)
        return results[0], feature_embedding, task_fea[0], tower_fea[0]

class CvrTeacherSingleTaskLitModel(pl.LightningModule):
    '''
    Teacher DA Training
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
        self.da_loss = DomainAdversarialLoss(DomainDiscriminator(in_feature=32, hidden_size=64))
        self.batch_transform = BatchTransform(batch_type)
        self.cvr_auc = BinaryAUROC()
        self.cvr_unclick_auc = BinaryAUROC()
        self.cvr_exposure_auc = BinaryAUROC()
        self.val_cvr_auc = BinaryAUROC()

    def training_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
       # oversampling and caculate da loss
        source_representations = tower_fea[click==1]
        target_representations = tower_fea[click==0]
        source_sampling_weight = torch.ones(len(source_representations))
        indices = torch.multinomial(source_sampling_weight, len(click[click==0]), replacement=True)
        real_S = source_representations[indices]
        real_T = target_representations
        da_loss = self.da_loss(real_S, real_T)
        da_acc = self.da_loss.domain_discriminator_accuracy
        # caculate normal loss
        classification_loss = self.loss.caculate_loss(conversion_pred, click, conversion)
        loss = classification_loss + 0.1*da_loss
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        self.log("train/da_loss", da_loss, on_epoch=True, on_step=True)
        self.log("train/classification_loss", classification_loss, on_epoch=True, on_step=True)
        self.log("train/da_acc", da_acc, on_epoch=True, on_step=True)
        
        return loss
        
        
        
    def validation_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.val_cvr_auc.update(conversion_pred_filter, conversion_filter)
        val_loss = self.loss.caculate_loss(conversion_pred, click, conversion)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        
        self.cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.cvr_unclick_auc.update(conversion_pred[click == 0], conversion[click == 0])
        self.cvr_exposure_auc.update(conversion_pred, conversion)

    def configure_optimizers(self):    
        # define optimizer and lr scheduler
        optimizer = torch.optim.Adam(itertools.chain(self.model.parameters(), self.da_loss.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    
class MmoeFeaStudent(torch.nn.Module):
    def __init__(self, input_dim, expert_num, task_num, expert_dims, tower_dims, expert_dropout, tower_dropout):
        super().__init__()
        self.input_dim = input_dim    
        self.task_num = 2
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
                    drop_last_dropout=True,
                )  # Add .cuda() to move the model to GPU
                for _ in range(self.task_num)
        ])
        self.tower_pred = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(self.tower_dims[-1], 1),
                )  # Add .cuda() to move the model to GPU         
    
                for _ in range(self.task_num + 1)
        ])


    def forward(self, x):
        fea = torch.cat([self.expert[i](x).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        gate_value = [self.gate[i](x).unsqueeze(1) for i in range(self.task_num)]
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        tower_fea = [self.tower_fea[i](task_fea[i]) for i in range(self.task_num)]
        results = [torch.sigmoid(self.tower_pred[i](tower_fea[i]).squeeze(1)) for i in range(self.task_num)]
        results.append(torch.sigmoid(self.tower_pred[2](tower_fea[1]).squeeze(1)))
        return results, task_fea, tower_fea

class EsmmFeaStudent(torch.nn.Module):
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
                    drop_last_dropout=True,
                )  # Add .cuda() to move the model to GPU
                for _ in range(self.task_num)
        ])
        self.tower_pred = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(self.tower_dims[-1], 1),
                )  # Add .cuda() to move the model to GPU         
    
                for _ in range(self.task_num + 1)
        ])


    def forward(self, x):
        tower_fea = [self.tower_fea[i](x) for i in range(self.task_num)]
        results = [torch.sigmoid(self.tower_pred[i](tower_fea[i]).squeeze(1)) for i in range(self.task_num)]
        results.append(torch.sigmoid(self.tower_pred[2](tower_fea[1]).squeeze(1)))
        return results, tower_fea, tower_fea
    
class CvrStudentMultiTask(torch.nn.Module):
    def __init__(
        self,
        embedding_layer: AlldataEmbeddingLayer,
        task_num: int = 2,
        expert_num: int = 8,
        expert_dims: List[int] = [256],
        expert_dropout: List[float] = [0.3],
        tower_dims: List[int] = [128, 64, 32],
        tower_dropout: List[float] = [0.1, 0.3, 0.3],
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

        self.mmoe = EsmmFeaStudent(
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
        return results[0], results[1], results[2], torch.mul(results[0], results[1]),feature_embedding, task_fea, tower_fea

class CvrStudentMultiTaskLitModel(pl.LightningModule):

    def __init__(self, 
                 model:torch.nn.Module,
                 loss:torch.nn.Module, 
                 lr:float, 
                 weight_decay:float=1, 
                 batch_type:str='fr'):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        # self.automatic_optimization = False
        cvr_teacher_path = './logs/train/multiruns/ukd_teacher/' +  batch_type + '/last.ckpt'
        self.cvr_teacher = CvrTeacherMultiTaskLitModel.load_from_checkpoint(cvr_teacher_path, model=CvrTeacherMultiTask(embedding_layer=AlldataEmbeddingLayer(batch_type=batch_type)))
        self.cvr_teacher.eval()
        self.cvr_teacher.freeze()
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
        click_pred, conversion_pred, conversion_pred_1, click_conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        click_pred_teacher, conversion_pred_teacher, click_conversion_pred_teacher,feature_embedding_teacher, task_fea_teacher, tower_fea_teacher = self.cvr_teacher.model(features)

        # caculate normal loss
        loss = self.loss.caculate_loss(click_pred, conversion_pred, conversion_pred_1, click_conversion_pred, click, conversion, conversion_pred_teacher)
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        
        return loss
        
        
    def validation_step(self, batch, batch_idx):        
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, conversion_pred_1, click_conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        click_pred_teacher, conversion_pred_teacher, click_conversion_pred_teacher,feature_embedding_teacher, task_fea_teacher, tower_fea_teacher = self.cvr_teacher.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.val_ctr_auc.update(click_pred, click)
        self.val_cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.val_ctcvr_auc.update(click_conversion_pred, click * conversion)
        val_loss = self.loss.caculate_loss(click_pred, conversion_pred, conversion_pred_1, click_conversion_pred, click, conversion, conversion_pred_teacher)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred_0, click_conversion_pred, conversion_pred_1, feature_embedding, task_fea, tower_fea = self.model(features)
        conversion_pred = (conversion_pred_0 + conversion_pred_1) / 2.0
        self.ctr_auc.update(click_pred, click)
        self.cvr_auc.update(conversion_pred[click == 1], conversion[click == 1])
        self.ctcvr_auc.update(click_conversion_pred, click * conversion)
        self.cvr_exposure_auc.update(conversion_pred, conversion)
        self.cvr_unclick_auc.update(conversion_pred[click == 0], conversion[click == 0])

    def configure_optimizers(self):    
        # define optimizer and lr scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
        
class CvrStudentMultiTaskLoss(nn.Module):
    def __init__(self, 
                 ctr_loss_proportion: float = 0.2, 
                 cvr_loss_proportion: float = 1.0, 
                 uncertainty_ratio: float = 100,
                 ):
        super().__init__()
        self.uncertainty_ratio = uncertainty_ratio
        self.ctr_loss_proportion = ctr_loss_proportion
        self.cvr_loss_proportion = cvr_loss_proportion
        
    def kl_divergence(self, p, q, epsilon=1e-6):
        p = torch.clamp(p, epsilon, 1.0 - epsilon)
        q = torch.clamp(q, epsilon, 1.0 - epsilon)
        kl_div = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
        
        return kl_div

    def caculate_loss(self, p_ctr, p_cvr, p_cvr_1, p_ctcvr, y_ctr, y_cvr, y_cvr_t):
        
        kl_div_0 = self.kl_divergence(p_cvr, p_cvr_1)
        kl_div_1 = self.kl_divergence(p_cvr_1, p_cvr)
        uncertainty_weights_0 = torch.exp(-self.uncertainty_ratio*kl_div_0.detach())
        uncertainty_weights_1 = torch.exp(-self.uncertainty_ratio*kl_div_1.detach())
        
        kl_div_loss_0 = torch.mean(kl_div_0)
        kl_div_loss_1 = torch.mean(kl_div_1)

        loss_cvr_click_0 = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr_unclick_0 = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr_t, reduction='none')
        loss_cvr_click_1 = torch.nn.functional.binary_cross_entropy(p_cvr_1, y_cvr, reduction='none')
        loss_cvr_unclick_1 = torch.nn.functional.binary_cross_entropy(p_cvr_1, y_cvr_t, reduction='none')
        loss_cvr_0 = torch.mean(y_ctr*loss_cvr_click_0 + 0.5*(1-y_ctr)*uncertainty_weights_0*loss_cvr_unclick_0)
        loss_cvr_1 = torch.mean(y_ctr*loss_cvr_click_1 + 0.5*(1-y_ctr)*uncertainty_weights_1*loss_cvr_unclick_1)
        
        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        
        loss = self.ctr_loss_proportion*loss_ctr + self.ctr_loss_proportion*(loss_cvr_0 + loss_cvr_1) + kl_div_loss_0 + kl_div_loss_1

        return loss