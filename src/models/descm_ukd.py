


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

        self.mmoe = MmoeFea(
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
    
class CvrTeacherMultiTask(torch.nn.Module):
    def __init__(
        self,
        embedding_layer: AlldataEmbeddingLayer=AlldataEmbeddingLayer(),
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
    
class CvrStudentMultiTask(torch.nn.Module):
    def __init__(
        self,
        embedding_layer: AlldataEmbeddingLayer=AlldataEmbeddingLayer(),
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
        self.cvr_teacher = CvrTeacherMultiTaskLitModel.load_from_checkpoint('./logs/train/cvr_teacher_esmm/last.ckpt', model=CvrTeacherMultiTask(embedding_layer=AlldataEmbeddingLayer(batch_type=batch_type)))
        self.cvr_teacher.eval()
        self.cvr_teacher.freeze()
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
        click_pred, conversion_pred, conversion_pred_1, click_conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        click_pred_teacher, conversion_pred_teacher, click_conversion_pred_teacher, feature_embedding_teacher, task_fea_teacher, tower_fea_teacher = self.cvr_teacher.model(features)

        # caculate normal loss
        loss = self.loss.caculate_loss(click_pred, conversion_pred, conversion_pred_1,click_conversion_pred, click, conversion, conversion_pred_teacher)
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        
        return loss
        
        
    def validation_step(self, batch, batch_idx):        
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, conversion_pred_1, click_conversion_pred, feature_embedding, task_fea, tower_fea = self.model(features)
        click_pred_teacher, conversion_pred_teacher, click_conversion_pred_teacher, feature_embedding_teacher, task_fea_teacher, tower_fea_teacher = self.cvr_teacher.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.val_ctr_auc.update(click_pred, click)
        self.val_cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.val_ctcvr_auc.update(click_conversion_pred, click * conversion)
        val_loss = self.loss.caculate_loss(click_pred, conversion_pred, conversion_pred_1, click_conversion_pred, click, conversion, conversion_pred_teacher)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, conversion_pred_1,feature_embedding, task_fea, tower_fea = self.model(features)
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
        self.val_cvr_auc = BinaryAUROC()
        self.cvr_auc = BinaryAUROC()

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
        loss = classification_loss + da_loss
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

    def configure_optimizers(self):    
        # define optimizer and lr scheduler
        optimizer = torch.optim.Adam(itertools.chain(self.model.parameters(), self.da_loss.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

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
        classification_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)
        loss = classification_loss + da_loss
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
        optimizer = torch.optim.Adam(itertools.chain(self.model.parameters(), self.da_loss.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

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
    

        
class UkdLoss(nn.Module):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 uncertainty_ratio: float = 1,
                 
                 ):
        super().__init__()
        self.uncertainty_ratio = uncertainty_ratio
        
    def kl_divergence(self, p, q, epsilon=1e-6):
        p = torch.clamp(p, epsilon, 1.0 - epsilon)
        q = torch.clamp(q, epsilon, 1.0 - epsilon)
        kl_div = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
        
        return kl_div

    def caculate_loss(self, p_ctr, p_cvr, p_cvr_1, p_ctcvr, y_ctr, y_cvr, y_cvr_s):
        
        kl_div = self.kl_divergence(p_cvr, p_cvr_1)
        uncertainty_weights = torch.exp(-self.uncertainty_ratio*kl_div.detach())
        
        kl_div_loss = torch.mean(kl_div)

        loss_cvr_click = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr_unclick = torch.nn.functional.binary_cross_entropy(p_cvr_1, y_cvr_s, reduction='none')
        # loss_cvr = torch.mean(y_ctr*loss_cvr_click+(1-y_ctr)*loss_cvr_unclick*uncertainty_weights)
        loss_cvr = torch.mean(y_ctr*loss_cvr_click)
        
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_ctr * y_cvr, reduction='mean')
        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        
        loss = loss_ctr + loss_cvr + 0.1*loss_ctcvr #+ kl_div_loss

        return loss