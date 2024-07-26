from typing import Any, Dict, Tuple, List
from torchmetrics.classification import BinaryAUROC # type: ignore
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.common import MultiLayerPerceptron, AlldataEmbeddingLayer, BatchTransform
from torchmetrics.classification import AUROC, BinaryAUROC
from tllib.translation.cyclegan.loss import LeastSquaresGenerativeAdversarialLoss, VanillaGenerativeAdversarialLoss,    WassersteinGenerativeAdversarialLoss
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss
import itertools

def set_requires_grad(net, requires_grad=False):
    """
    Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    """
    for param in net.parameters():
        param.requires_grad = requires_grad
        
class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims=256, hidden_dims=64, output_dims=1):
        """Init discriminator."""
        super().__init__()

        # self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dims, output_dims),
            nn.Sigmoid()
        )
        # self.layer = nn.Sequential(
        #     nn.Linear(input_dims, hidden_dims),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dims, output_dims)
        # )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

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
                )
                for _ in range(self.expert_num)
            ]
        )

        self.gate = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.input_dim, self.expert_num),
                    torch.nn.Softmax(dim=1),
                )
                for _ in range(self.task_num)
            ]
        )
        self.ctr_tower_fea = MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                    output_layer=False,
                )
        self.cvr_tower_fea = MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                    output_layer=False,
                )
        self.ctr_tower_pred = nn.Linear(self.tower_dims[-1], 1)
        self.cvr_tower_pred = nn.Linear(self.tower_dims[-1], 1)

    def forward(self, x):
        fea = torch.cat([self.expert[i](x).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        gate_value = [self.gate[i](x).unsqueeze(1) for i in range(self.task_num)]
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        ctr_tower_fea = self.ctr_tower_fea(task_fea[0])
        cvr_tower_fea = self.cvr_tower_fea(task_fea[1])
        pctr = torch.sigmoid(self.ctr_tower_pred(ctr_tower_fea).squeeze(1))
        pcvr = torch.sigmoid(self.cvr_tower_pred(cvr_tower_fea).squeeze(1))
        results = [pctr, pcvr]
        tower_fea = [ctr_tower_fea, cvr_tower_fea]
        return results, task_fea, tower_fea

class Escm(nn.Module):
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
        self.embed_output_dim = embedding_layer.embed_output_dim
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_dims = expert_dims
        self.expert_dropout = expert_dropout
        self.tower_dims = tower_dims
        self.tower_dropout = tower_dropout
            
        self.experts_gates_towers = MmoeFea(
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
        results, task_fea, tower_fea = self.experts_gates_towers(feature_embedding)
        return results[0], results[1], results[0]*results[1], feature_embedding, task_fea, tower_fea
    
class EscmAddPctr(nn.Module):
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
        self.embed_output_dim = embedding_layer.embed_output_dim + 1
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_dims = expert_dims
        self.expert_dropout = expert_dropout
        self.tower_dims = tower_dims
        self.tower_dropout = tower_dropout
            
        self.experts_gates_towers = MmoeFea(
            input_dim=self.embed_output_dim,
            expert_num=self.expert_num, 
            task_num=self.task_num,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        # self.pctr_embedding = nn.Linear(1, 5)
        
        
    def forward(self, x, pctr):
        feature_embedding = self.embedding_layer(x)
        # pctr_embedding = self.pctr_embedding(pctr)
        feature_embedding = torch.cat((feature_embedding, pctr), dim=1)
        results, task_fea, tower_fea = self.experts_gates_towers(feature_embedding)
        return results[0], results[1], results[0]*results[1], feature_embedding, task_fea, tower_fea


class EscmCdanReverseLitModel(pl.LightningModule):
    '''
    使用ReverseLayer的方式进行DA
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
        self.da_loss = DomainAdversarialLoss(DomainDiscriminator(in_feature=self.model.embed_output_dim, hidden_size=64))
        self.batch_transform = BatchTransform(batch_type)
        self.ctr_auc = BinaryAUROC()
        self.cvr_auc = BinaryAUROC()
        self.ctcvr_auc = BinaryAUROC()
        self.trained_model = EscmLitModel.load_from_checkpoint('logs/train/multiruns/fr_ctr_base/checkpoints/fr + src.models.escm_cdan.Escm + src.models.common.Basic_Loss + "epoch_000".ckpt', model=Escm(AlldataEmbeddingLayer(batch_type='fr')))
        self.trained_model.eval()
        self.trained_model.freeze()
        
    
    def training_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred_trained, conversion_pred_trained, click_conversion_pred_trained, feature_embedding_trained, task_fea_trained, tower_fea_trained = self.trained_model.model(features)
        click_pred, conversion_pred, click_conversion_pred, representations, task_fea, tower_fea = self.model(features, click_pred_trained.detach().unsqueeze(1))
        
        # oversampling and caculate da loss
        source_representations = representations[click==1]
        target_representations = representations[click==0]
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
        click_pred_trained, conversion_pred_trained, click_conversion_pred_trained, feature_embedding_trained, task_fea_trained, tower_fea_trained = self.trained_model.model(features)
        click_pred, conversion_pred, click_conversion_pred, representations, task_fea, tower_fea = self.model(features, click_pred_trained.detach().unsqueeze(1))
        val_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred_trained, conversion_pred_trained, click_conversion_pred_trained, feature_embedding_trained, task_fea_trained, tower_fea_trained = self.trained_model.model(features)
        click_pred, conversion_pred, click_conversion_pred, representations, task_fea, tower_fea = self.model(features, click_pred_trained.detach().unsqueeze(1))
        
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.ctr_auc.update(click_pred, click)
        self.cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.ctcvr_auc.update(click_conversion_pred, click * conversion)

    def configure_optimizers(self):    
        # define optimizer and lr scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    
class EscmCdanGanLitModel(pl.LightningModule):
    '''
    使用GAN的训练进行DA
    '''
    def __init__(self, 
                 model:torch.nn.Module,
                 loss:torch.nn.Module, 
                 lr:float, 
                 weight_decay:float=1, 
                 batch_type:str='fr'):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.automatic_optimization = False
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.discriminator = Discriminator(input_dims=95, hidden_dims=64, output_dims=1)
        self.batch_transform = BatchTransform(batch_type)
        self.ctr_auc = BinaryAUROC()
        self.cvr_auc = BinaryAUROC()
        self.ctcvr_auc = BinaryAUROC()
        self.trained_model = EscmLitModel.load_from_checkpoint('logs/train/multiruns/fr_ctr_base/checkpoints/last.ckpt', model=Escm())
        self.trained_model.eval()
        self.trained_model.freeze()
        
        

    
    def training_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred_trained, conversion_pred_trained, click_conversion_pred_trained, feature_embedding_trained, task_fea_trained, tower_fea_trained = self.trained_model.model(features)
        click_pred, conversion_pred, click_conversion_pred, representations, task_fea, tower_fea = self.model(features, click_pred_trained.detach().unsqueeze(1))
        
        # oversampling and caculate da loss
        source_representations = representations[click==1]
        target_representations = representations[click==0]
        source_sampling_weight = torch.ones(len(source_representations))
        indices = torch.multinomial(source_sampling_weight, len(click[click==0]), replacement=True)
        real_S = source_representations[indices]
        real_T = target_representations
        
        optimizer_D, optimizer_G = self.optimizers()
        
        self.toggle_optimizer(optimizer_D)
        optimizer_D.zero_grad()
        source_domain_pred = self.discriminator(real_S)
        target_domain_pred = self.discriminator(real_T)
        dis_loss = torch.nn.functional.binary_cross_entropy(source_domain_pred, torch.ones_like(source_domain_pred)) + torch.nn.functional.binary_cross_entropy(target_domain_pred, torch.zeros_like(target_domain_pred))
        
        self.manual_backward(dis_loss, retain_graph=True)
        optimizer_D.step()
        self.untoggle_optimizer(optimizer_D)
        
        self.toggle_optimizer(optimizer_G)
        optimizer_G.zero_grad()
        target_domain_pred = self.discriminator(real_T)
        da_loss = torch.nn.functional.binary_cross_entropy(target_domain_pred, torch.ones_like(target_domain_pred))
        # caculate normal loss
        classification_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)
        loss = classification_loss + da_loss
        self.manual_backward(loss)
        optimizer_G.step()
        self.untoggle_optimizer(optimizer_G)
        
        # caculate the discriminator accuracy
        source_domain_pred = self.discriminator(real_S)
        target_domain_pred = self.discriminator(real_T)
        da_acc = (torch.sum(source_domain_pred>0.5) + torch.sum(target_domain_pred<0.5)) / (len(source_domain_pred) + len(target_domain_pred))
        
        
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        self.log("train/da_loss", da_loss, on_epoch=True, on_step=True)
        self.log("train/classification_loss", classification_loss, on_epoch=True, on_step=True)
        self.log("train/da_acc", da_acc, on_epoch=True, on_step=True)
        
        return loss
        
        
        
    def validation_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred_trained, conversion_pred_trained, click_conversion_pred_trained, feature_embedding_trained, task_fea_trained, tower_fea_trained = self.trained_model.model(features)
        click_pred, conversion_pred, click_conversion_pred, representations, task_fea, tower_fea = self.model(features, click_pred_trained.detach().unsqueeze(1))
        # filter the conversion_pred where click is 0
        val_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred_trained, conversion_pred_trained, click_conversion_pred_trained, feature_embedding_trained, task_fea_trained, tower_fea_trained = self.trained_model.model(features)
        click_pred, conversion_pred, click_conversion_pred, representations, task_fea, tower_fea = self.model(features, click_pred_trained.detach().unsqueeze(1))
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.ctr_auc.update(click_pred, click)
        self.cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.ctcvr_auc.update(click_conversion_pred, click * conversion)

    def configure_optimizers(self):    
        # define optimizer and lr scheduler
        optimizer_G = torch.optim.Adam(itertools.chain(self.discriminator.parameters(), self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer_D, optimizer_G
    

class EscmLitModel(pl.LightningModule):
    '''
    ESCM最基础的model
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