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
from src.models.common import MultiLayerPerceptron, AlldataEmbeddingLayer
from src.models.descm_cycada import MMOE

class DESCM_Embedding_Old(torch.nn.Module):
    def __init__(
        self,
        embedding_layer: AlldataEmbeddingLayer,
        task_num: int = 3,
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

        self.expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.embed_output_dim+ self.embedding_layer.embedding_size,
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
                    torch.nn.Linear(self.embed_output_dim+ self.embedding_layer.embedding_size, self.expert_num),
                    torch.nn.Softmax(dim=1),
                )
                for _ in range(self.task_num)
            ]
        )
        self.tower = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                )
                for _ in range(self.task_num)
            ]
        )
        
        self.ctr_expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.embed_output_dim,
                    self.expert_dims,
                    self.expert_dropout,
                    output_layer=False,
                )
                for _ in range(self.expert_num)
            ]
        )
        self.ctr_gate = torch.nn.Sequential(
            torch.nn.Linear(self.embed_output_dim, self.expert_num), torch.nn.Softmax(dim=1)
        )
        self.ctr_tower = MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                )

        self.ratio = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.confounder_dense = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense.weight.data)
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        ctr_fea = torch.cat([self.ctr_expert[i](feature_embedding).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        ctr_gate_value = self.ctr_gate(feature_embedding).unsqueeze(1) 
        ctr_task_fea = torch.bmm(ctr_gate_value, ctr_fea).squeeze(1)
        pctr = torch.sigmoid(self.ctr_tower(ctr_task_fea).squeeze(1))
        pctr = pctr.reshape(-1, 1)
        pctr_embedding = self.confounder_dense(pctr)
        feature_embedding = torch.cat((feature_embedding, pctr_embedding), 1)
        
        fea = torch.cat([self.expert[i](feature_embedding).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        gate_value = [self.gate[i](feature_embedding).unsqueeze(1) for i in range(self.task_num)]
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        results[0] = pctr.squeeze(1)
        task_fea[0] = ctr_task_fea
        return results[0], results[1], torch.mul(results[0], results[1]), results[2], task_fea[0]                      

class DESCM_Embedding_Stop_Gradients(torch.nn.Module):
    def __init__(
        self,
        embedding_layer: AlldataEmbeddingLayer,
        task_num: int = 3,
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

        self.expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.embed_output_dim+ self.embedding_layer.embedding_size,
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
                    torch.nn.Linear(self.embed_output_dim+ self.embedding_layer.embedding_size, self.expert_num),
                    torch.nn.Softmax(dim=1),
                )
                for _ in range(self.task_num)
            ]
        )
        self.tower = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                )
                for _ in range(self.task_num)
            ]
        )
        
        self.ctr_expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.embed_output_dim,
                    self.expert_dims,
                    self.expert_dropout,
                    output_layer=False,
                )
                for _ in range(self.expert_num)
            ]
        )
        self.ctr_gate = torch.nn.Sequential(
            torch.nn.Linear(self.embed_output_dim, self.expert_num), torch.nn.Softmax(dim=1)
        )
        self.ctr_tower = MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                )

        self.ratio = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.confounder_dense = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense.weight.data)
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        ctr_fea = torch.cat([self.ctr_expert[i](feature_embedding).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        ctr_gate_value = self.ctr_gate(feature_embedding).unsqueeze(1) 
        ctr_task_fea = torch.bmm(ctr_gate_value, ctr_fea).squeeze(1)
        pctr = torch.sigmoid(self.ctr_tower(ctr_task_fea).squeeze(1))
        pctr = pctr.reshape(-1, 1)
        pctr_embedding = self.confounder_dense(pctr.detach())
        feature_embedding = torch.cat((feature_embedding, pctr_embedding), 1)
        
        fea = torch.cat([self.expert[i](feature_embedding).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        gate_value = [self.gate[i](feature_embedding).unsqueeze(1) for i in range(self.task_num)]
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        results[0] = pctr.squeeze(1)
        task_fea[0] = ctr_task_fea
        return results[0], results[1], torch.mul(results[0], results[1]), results[2], task_fea[0] 

class DESCM_Embedding_DA_1_1_Base(torch.nn.Module):
    '''
    具体细节见: deconfounder + DA 示意图
    '''
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

        self.expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.embed_output_dim+self.embedding_layer.embedding_size,
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
                    torch.nn.Linear(self.embed_output_dim+self.embedding_layer.embedding_size, self.expert_num),
                    torch.nn.Softmax(dim=1),
                )
                for _ in range(self.task_num)
            ]
        )
        self.tower = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                )
                for _ in range(self.task_num)
            ]
        )
        
        self.ctr_expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.embed_output_dim,
                    self.expert_dims,
                    self.expert_dropout,
                    output_layer=False,
                )
                for _ in range(self.expert_num)
            ]
        )
        self.ctr_gate = torch.nn.Sequential(
            torch.nn.Linear(self.embed_output_dim, self.expert_num), torch.nn.Softmax(dim=1)
        )
        self.ctr_tower = MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                    
                )

        self.ratio = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.confounder_dense = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense.weight.data)
        self.A_layer = MultiLayerPerceptron(
                    self.embed_output_dim,
                    [256, self.embed_output_dim],
                    [0, 0],
                    output_layer=False,
                )
        
        self.task_feature_dim = self.embed_output_dim
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        ctr_fea = torch.cat([self.ctr_expert[i](feature_embedding).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        ctr_gate_value = self.ctr_gate(feature_embedding).unsqueeze(1) 
        ctr_task_fea = torch.bmm(ctr_gate_value, ctr_fea).squeeze(1)
        pctr = torch.sigmoid(self.ctr_tower(ctr_task_fea).squeeze(1))
        pctr = pctr.reshape(-1, 1)
        pctr_embedding = self.confounder_dense(pctr.detach())
        
        A_embedding = self.A_layer(feature_embedding)
        new_embedding = torch.cat((A_embedding, pctr_embedding), 1)
        fea = torch.cat([self.expert[i](new_embedding).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        gate_value = [self.gate[i](new_embedding).unsqueeze(1) for i in range(self.task_num)]
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        results[0] = pctr.squeeze(1)
        task_fea[0] = ctr_task_fea
        return results[0], results[1], torch.mul(results[0], results[1]), results[2], A_embedding.detach()
    

class DESCM_Embedding_DA_1_1(torch.nn.Module):
    '''
    具体细节见: deconfounder + DA 示意图
    '''
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
        

        self.expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.embed_output_dim+self.embedding_layer.embedding_size,
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
                    torch.nn.Linear(self.embed_output_dim+self.embedding_layer.embedding_size, self.expert_num),
                    torch.nn.Softmax(dim=1),
                )
                for _ in range(self.task_num)
            ]
        )
        self.tower = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                )
                for _ in range(self.task_num)
            ]
        )
        
        self.ctr_expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.embed_output_dim,
                    self.expert_dims,
                    self.expert_dropout,
                    output_layer=False,
                )
                for _ in range(self.expert_num)
            ]
        )
        self.ctr_gate = torch.nn.Sequential(
            torch.nn.Linear(self.embed_output_dim, self.expert_num), torch.nn.Softmax(dim=1)
        )
        self.ctr_tower = MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                    
                )

        self.ratio = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.confounder_dense = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense.weight.data)
        self.A_layer = MultiLayerPerceptron(
                    self.embed_output_dim,
                    [256, self.embed_output_dim],
                    [0, 0],
                    output_layer=False,
                )
        self.task_feature_dim = self.embed_output_dim
        
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        ctr_fea = torch.cat([self.ctr_expert[i](feature_embedding).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        ctr_gate_value = self.ctr_gate(feature_embedding).unsqueeze(1) 
        ctr_task_fea = torch.bmm(ctr_gate_value, ctr_fea).squeeze(1)
        pctr = torch.sigmoid(self.ctr_tower(ctr_task_fea).squeeze(1))
        pctr = pctr.reshape(-1, 1)
        pctr_embedding = self.confounder_dense(pctr.detach())
        
        A_embedding = self.A_layer(feature_embedding)
        new_embedding = torch.cat((A_embedding, pctr_embedding), 1)
        fea = torch.cat([self.expert[i](new_embedding).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        gate_value = [self.gate[i](new_embedding).unsqueeze(1) for i in range(self.task_num)]
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        results[0] = pctr.squeeze(1)
        task_fea[0] = ctr_task_fea
        return results[0], results[1], torch.mul(results[0], results[1]), results[2], A_embedding

class DESCM_Embedding_DA_2_1(torch.nn.Module):
    '''
    具体细节见: deconfounder + DA 示意图, 这个版本的重点在于caoncate(原始embed, 消embed,pCTR)
    '''
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
        if A_embed_output_dim == 0:
            self.A_embed_output_dim  = self.embedding_layer.get_embed_output_dim()
        else:
            self.A_embed_output_dim = A_embed_output_dim
        

        self.expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.embed_output_dim + self.embedding_layer.embedding_size +self.A_embed_output_dim,
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
                    torch.nn.Linear(self.embed_output_dim + self.embedding_layer.embedding_size + self.A_embed_output_dim, self.expert_num),
                    torch.nn.Softmax(dim=1),
                )
                for _ in range(self.task_num)
            ]
        )
        self.tower = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                )
                for _ in range(self.task_num)
            ]
        )
        
        self.ctr_expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.embed_output_dim,
                    self.expert_dims,
                    self.expert_dropout,
                    output_layer=False,
                )
                for _ in range(self.expert_num)
            ]
        )
        self.ctr_gate = torch.nn.Sequential(
            torch.nn.Linear(self.embed_output_dim, self.expert_num), torch.nn.Softmax(dim=1)
        )
        self.ctr_tower = MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                    
                )

        self.ratio = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.confounder_dense = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense.weight.data)
        self.A_layer = MultiLayerPerceptron(
                    self.embed_output_dim,
                    [512, self.A_embed_output_dim],
                    [0.3, 0.3],
                    output_layer=False,
                )
        self.task_feature_dim = self.A_embed_output_dim
        
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        ctr_fea = torch.cat([self.ctr_expert[i](feature_embedding).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        ctr_gate_value = self.ctr_gate(feature_embedding).unsqueeze(1) 
        ctr_task_fea = torch.bmm(ctr_gate_value, ctr_fea).squeeze(1)
        pctr = torch.sigmoid(self.ctr_tower(ctr_task_fea).squeeze(1))
        pctr = pctr.reshape(-1, 1)
        pctr_embedding = self.confounder_dense(pctr.detach())
        
        A_embedding = self.A_layer(feature_embedding)
        new_embedding = torch.cat((feature_embedding, pctr_embedding, A_embedding), 1)
        fea = torch.cat([self.expert[i](new_embedding).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        gate_value = [self.gate[i](new_embedding).unsqueeze(1) for i in range(self.task_num)]
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        results[0] = pctr.squeeze(1)
        task_fea[0] = ctr_task_fea
        return results[0], results[1], torch.mul(results[0], results[1]), results[2], A_embedding

class ESCM(torch.nn.Module):
    def __init__(
        self,
        embedding_layer: AlldataEmbeddingLayer,
        task_num: int = 3,
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

        self.expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.embed_output_dim,
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
                    torch.nn.Linear(self.embed_output_dim, self.expert_num),
                    torch.nn.Softmax(dim=1),
                )
                for _ in range(self.task_num)
            ]
        )
        self.tower = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                )
                for _ in range(self.task_num)
            ]
        )

    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        fea = torch.cat([self.expert[i](feature_embedding).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        gate_value = [self.gate[i](feature_embedding).unsqueeze(1) for i in range(self.task_num)]
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results[0], results[1], torch.mul(results[0], results[1]), results[2], task_fea[0]

class MultiTaskLitModel_DA(pl.LightningModule):
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
        self.da_loss = DomainAdversarialLoss(DomainDiscriminator(in_feature=model.task_feature_dim, hidden_size=64))


    def training_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, representations = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        
        # oversampling and caculate da loss
        source_representations = representations[click==1]
        target_representations = representations[click==0]
        source_sampling_weight = torch.ones(len(source_representations))
        indices = torch.multinomial(source_sampling_weight, len(click[click==0]), replacement=True)
        oversampled_source_representations = source_representations[indices]
        da_loss = self.da_loss(oversampled_source_representations, target_representations)
        domain_acc = self.da_loss.domain_discriminator_accuracy
        
        classification_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion, p_imp=imp_pred)
        loss = classification_loss + da_loss
        self.log("train/da_loss", da_loss, on_epoch=True, on_step=True)
        self.log("train/classification_loss", classification_loss, on_epoch=True, on_step=True)
        self.log("train/domain_acc", domain_acc, on_epoch=True, on_step=True)
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
    
class MultiTaskLitModel_DA_click_to_impression(pl.LightningModule):
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
        self.da_loss = DomainAdversarialLoss(DomainDiscriminator(in_feature=model.task_feature_dim, hidden_size=64))


    def training_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, representations = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        
        # oversampling and caculate da loss
        source_representations = representations[click==1]
        # target_representations = representations[click==0]
        source_sampling_weight = torch.ones(len(source_representations))
        indices = torch.multinomial(source_sampling_weight, len(click), replacement=True)
        oversampled_source_representations = source_representations[indices]
        da_loss = self.da_loss(oversampled_source_representations, representations)
        domain_acc = self.da_loss.domain_discriminator_accuracy
        
        classification_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion, p_imp=imp_pred)
        loss = classification_loss + da_loss
        self.log("train/da_loss", da_loss, on_epoch=True, on_step=True)
        self.log("train/classification_loss", classification_loss, on_epoch=True, on_step=True)
        self.log("train/domain_acc", domain_acc, on_epoch=True, on_step=True)
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


class DESCM_Embedding(torch.nn.Module):
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
            input_dim=self.embed_output_dim + self.embedding_layer.embedding_size + self.A_embed_output_dim, 
            expert_num=self.expert_num, 
            task_num=self.task_num-1,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )
        self.ctr_mmoe = MMOE(
            input_dim=self.embed_output_dim, 
            expert_num=self.expert_num, 
            task_num=2,
            expert_dims=self.expert_dims,
            tower_dims=self.tower_dims, 
            expert_dropout=self.expert_dropout,
            tower_dropout=self.tower_dropout,
        )

        # self.ratio = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.confounder_dense = torch.nn.Linear(1, self.embedding_layer.embedding_size)
        torch.nn.init.xavier_uniform_(self.confounder_dense.weight.data)
        self.A_layer = MultiLayerPerceptron(
                    self.embed_output_dim,
                    [512, self.A_embed_output_dim],
                    [0.3, 0.3],
                    output_layer=False,
                )
        
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        pctr = self.ctr_mmoe(feature_embedding)[0]
        pctr = pctr.reshape(-1, 1)
        pctr_embedding = self.confounder_dense(pctr.detach())
        A_embedding = self.A_layer(feature_embedding)
        new_embedding = torch.cat((feature_embedding, pctr_embedding, A_embedding), 1)
        results = self.mmoe(new_embedding)
        return pctr.squeeze(1), results[0], torch.mul(pctr.squeeze(1), results[0]), results[1], feature_embedding