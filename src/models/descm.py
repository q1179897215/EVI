from typing import Any, Dict, Tuple, List
from abc import abstractmethod, ABC
import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig
from torch import nn
from torchmetrics.classification import AUROC, Accuracy, BinaryAUROC

from src.models.common import MultiLayerPerceptron, AlldataEmbeddingLayer

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
    def forward(self, x):
        feature_embedding = self.embedding_layer(x)
        ctr_fea = torch.cat([self.ctr_expert[i](feature_embedding).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        ctr_gate_value = self.ctr_gate(feature_embedding).unsqueeze(1) 
        ctr_task_fea = torch.bmm(ctr_gate_value, ctr_fea).squeeze(1)
        pctr = torch.sigmoid(self.ctr_tower(ctr_task_fea).squeeze(1))
        pctr = pctr.reshape(-1, 1)
        pctr_embedding = self.confounder_dense(pctr.detach())
        
        new_embedding = self.A_layer(feature_embedding)
        new_embedding = torch.cat((new_embedding, pctr_embedding), 1)
        fea = torch.cat([self.expert[i](new_embedding).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        gate_value = [self.gate[i](new_embedding).unsqueeze(1) for i in range(self.task_num)]
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        results[0] = pctr.squeeze(1)
        task_fea[0] = ctr_task_fea
        return results[0], results[1], torch.mul(results[0], results[1]), results[2], task_fea[0]
    
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
    


