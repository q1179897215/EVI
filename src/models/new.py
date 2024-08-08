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
    
class EsmmOuterProductEmbFea(torch.nn.Module):
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
        self.tower_pred[1] = nn.Linear(self.input_dim*5, 1)

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
        ait = self.outer_product(x, pctr_embedding)
        
        pcvr = torch.sigmoid(self.tower_pred[1](ait).squeeze(1))
        results = [pctr, pcvr]
        return results, tower_fea, tower_fea

class NewOuterProductFea(torch.nn.Module):
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
        self.tower_pred[2] = nn.Linear(self.tower_dims[-1]*5, 1)

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
        ait = self.outer_product(tower_fea[2], pctr_embedding)
        pcvr = torch.sigmoid(self.tower_pred[1](tower_fea[1]).squeeze(1))
        pconversion = torch.sigmoid(self.tower_pred[2](ait).squeeze(1))
        results = [pctr, pcvr, pconversion]
        return results, tower_fea, tower_fea
    
class NewMmoeOuterProductFea(torch.nn.Module):
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
                for _ in range(self.task_num-1)
            ]
        )

        

        self.tower_fea = torch.nn.ModuleList([
            MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                    output_layer=False,
                )  # Add .cuda() to move the model to GPU
                for _ in range(self.task_num-1)
        ])
        self.tower_pred = torch.nn.ModuleList([
                nn.Linear(self.tower_dims[-1], 1) # Add .cuda() to move the model to GPU
                for _ in range(self.task_num-1)
        ])
        
        self.conversion_expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.input_dim*5,
                    self.expert_dims,
                    self.expert_dropout,
                    output_layer=False,
                ).cuda()  # Add .cuda() to move the model to GPU
                for _ in range(self.expert_num)
            ]
        )

        self.conversion_gate = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.input_dim*5, self.expert_num),
                    torch.nn.Softmax(dim=1),
                )  # Add .cuda() to move the model to GPU
                for _ in range(1)
            ]
        )
        self.conversion_tower_fea = torch.nn.ModuleList([
            MultiLayerPerceptron(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                    output_layer=False,
                )  
                for _ in range(1)
        ])
        self.conversion_tower_pred = torch.nn.ModuleList([
                nn.Linear(self.tower_dims[-1], 1) # Add .cuda() to move the model to GPU
                for _ in range(1)
        ])
        
        

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
        fea = torch.cat([self.expert[i](x).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        gate_value = [self.gate[i](x).unsqueeze(1) for i in range(self.task_num-1)]
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num-1)]
        tower_fea = [self.tower_fea[i](task_fea[i]) for i in range(self.task_num-1)]
        pctr = torch.sigmoid(self.tower_pred[0](tower_fea[0]).squeeze(1))
        pcvr = torch.sigmoid(self.tower_pred[1](tower_fea[1]).squeeze(1))
        
        pctr_embedding = self.pctr_embedding(pctr.detach().unsqueeze(1))
        ait = self.outer_product(x, pctr_embedding)
        
        conversion_fea = torch.cat([self.conversion_expert[i](ait).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        conversion_gate = [self.conversion_gate[i](ait).unsqueeze(1) for i in range(1)]
        conversion_task_fea = [torch.bmm(conversion_gate[i], conversion_fea).squeeze(1) for i in range(1)]
        conversion_tower_fea = [self.conversion_tower_fea[0](conversion_task_fea[0])]
        pconversion = torch.sigmoid(self.conversion_tower_pred[0](conversion_tower_fea[0]).squeeze(1))
        
        results = [pctr, pcvr, pconversion]
        return results, conversion_task_fea, conversion_tower_fea
    
class NewMmoeOuterProductLayerFea(torch.nn.Module):
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
        self.expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.input_dim,
                    self.expert_dims,
                    self.expert_dropout,
                    output_layer=False,
                )  # Add .cuda() to move the model to GPU
                for _ in range(self.expert_num)
            ]
        )

        self.gate = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.input_dim, self.expert_num),
                    torch.nn.Softmax(dim=1),
                )  # Add .cuda() to move the model to GPU
                for _ in range(self.task_num-1)
            ]
        )

        self.tower_fea = torch.nn.ModuleList([
            MlpLayerFea(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                    output_layer=False,
                )  # Add .cuda() to move the model to GPU
                for _ in range(self.task_num-1)
        ])
        self.tower_pred = torch.nn.ModuleList([
                nn.Linear(self.tower_dims[-1], 1) # Add .cuda() to move the model to GPU
                for _ in range(self.task_num-1)
        ])
        
        self.conversion_expert = torch.nn.ModuleList(
            [
                MultiLayerPerceptron(
                    self.input_dim*5,
                    self.expert_dims,
                    self.expert_dropout,
                    output_layer=False,
                ).cuda()  # Add .cuda() to move the model to GPU
                for _ in range(self.expert_num)
            ]
        )

        self.conversion_gate = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.input_dim*5, self.expert_num),
                    torch.nn.Softmax(dim=1),
                )  # Add .cuda() to move the model to GPU
                for _ in range(1)
            ]
        )
        self.conversion_tower_fea = torch.nn.ModuleList([
            MlpLayerFea(
                    self.expert_dims[-1],
                    self.tower_dims,
                    self.tower_dropout,
                    output_layer=False,
                )  
                for _ in range(1)
        ])
        self.conversion_tower_pred = torch.nn.ModuleList([
                nn.Linear(self.tower_dims[-1], 1) # Add .cuda() to move the model to GPU
                for _ in range(1)
        ])
        
        

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
        fea = torch.cat([self.expert[i](x).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        gate_value = [self.gate[i](x).unsqueeze(1) for i in range(self.task_num-1)]
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num-1)]
        tower_fea = [self.tower_fea[i](task_fea[i]) for i in range(self.task_num-1)]
        pctr = torch.sigmoid(self.tower_pred[0](tower_fea[0][-1]).squeeze(1))
        pcvr = torch.sigmoid(self.tower_pred[1](tower_fea[1][-1]).squeeze(1))
        
        pctr_embedding = self.pctr_embedding(pctr.detach().unsqueeze(1))
        ait = self.outer_product(x, pctr_embedding)
        
        conversion_fea = torch.cat([self.conversion_expert[i](ait).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        conversion_gate = [self.conversion_gate[i](ait).unsqueeze(1) for i in range(1)]
        conversion_task_fea = [torch.bmm(conversion_gate[i], conversion_fea).squeeze(1) for i in range(1)]
        conversion_tower_fea = [self.conversion_tower_fea[0](conversion_task_fea[0])]
        pconversion = torch.sigmoid(self.conversion_tower_pred[0](conversion_tower_fea[0][-1]).squeeze(1))
        
        tower_fea.append(conversion_tower_fea[0])
        task_fea.append(conversion_task_fea[0])
        
        
        results = [pctr, pcvr, pconversion]
        return results, task_fea, tower_fea
    
class DdpoFea(torch.nn.Module):
    def __init__(self, input_dim, expert_num, task_num, expert_dims, tower_dims, expert_dropout, tower_dropout):
        super().__init__()
        self.input_dim = input_dim    
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_dims = expert_dims
        self.expert_dropout = expert_dropout
        self.tower_dims = tower_dims
        self.tower_dropout = tower_dropout
        self.attention_layer_ctr = Attention(self.tower_dims[-1])
        self.attention_layer_ctcvr = Attention(self.tower_dims[-1])
        self.cvr_info = nn.Linear(64, 32)
        

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
        pctr = torch.sigmoid(self.tower_pred[0](tower_fea[0]).squeeze(1))
        pconversion = torch.sigmoid(self.tower_pred[2](tower_fea[2]).squeeze(1))
        ait_ctr = self.attention_layer_ctr(torch.cat([torch.unsqueeze(tower_fea[1], 1), torch.unsqueeze(tower_fea[0], 1)], 1))
        ait_ctcvr = self.attention_layer_ctcvr(torch.cat([torch.unsqueeze(tower_fea[1], 1), torch.unsqueeze(tower_fea[2], 1)], 1))
        ait = torch.cat([ait_ctr, ait_ctcvr], 1)
        cvr_tower_fea = self.cvr_info(ait)
        pcvr = torch.sigmoid(self.tower_pred[1](cvr_tower_fea).squeeze(1))
        
        results = [pctr, pcvr, pconversion]
        return results, tower_fea, tower_fea
    
class DdpoConFea(torch.nn.Module):
    def __init__(self, input_dim, expert_num, task_num, expert_dims, tower_dims, expert_dropout, tower_dropout):
        super().__init__()
        self.input_dim = input_dim    
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_dims = expert_dims
        self.expert_dropout = expert_dropout
        self.tower_dims = tower_dims
        self.tower_dropout = tower_dropout
        self.attention_layer_ctr = Attention(self.tower_dims[-1])
        self.attention_layer_ctcvr = Attention(self.tower_dims[-1])
        self.cvr_info = nn.Linear(64, 32)
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
        self.tower_pred[2] = nn.Linear(self.tower_dims[-1]*5, 1)
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
        conversion_fea = self.outer_product(tower_fea[2], pctr_embedding)
        pconversion = torch.sigmoid(self.tower_pred[2](conversion_fea).squeeze(1))
        ait_ctr = self.attention_layer_ctr(torch.cat([torch.unsqueeze(tower_fea[1], 1), torch.unsqueeze(tower_fea[0], 1)], 1))
        ait_ctcvr = self.attention_layer_ctcvr(torch.cat([torch.unsqueeze(tower_fea[1], 1), torch.unsqueeze(tower_fea[2], 1)], 1))
        ait = torch.cat([ait_ctr, ait_ctcvr], 1)
        cvr_tower_fea = self.cvr_info(ait)
        pcvr = torch.sigmoid(self.tower_pred[1](cvr_tower_fea).squeeze(1))
        
        results = [pctr, pcvr, pconversion]
        return results, tower_fea, tower_fea


class New(torch.nn.Module):
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

        self.extractor_and_predictor = MmoeFea(
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

class NewDaLitModel(pl.LightningModule):
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
        
        self.da_loss = DomainAdversarialLoss(DomainDiscriminator(in_feature=self.model.embed_output_dim, hidden_size=64))
    
    def training_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        results, feature_embedding, task_fea, tower_fea = self.model(features)
        click_pred, conversion_pred, click_conversion_pred = results[0], results[1], results[2]
        
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
        classification_loss = self.loss.caculate_loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)
        loss = classification_loss #+ 0.5*da_loss
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        self.log("train/da_loss", da_loss, on_epoch=True, on_step=True)
        self.log("train/classification_loss", classification_loss, on_epoch=True, on_step=True)
        self.log("train/da_acc", da_acc, on_epoch=True, on_step=True)
        
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
        optimizer = torch.optim.Adam(itertools.chain(self.model.parameters(), self.da_loss.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    

    
class NewMiLitModel(pl.LightningModule):
    def __init__(self, 
                 model:torch.nn.Module,
                 loss:torch.nn.Module, 
                 lr:float, 
                 weight_decay:float=1, 
                 batch_type:str='fr',
                 mi_ratio:float=0.8,
                 var_ratio:float=5.0,
                 info_layer_num:float=3.0):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.automatic_optimization = False
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.mi_ratio = mi_ratio
        self.info_layer_num = info_layer_num
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
        
        self.d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha_0 = torch.nn.Parameter(torch.tensor([var_ratio] * 128)).to(self.d)
        self.alpha_1 = torch.nn.Parameter(torch.tensor([var_ratio] * 64)).to(self.d)
        self.alpha_2 = torch.nn.Parameter(torch.tensor([var_ratio] * 32)).to(self.d)
        self.var_0 = torch.nn.functional.softplus(self.alpha_0) + 1e-3
        self.var_1 = torch.nn.functional.softplus(self.alpha_1) + 1e-3
        self.var_2 = torch.nn.functional.softplus(self.alpha_2) + 1e-3
        
        self.mean_layer_0 = nn.Linear(128, 128)
        self.mean_layer_1 = nn.Linear(64, 64)
        self.mean_layer_2 = nn.Linear(32, 32)
        # self.var_layer_0 = nn.Linear(128, 128)
        # self.var_layer_1 = nn.Linear(64, 64)
        # self.var_layer_2 = nn.Linear(32, 32)
        
        # self.ctr_alpha_0 = torch.nn.Parameter(torch.tensor([5.0] * 128)).to(self.d)
        # self.ctr_alpha_1 = torch.nn.Parameter(torch.tensor([5.0] * 64)).to(self.d)
        # self.ctr_alpha_2 = torch.nn.Parameter(torch.tensor([5.0] * 32)).to(self.d)
        # self.ctr_var_0 = torch.nn.functional.softplus(self.alpha_0) + 1e-3
        # self.ctr_var_1 = torch.nn.functional.softplus(self.alpha_1) + 1e-3
        # self.ctr_var_2 = torch.nn.functional.softplus(self.alpha_2) + 1e-3
        # self.ctr_mean_layer_0 = nn.Linear(128, 128)
        # self.ctr_mean_layer_1 = nn.Linear(64, 64)
        # self.ctr_mean_layer_2 = nn.Linear(32, 32)
        
    
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        self.toggle_optimizer(optimizer)
        optimizer.zero_grad()
        
        click, conversion, features = self.batch_transform(batch)
        results, feature_embedding, task_fea, tower_fea = self.model(features)
        click_pred, conversion_pred, click_conversion_pred = results[0], results[1], results[2]
        
        if self.info_layer_num != 0.0:
            teacher_layers = tower_fea[2]
            student_layers = tower_fea[1]
            
            student_layer_mean_0 = self.mean_layer_0(student_layers[0])
            student_layer_mean_1 = self.mean_layer_1(student_layers[1])
            student_layer_mean_2 = self.mean_layer_2(student_layers[2])
            
            # student_layer_var_0 = self.var_layer_0(student_layers[0])
            # student_layer_var_1 = self.var_layer_1(student_layers[1])
            # student_layer_var_2 = self.var_layer_2(student_layers[2])
            
            # student_layer_var_0 = torch.nn.functional.softplus(student_layer_var_0) + 1e-3
            # student_layer_var_1 = torch.nn.functional.softplus(student_layer_var_1) + 1e-3
            # student_layer_var_2 = torch.nn.functional.softplus(student_layer_var_2) + 1e-3
            
            # vids_loss_0 = torch.mean(torch.log(student_layer_var_0) + torch.square(teacher_layers[0] - student_layer_mean_0) / student_layer_var_0) / 2.0
            # vids_loss_1 = torch.mean(torch.log(student_layer_var_1) + torch.square(teacher_layers[1] - student_layer_mean_1) / student_layer_var_1) / 2.0
            # vids_loss_2 = torch.mean(torch.log(student_layer_var_2) + torch.square(teacher_layers[2] - student_layer_mean_2) / student_layer_var_2) / 2.0
            
            vids_loss_0 = torch.mean(torch.log(self.var_0) + torch.square(teacher_layers[0] - student_layer_mean_0) / self.var_0) / 2.0
            vids_loss_1 = torch.mean(torch.log(self.var_1) + torch.square(teacher_layers[1] - student_layer_mean_1) / self.var_1) / 2.0
            vids_loss_2 = torch.mean(torch.log(self.var_2) + torch.square(teacher_layers[2] - student_layer_mean_2) / self.var_2) / 2.0
        
        if self.info_layer_num == 0.0:
            vids_loss = 0.0
        elif self.info_layer_num == 1.0:
            vids_loss = vids_loss_2
        elif self.info_layer_num == 2.0:
            vids_loss = vids_loss_1 + vids_loss_2
        elif self.info_layer_num == 3.0:
            vids_loss = vids_loss_0 + vids_loss_1 + vids_loss_2

        
        
        # caculate normal loss
        classification_loss = self.loss.caculate_loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)
        loss = classification_loss + vids_loss * self.mi_ratio
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        self.log("train/vids_loss", vids_loss, on_epoch=True, on_step=True)
        self.log("train/classification_loss", classification_loss, on_epoch=True, on_step=True)
        
        self.manual_backward(loss, retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        self.untoggle_optimizer(optimizer)
        
        
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


class NewMultiTaskLoss(nn.Module):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 ):
        super().__init__()
    
    
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr):
        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        loss_cvr_click = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr_unclick = torch.nn.functional.binary_cross_entropy(p_cvr, p_ctcvr.detach(), reduction='none')
        loss_cvr = torch.mean(y_ctr*loss_cvr_click + 0.5*(1-y_ctr)*loss_cvr_unclick)
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_cvr, reduction='mean')
        loss = loss_ctr + loss_cvr + loss_ctcvr
        return loss

class CvrAllSpaceMultiTaskLoss(nn.Module):
    def __init__(self, 
                 ctr_loss_proportion: float = 1, 
                 cvr_loss_proportion: float = 1, 
                 ctcvr_loss_proportion: float = 0.1,
                 unclick_space_loss_proportion: float = 0.2,
                 ):
        super().__init__()
        self.unclick_space_loss_proportion = unclick_space_loss_proportion
        self.ctr_loss_proportion =  ctr_loss_proportion
        self.ctcvr_loss_proportion = ctcvr_loss_proportion
    
    def caculate_loss(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr):
        pctr_clamp = torch.clamp(p_ctr.detach(), 0.001, 1-0.001)
        ips = 1 / pctr_clamp
        non_ips = 1 / (1 - pctr_clamp)
        loss_ctr = torch.nn.functional.binary_cross_entropy(p_ctr, y_ctr, reduction='mean')
        loss_cvr_click = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr_unclick = torch.nn.functional.binary_cross_entropy(p_cvr, p_ctcvr.detach(), reduction='none')
        loss_cvr = torch.mean(ips*y_ctr*loss_cvr_click + non_ips*self.unclick_space_loss_proportion*(1-y_ctr)*loss_cvr_unclick)
        loss_ctcvr = torch.nn.functional.binary_cross_entropy(p_ctcvr, y_cvr, reduction='none')
        loss_ctcvr = torch.mean(loss_ctcvr)
        loss_ctcvr2 = torch.nn.functional.binary_cross_entropy(p_cvr*p_ctr, y_cvr, reduction='mean')
        loss = self.ctr_loss_proportion*loss_ctr + loss_cvr + self.ctcvr_loss_proportion*loss_ctcvr + 0.1 * loss_ctcvr2
        return loss