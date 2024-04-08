import torch
from type import int, List, Tuple
from src.models.common import AlldataEmbeddingLayer

class CGCM(torch.nn.Module):

    def __init__(self, 
                 embedding_layer:AlldataEmbeddingLayer, 
                 task_num:int=3,
                 expert_num:int=8,):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.embed_output_dim = self.embedding_layer.get_embed_output_dim()
        # expert, gates and towers
        self.task_num = config['task_num']
        self.expert_num = config['expert_num']
        
        self.expert = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, config['bottom_mlp_dims'], config['dropout_expert'], output_layer=False) for i in range(self.expert_num)])
        
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(config['bottom_mlp_dims'][-1], config['tower_mlp_dims'], config['dropout_tower']) for i in range(self.task_num)])

        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.embed_output_dim, self.expert_num), torch.nn.Softmax(dim=1)) for i in range(self.task_num)])

        if self.config['deconfounder'] == True:

            self.ctr_expert = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, config['bottom_mlp_dims'], config['dropout_expert'], output_layer=False) for i in range(self.expert_num)])
            self.ctr_gate = torch.nn.Sequential(torch.nn.Linear(self.embed_output_dim, self.expert_num), torch.nn.Softmax(dim=1))
            self.ctr_tower = MultiLayerPerceptron(config['bottom_mlp_dims'][-1], config['tower_mlp_dims'], config['dropout_tower'])

            if self.config['deconfounder_type'] == 'add_fea':
                self.tower[1] = MultiLayerPerceptron(config['bottom_mlp_dims'][-1]+1, config['tower_mlp_dims'], config['dropout_tower'])
            elif self.config['deconfounder_type'] == 'add_embedding':
                self.expert = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim+config['embedding_size'], config['bottom_mlp_dims'], config['dropout_expert'], output_layer=False) for i in range(self.expert_num)])
                self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.embed_output_dim+config['embedding_size'], self.expert_num), torch.nn.Softmax(dim=1)) for i in range(self.task_num)])


        self.ratio = torch.nn.Parameter(torch.FloatTensor([0.1]))
        self.embedding_layer_pcvr =  torch.nn.Linear(1, config['embedding_size'])
        torch.nn.init.xavier_uniform_(self.embedding_layer_pcvr.weight.data)