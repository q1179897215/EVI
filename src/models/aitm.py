import torch
import torch.nn as nn

class Tower(nn.Module):
  def __init__(self,
               input_dim,
               dims=[128, 64, 32],
               drop_prob=[0.1, 0.3, 0.3]):
    super(Tower, self).__init__()
    self.dims = dims
    self.drop_prob = drop_prob
    self.layer = nn.Sequential(nn.Linear(input_dim, dims[0]), 
                               nn.ReLU(),
                               nn.Dropout(drop_prob[0]),
                               nn.Linear(dims[0], dims[1]), 
                               nn.ReLU(),
                               nn.Dropout(drop_prob[1]),
                               nn.Linear(dims[1], dims[2]), 
                               nn.ReLU(),
                               nn.Dropout(drop_prob[2]))

  def forward(self, x):
    x = torch.flatten(x, start_dim=1)
    x = self.layer(x)
    return x


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


class AITM(nn.Module):
    def __init__(
        self,
        embedding_layer,
        tower_mlp_dims,
        dropout_tower,  
    ):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.embed_output_dim = self.embedding_layer.get_embed_output_dim()
        self.tower_mlp_dims = tower_mlp_dims
        self.dropout_tower = dropout_tower    
        
        self.click_tower = Tower(self.embed_output_dim, self.tower_mlp_dims, self.dropout_tower)
        self.conversion_tower = Tower(self.embed_output_dim, self.tower_mlp_dims, self.dropout_tower)
        self.attention_layer = Attention(self.tower_mlp_dims[-1])

        self.info_layer = nn.Sequential(nn.Linear(self.tower_mlp_dims[-1], 32), 
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout_tower[-1]))

        self.click_layer = nn.Sequential(nn.Linear(self.tower_mlp_dims[-1], 1), nn.Sigmoid())
        self.conversion_layer = nn.Sequential(nn.Linear(self.tower_mlp_dims[-1], 1), nn.Sigmoid())


    def forward(self, x):
        
        feature_embedding = self.embedding_layer(x)
        
        tower_click = self.click_tower(feature_embedding)

        tower_conversion = torch.unsqueeze(
            self.conversion_tower(feature_embedding), 1)

        info = torch.unsqueeze(self.info_layer(tower_click), 1)

        ait = self.attention_layer(torch.cat([tower_conversion, info], 1))

        click = torch.squeeze(self.click_layer(tower_click), dim=1)
        conversion = torch.squeeze(self.conversion_layer(ait), dim=1)

        return click, conversion, torch.mul(click, conversion), click, feature_embedding
  
class Loss_AITM(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()
        
    def forward(self, p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, **kwargs):
        constraint_weight=0.6
        loss_ctr = self.loss(p_ctr, y_ctr)
        loss_cvr = torch.nn.functional.binary_cross_entropy(p_cvr, y_cvr, reduction='none')
        loss_cvr = torch.mean(loss_cvr * y_ctr)

        label_constraint = torch.maximum(y_ctr*(p_cvr - p_ctr), torch.zeros_like(y_ctr))
        constraint_loss = torch.mean(label_constraint)
        loss = loss_ctr + loss_cvr + constraint_weight * constraint_loss
        
        return loss
