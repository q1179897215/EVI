from typing import Any, Dict, Tuple, List
from torchmetrics.classification import BinaryAUROC # type: ignore
import lightning.pytorch as pl
import torch
import torch.nn as nn
from src.models.common import BatchTransform, MultiLayerPerceptron, AlldataEmbeddingLayer

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

class MMOE(torch.nn.Module):
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
        fea = torch.cat([self.expert[i](x).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        gate_value = [self.gate[i](x).unsqueeze(1) for i in range(self.task_num)]
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results
    

class DESCM_Embedding_DA_cycada(torch.nn.Module):
    '''
    具体细节见: deconfounder + DA 示意图 cycada
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
        return pctr.squeeze(1), results[0], torch.mul(pctr.squeeze(1), results[0]), results[1], feature_embedding.detach()
    

class MultiTaskLitModel_cycada(pl.LightningModule):
    def __init__(self, 
                 model:torch.nn.Module,
                 loss:torch.nn.Module, 
                 lr:float, 
                 weight_decay:float=1, 
                 batch_type:str='ccp', 
                 trade_off_cycle:int=1,
                 trade_off_identity:int=1,
                 trade_off_semantic:int=1):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.automatic_optimization = False
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.bath_type = batch_type
        self.trade_off_cycle = trade_off_cycle
        self.trade_off_identity = trade_off_identity
        self.trade_off_semantic = trade_off_semantic
        self.ctr_auc = BinaryAUROC()
        self.cvr_auc = BinaryAUROC()
        self.ctcvr_auc = BinaryAUROC()
        self.batch_transform = BatchTransform(batch_type)
        # define new loss
        self.criterion_gan = LeastSquaresGenerativeAdversarialLoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_semantic = torch.nn.BCELoss()
        
        # define generators and discriminators
        self.A_embed_output_dim = model.A_embed_output_dim
        self.netG_T2S = MultiLayerPerceptron(
                    self.A_embed_output_dim,
                    [512, self.A_embed_output_dim],
                    [0.3, 0.3],
                    output_layer=False,
                )
        self.netD_S = DomainDiscriminator(in_feature=self.A_embed_output_dim, hidden_size=64)
        self.netD_T = DomainDiscriminator(in_feature=self.A_embed_output_dim, hidden_size=64)
        
        self.predict = MMOE(
            input_dim=self.A_embed_output_dim, 
            expert_num=self.model.expert_num, 
            task_num=1,
            expert_dims=self.model.expert_dims,
            tower_dims=self.model.tower_dims, 
            expert_dropout=self.model.expert_dropout,
            tower_dropout=self.model.tower_dropout,
        )
        

    
    def training_step(self, batch, batch_idx):
        
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, representations = self.model(features)
        conversion_pred_filter = conversion_pred[click==1]
        conversion_filter = conversion[click==1]
        # oversampling and caculate da loss
        source_representations = representations[click==1]
        target_representations = representations[click==0]
        source_sampling_weight = torch.ones(len(source_representations))
        indices = torch.multinomial(source_sampling_weight, len(click[click==0]), replacement=True)
        real_S = source_representations[indices]
        label_s = conversion[indices]
        real_T = target_representations
        
        
        # Compute fake images and reconstruction images.
        fake_T = self.model.A_layer(real_S)
        rec_S = self.netG_T2S(fake_T)
        fake_S = self.netG_T2S(real_T)
        rec_T = self.model.A_layer(fake_S)
        
        optimizer_D, optimizer_G = self.optimizers()
        
        self.toggle_optimizer(optimizer_G)
        set_requires_grad(self.netD_S, False)
        set_requires_grad(self.netD_T, False)
        optimizer_G.zero_grad()

        # GAN loss D_T(G_S2T(S))
        loss_G_S2T = self.criterion_gan(self.netD_T(fake_T), real=True)
        # GAN loss D_S(G_T2S(B))
        loss_G_T2S = self.criterion_gan(self.netD_S(fake_S), real=True)
        # Cycle loss || G_T2S(G_S2T(S)) - S||
        loss_cycle_S = self.criterion_cycle(rec_S, real_S) * self.trade_off_cycle
        # Cycle loss || G_S2T(G_T2S(T)) - T||
        loss_cycle_T = self.criterion_cycle(rec_T, real_T) * self.trade_off_cycle
        # Identity loss
        # G_S2T should be identity if real_T is fed: ||G_S2T(real_T) - real_T||
        identity_T = self.model.A_layer(real_T)
        loss_identity_T = self.criterion_identity(identity_T, real_T) * self.trade_off_identity
        # G_T2S should be identity if real_S is fed: ||G_T2S(real_S) - real_S||
        identity_S = self.netG_T2S(real_S)
        loss_identity_S = self.criterion_identity(identity_S, real_S) * self.trade_off_identity
        # Semantic loss
        pred_fake_T = self.predict(fake_T)[0]
        pred_real_S = self.predict(real_S)[0]
        loss_semantic_S2T = self.criterion_semantic(pred_fake_T, label_s) * self.trade_off_semantic
        
        pred_fake_S = self.predict(fake_S)[0]
        pred_real_T = self.predict(real_T)[0]
        loss_semantic_T2S = self.criterion_semantic(pred_fake_S, pred_real_T) * self.trade_off_semantic
        
        task_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion, p_imp=imp_pred)
        # combined loss and calculate gradients
        loss_G = loss_G_S2T + loss_G_T2S + loss_cycle_S + loss_cycle_T + \
                 loss_identity_S + loss_identity_T + loss_semantic_S2T + loss_semantic_T2S + task_loss
        
        self.manual_backward(loss_G, retain_graph=True)
        optimizer_G.step()
        optimizer_G.zero_grad()
        self.untoggle_optimizer(optimizer_G)


        self.toggle_optimizer(optimizer_D)
        set_requires_grad(self.netD_S, True)
        set_requires_grad(self.netD_T, True)
        optimizer_D.zero_grad()
        # Calculate GAN loss for discriminator D_S
        fake_S_ = fake_S.detach()
        fake_T_ = fake_T.detach()
        loss_D_S = 0.5 * (self.criterion_gan(self.netD_S(real_S), True) + self.criterion_gan(self.netD_S(fake_S_), False))
        # Calculate GAN loss for discriminator D_T
        
        loss_D_T = 0.5 * (self.criterion_gan(self.netD_T(real_T), True) + self.criterion_gan(self.netD_T(fake_T_), False))
        loss_D = loss_D_S + loss_D_T
        self.manual_backward(loss_D)
        optimizer_D.step()
        optimizer_D.zero_grad()
        self.untoggle_optimizer(optimizer_D)
        self.log('train/loss_G_S2T', loss_G_S2T.item(), on_epoch=True, on_step=True)
        self.log('train/loss_G_T2S', loss_G_T2S.item(), on_epoch=True, on_step=True)
        self.log('train/loss_D_S', loss_D_S.item(), on_epoch=True, on_step=True)
        self.log('train/loss_D_T', loss_D_T.item(), on_epoch=True, on_step=True)
        self.log('train/loss_cycle_S', loss_cycle_S.item(), on_epoch=True, on_step=True)
        self.log('train/loss_cycle_T', loss_cycle_T.item(), on_epoch=True, on_step=True)
        self.log('train/loss_identity_S', loss_identity_S.item(), on_epoch=True, on_step=True)
        self.log('train/loss_identity_T', loss_identity_T.item(), on_epoch=True, on_step=True)
        self.log('train/loss_semantic_S2T', loss_semantic_S2T.item(), on_epoch=True, on_step=True)
        self.log('train/loss_semantic_T2S', loss_semantic_T2S.item(), on_epoch=True, on_step=True)
        self.log('train/task_loss', task_loss.item(), on_epoch=True, on_step=True)
        
        
        
    def validation_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, representations = self.model(features)
        # filter the conversion_pred where click is 0
        val_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion, p_imp=imp_pred)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, representations = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.ctr_auc.update(click_pred, click)
        self.cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.ctcvr_auc.update(click_conversion_pred, click * conversion)

    def configure_optimizers(self):    
        # define optimizer and lr scheduler
        optimizer_G = torch.optim.Adam(itertools.chain(self.netG_T2S.parameters(), self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_D = torch.optim.Adam(itertools.chain(self.netD_S.parameters(), self.netD_T.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer_D, optimizer_G
    
    
class DESCM_Embedding_DA_cycada_tts(torch.nn.Module):
    '''
    具体细节见: deconfounder + DA 示意图 cycada
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
        self.netG_S2T = MultiLayerPerceptron(
                    self.embed_output_dim,
                    [512, self.A_embed_output_dim],
                    [0.3, 0.3],
                    output_layer=False,
                )
        self.netG_T2S = MultiLayerPerceptron(
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
        A_embedding = self.netG_T2S(feature_embedding)
        new_embedding = torch.cat((feature_embedding, pctr_embedding, A_embedding), 1)
        results = self.mmoe(new_embedding)
        return pctr.squeeze(1), results[0], torch.mul(pctr.squeeze(1), results[0]), results[1], feature_embedding.detach()
    
class MultiTaskLitModel_cycada_tts(pl.LightningModule):
    '''
    tts中， pCTR和原始embedding concatene netG_T2S产生的embedding进行pCVR预测
    '''
    
    def __init__(self, 
                 model:torch.nn.Module,
                 loss:torch.nn.Module, 
                 lr:float, 
                 weight_decay:float=1, 
                 batch_type:str='ccp', 
                 trade_off_cycle:int=1,
                 trade_off_identity:int=1,
                 trade_off_semantic:int=1):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.automatic_optimization = False
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.bath_type = batch_type
        self.trade_off_cycle = trade_off_cycle
        self.trade_off_identity = trade_off_identity
        self.trade_off_semantic = trade_off_semantic
        self.ctr_auc = BinaryAUROC()
        self.cvr_auc = BinaryAUROC()
        self.ctcvr_auc = BinaryAUROC()
        self.batch_transform = BatchTransform(batch_type)
        # define new loss
        self.criterion_gan = LeastSquaresGenerativeAdversarialLoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_semantic = torch.nn.BCELoss()
        
        # define discriminators
        self.A_embed_output_dim = model.A_embed_output_dim

        self.netD_S = DomainDiscriminator(in_feature=self.A_embed_output_dim, hidden_size=64)
        self.netD_T = DomainDiscriminator(in_feature=self.A_embed_output_dim, hidden_size=64)
        
        self.predict = MMOE(
            input_dim=self.A_embed_output_dim, 
            expert_num=self.model.expert_num, 
            task_num=1,
            expert_dims=self.model.expert_dims,
            tower_dims=self.model.tower_dims, 
            expert_dropout=self.model.expert_dropout,
            tower_dropout=self.model.tower_dropout,
        )
        

    
    def training_step(self, batch, batch_idx):
        
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, representations = self.model(features)
        conversion_pred_filter = conversion_pred[click==1]
        conversion_filter = conversion[click==1]
        # oversampling and caculate da loss
        source_representations = representations[click==1]
        target_representations = representations[click==0]
        source_sampling_weight = torch.ones(len(source_representations))
        indices = torch.multinomial(source_sampling_weight, len(click[click==0]), replacement=True)
        real_S = source_representations[indices]
        label_s = conversion[indices]
        real_T = target_representations
        
        
        # Compute fake images and reconstruction images.
        fake_T = self.model.netG_S2T(real_S)
        rec_S = self.model.netG_T2S(fake_T)
        fake_S = self.model.netG_T2S(real_T)
        rec_T = self.model.netG_S2T(fake_S)
        
        optimizer_D, optimizer_G = self.optimizers()
        
        self.toggle_optimizer(optimizer_G)
        set_requires_grad(self.netD_S, False)
        set_requires_grad(self.netD_T, False)
        optimizer_G.zero_grad()

        # GAN loss D_T(G_S2T(S))
        loss_G_S2T = self.criterion_gan(self.netD_T(fake_T), real=True)
        # GAN loss D_S(G_T2S(B))
        loss_G_T2S = self.criterion_gan(self.netD_S(fake_S), real=True)
        # Cycle loss || G_T2S(G_S2T(S)) - S||
        loss_cycle_S = self.criterion_cycle(rec_S, real_S) * self.trade_off_cycle
        # Cycle loss || G_S2T(G_T2S(T)) - T||
        loss_cycle_T = self.criterion_cycle(rec_T, real_T) * self.trade_off_cycle
        # Identity loss
        # G_S2T should be identity if real_T is fed: ||G_S2T(real_T) - real_T||
        identity_T = self.model.netG_S2T(real_T)
        loss_identity_T = self.criterion_identity(identity_T, real_T) * self.trade_off_identity
        # G_T2S should be identity if real_S is fed: ||G_T2S(real_S) - real_S||
        identity_S = self.model.netG_T2S(real_S)
        loss_identity_S = self.criterion_identity(identity_S, real_S) * self.trade_off_identity
        # Semantic loss
        pred_fake_T = self.predict(fake_T)[0]
        pred_real_S = self.predict(real_S)[0]
        loss_semantic_S2T = self.criterion_semantic(pred_fake_T, label_s) * self.trade_off_semantic
        
        pred_fake_S = self.predict(fake_S)[0]
        pred_real_T = self.predict(real_T)[0]
        loss_semantic_T2S = self.criterion_semantic(pred_fake_S, pred_real_T) * self.trade_off_semantic
        
        task_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion, p_imp=imp_pred)
        # combined loss and calculate gradients
        loss_G = loss_G_S2T + loss_G_T2S + loss_cycle_S + loss_cycle_T + \
                 loss_identity_S + loss_identity_T + loss_semantic_S2T + loss_semantic_T2S + task_loss
        
        self.manual_backward(loss_G, retain_graph=True)
        optimizer_G.step()
        optimizer_G.zero_grad()
        self.untoggle_optimizer(optimizer_G)


        self.toggle_optimizer(optimizer_D)
        set_requires_grad(self.netD_S, True)
        set_requires_grad(self.netD_T, True)
        optimizer_D.zero_grad()
        # Calculate GAN loss for discriminator D_S
        fake_S_ = fake_S.detach()
        fake_T_ = fake_T.detach()
        loss_D_S = 0.5 * (self.criterion_gan(self.netD_S(real_S), True) + self.criterion_gan(self.netD_S(fake_S_), False))
        # Calculate GAN loss for discriminator D_T
        
        loss_D_T = 0.5 * (self.criterion_gan(self.netD_T(real_T), True) + self.criterion_gan(self.netD_T(fake_T_), False))
        loss_D = loss_D_S + loss_D_T
        self.manual_backward(loss_D)
        optimizer_D.step()
        optimizer_D.zero_grad()
        self.untoggle_optimizer(optimizer_D)
        self.log('train/loss_G_S2T', loss_G_S2T.item(), on_epoch=True, on_step=True)
        self.log('train/loss_G_T2S', loss_G_T2S.item(), on_epoch=True, on_step=True)
        self.log('train/loss_D_S', loss_D_S.item(), on_epoch=True, on_step=True)
        self.log('train/loss_D_T', loss_D_T.item(), on_epoch=True, on_step=True)
        self.log('train/loss_cycle_S', loss_cycle_S.item(), on_epoch=True, on_step=True)
        self.log('train/loss_cycle_T', loss_cycle_T.item(), on_epoch=True, on_step=True)
        self.log('train/loss_identity_S', loss_identity_S.item(), on_epoch=True, on_step=True)
        self.log('train/loss_identity_T', loss_identity_T.item(), on_epoch=True, on_step=True)
        self.log('train/loss_semantic_S2T', loss_semantic_S2T.item(), on_epoch=True, on_step=True)
        self.log('train/loss_semantic_T2S', loss_semantic_T2S.item(), on_epoch=True, on_step=True)
        self.log('train/task_loss', task_loss.item(), on_epoch=True, on_step=True)
        
        
        
    def validation_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, representations = self.model(features)
        # filter the conversion_pred where click is 0
        val_loss = self.loss(click_pred, conversion_pred, click_conversion_pred, click, conversion, p_imp=imp_pred)

        self.log("val/loss", val_loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        click, conversion, features = self.batch_transform(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, representations = self.model(features)
        conversion_pred_filter = conversion_pred[click == 1]
        conversion_filter = conversion[click == 1]
        self.ctr_auc.update(click_pred, click)
        self.cvr_auc.update(conversion_pred_filter, conversion_filter)
        self.ctcvr_auc.update(click_conversion_pred, click * conversion)

    def configure_optimizers(self):    
        # define optimizer and lr scheduler
        optimizer_G = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_D = torch.optim.Adam(itertools.chain(self.netD_S.parameters(), self.netD_T.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer_D, optimizer_G   
    
    
    