import os
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torchmetrics.classification import BinaryAUROC

class MultiTaskCallback(Callback):
    def __init__(
        self,
    ):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("start training")
        
    def dim_zero_cat(self, x):
        """Concatenation along the zero dimension."""
        if isinstance(x, torch.Tensor):
            return x
        x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
        if not x:  # empty list
            raise ValueError("No samples to concatenate")
        return torch.cat(x, dim=0)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # judge if a attribute is not None

        if hasattr(pl_module, 'val_ctr_auc'):
            self.log("val/ctr_auc", pl_module.val_ctr_auc.compute())
            pval_ctr = self.dim_zero_cat(pl_module.val_ctr_auc.preds)
            pval_ctr_target =  self.dim_zero_cat(pl_module.val_ctr_auc.target)
            self.log("val/ctr_logloss", torch.nn.functional.binary_cross_entropy(pval_ctr, pval_ctr_target))
            pl_module.val_ctr_auc.reset()
        if hasattr(pl_module, 'val_cvr_auc'):
            self.log("val/cvr_auc", pl_module.val_cvr_auc.compute())
            pval_cvr = self.dim_zero_cat(pl_module.val_cvr_auc.preds)
            pval_cvr_target =  self.dim_zero_cat(pl_module.val_cvr_auc.target)
            self.log("val/cvr_logloss", torch.nn.functional.binary_cross_entropy(pval_cvr, pval_cvr_target))
            pl_module.val_cvr_auc.reset()
        if hasattr(pl_module, 'val_ctcvr_auc'):
            self.log("val/ctcvr_auc", pl_module.val_ctcvr_auc.compute())
            pval_ctcvr = self.dim_zero_cat(pl_module.val_ctcvr_auc.preds)
            pval_ctcvr_target =  self.dim_zero_cat(pl_module.val_ctcvr_auc.target)
            self.log("val/ctcvr_logloss", torch.nn.functional.binary_cross_entropy(pval_ctcvr, pval_ctcvr_target))
            pl_module.val_ctcvr_auc.reset()
        if hasattr(pl_module, 'val_cvr_unclick_auc'):
            self.log("val/cvr_unclick_auc", pl_module.val_cvr_unclick_auc.compute())
            pval_cvr_unclick = self.dim_zero_cat(pl_module.val_cvr_unclick_auc.preds)
            pval_cvr_unclick_target =  self.dim_zero_cat(pl_module.val_cvr_unclick_auc.target)
            self.log("val/cvr_unclick_logloss", torch.nn.functional.binary_cross_entropy(pval_cvr_unclick, pval_cvr_unclick_target))
            pl_module.cvr_unclick_auc.reset()
        if hasattr(pl_module, 'val_cvr_exposure_auc'):
            self.log("val/cvr_exposure_auc", pl_module.val_cvr_exposure_auc.compute())
            pval_cvr_exposure = self.dim_zero_cat(pl_module.val_cvr_exposure_auc.preds)
            pval_cvr_exposure_target =  self.dim_zero_cat(pl_module.val_cvr_exposure_auc.target)
            self.log("val/cvr_exposure_logloss", torch.nn.functional.binary_cross_entropy(pval_cvr_exposure, pval_cvr_exposure_target))
            pl_module.val_cvr_exposure_auc.reset()
    
    def on_test_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module, 'ctr_auc'):
            self.log("test/ctr_auc", pl_module.ctr_auc.compute())
            pctr = self.dim_zero_cat(pl_module.ctr_auc.preds)
            pctr_target =  self.dim_zero_cat(pl_module.ctr_auc.target)
            ctr_logloss = torch.nn.functional.binary_cross_entropy(pctr, pctr_target)
            self.log("test/ctr_logloss", ctr_logloss)
            pl_module.ctr_auc.reset()
        if hasattr(pl_module, 'cvr_auc'):
            self.log("test/cvr_auc", pl_module.cvr_auc.compute())
            pcvr = self.dim_zero_cat(pl_module.cvr_auc.preds)
            pcvr_target =  self.dim_zero_cat(pl_module.cvr_auc.target)
            self.log("test/cvr_logloss", torch.nn.functional.binary_cross_entropy(pcvr, pcvr_target))
            pl_module.cvr_auc.reset()
        if hasattr(pl_module, 'ctcvr_auc'):
            self.log("test/ctcvr_auc", pl_module.ctcvr_auc.compute())
            pctcvr = self.dim_zero_cat(pl_module.ctcvr_auc.preds)
            pctcvr_target =  self.dim_zero_cat(pl_module.ctcvr_auc.target)
            self.log("test/ctcvr_logloss", torch.nn.functional.binary_cross_entropy(pctcvr, pctcvr_target))
            pl_module.ctcvr_auc.reset()        
        if hasattr(pl_module, 'cvr_unclick_auc'):
            self.log("test/cvr_unclick_auc", pl_module.cvr_unclick_auc.compute())
            pcvr_unclick = self.dim_zero_cat(pl_module.cvr_unclick_auc.preds)
            pcvr_unclick_target =  self.dim_zero_cat(pl_module.cvr_unclick_auc.target)
            self.log("test/cvr_unclick_logloss", torch.nn.functional.binary_cross_entropy(pcvr_unclick, pcvr_unclick_target))
            pl_module.cvr_unclick_auc.reset()
        if hasattr(pl_module, 'cvr_exposure_auc'):
            self.log("test/cvr_exposure_auc", pl_module.cvr_exposure_auc.compute())
            pcvr_exposure = self.dim_zero_cat(pl_module.cvr_exposure_auc.preds)
            pcvr_exposure_target =  self.dim_zero_cat(pl_module.cvr_exposure_auc.target)
            self.log("test/cvr_exposure_logloss", torch.nn.functional.binary_cross_entropy(pcvr_exposure, pcvr_exposure_target))
            pl_module.cvr_exposure_auc.reset()
        
        # pctr = self.dim_zero_cat(pl_module.ctr_auc.preds)
        # target =  self.dim_zero_cat(pl_module.ctr_auc.target)
        # m = target.sum()
        # n = len(target) - target.sum()
        # p = m / (m + n)
        # q = m / (m + 2*n)
        # self.log("pctr_mean", pctr.mean())
        # self.log("m", m)
        # self.log('n', n)
        # self.log('p', p)
        # self.log('q', q)
        
class MultiTaskCallback_Plot0(Callback):
    def __init__(
        self, 
        fig_dir: str = './outputs'
    ):
        super().__init__()
        self.fig_dir = fig_dir 

    def on_test_start(self, trainer, pl_module):
        self.click_pred_test = []
        self.conversion_pred_test = []
        self.conversion_pred_filter_test = []
        self.click_conversion_pred_test = []
        self.click_label_test = []
        self.conversion_label_test = []
        self.conversion_label_filter_test = []
        self.click_conversion_label_test = []
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.click_pred_test.append(outputs['click_pred_test'])
        self.conversion_pred_test.append(outputs['conversion_pred_test'])
        self.conversion_pred_filter_test.append(outputs['conversion_pred_filter_test'])
        self.click_conversion_pred_test.append(outputs['click_conversion_pred_test'])
        self.click_label_test.append(outputs['click_label_test'])
        self.conversion_label_test.append(outputs['conversion_label_test'])
        self.conversion_label_filter_test.append(outputs['conversion_label_filter_test'])
        self.click_conversion_label_test.append(outputs['click_conversion_label_test'])

    def on_test_epoch_end(self, trainer, pl_module):
        self.log("test/ctr_auc", pl_module.ctr_auc.compute())
        self.log("test/cvr_auc", pl_module.cvr_auc.compute())
        self.log("test/ctcvr_auc", pl_module.ctcvr_auc.compute())
        pl_module.ctr_auc.reset()
        pl_module.cvr_auc.reset()
        pl_module.ctcvr_auc.reset()
        
        self.click_label_test = torch.cat(self.click_label_test).cpu().detach().numpy()
        self.click_pred_test = torch.cat(self.click_pred_test).cpu().detach().numpy()
        self.conversion_pred_test = torch.cat(self.conversion_pred_test).cpu().detach().numpy()
        self.conversion_label_test = torch.cat(self.conversion_label_test).cpu().detach().numpy()
        self.conversion_pred_filter_test = torch.cat(self.conversion_pred_filter_test).cpu().detach().numpy()
        self.conversion_label_filter_test = torch.cat(self.conversion_label_filter_test).cpu().detach().numpy()
        
        propensity_scores = 1 / self.click_pred_test
        propensity_scores_click = 1 / self.click_pred_test[self.click_label_test==1]
        propensity_scores_unclick = 1 / self.click_pred_test[self.click_label_test==0]
        propensity_scores_click_expection = propensity_scores_click.mean()
        
        print('propensity_scores_click_expection', propensity_scores_click_expection)
        print('propensity_scores_click_variance', propensity_scores_click.var())

        
        print('conversion_label_mean', self.conversion_label_test.mean())
        print('conversion_pred_mean', self.conversion_pred_test.mean())
        print('conversion_pred_variance', self.conversion_pred_test.var())
        
        print('conversion_label_filter_mean', self.conversion_label_filter_test.mean())
        print('conversion_pred_filter_mean', self.conversion_pred_filter_test.mean())
        print('conversion_pred_filter_variance', self.conversion_pred_filter_test.var()) 
        
        if os.path.exists(self.fig_dir) == False:
            os.makedirs(self.fig_dir)
            
        
        plt.figure(figsize=(12,9), dpi=200)
        plt.hist(self.conversion_pred_filter_test, bins=100, color='whitesmoke', alpha=1, edgecolor='black', density=True, label='Count')
        # Add a vertical line at the mean
        pred_mean = self.conversion_pred_filter_test.mean()
        pred_var = self.conversion_pred_filter_test.var()
        # var show in scientific notation
        pred_var = '%.2E' % pred_var
        label_mean = self.conversion_label_filter_test.mean()
        increase = (pred_mean - label_mean) / label_mean
        # get round of increase
        if increase > 0:
            increase = '+' + str(round(increase*100, 2))
        else:
            increase = str(round(increase*100, 2))
        increase = increase + '%'
        

        plt.axvline(pred_mean, color='r', linestyle='dashed', linewidth=2, label='Prediction Mean:{:.4f}({}), Variance:{}'.format(pred_mean, increase, pred_var))
        plt.axvline(label_mean, color='g', linestyle='dashed', linewidth=2, label='Label Mean:{:.4f}'.format(label_mean))

        # Display the mean value on the plot
        # plt.text(pred_mean+1, 0.03, 'Mean: {:.2f}'.format(pred_mean), fontsize=20, color='r')

        plt.xlabel('CVR prediction values', fontsize=28, color='black')
        plt.ylabel('Count', fontsize=28, color='black')
        plt.xticks(fontsize=20, color='black')
        plt.yticks(fontsize=20, color='black')
        plt.style.use('fast')
        plt.legend(fontsize=20)
        plt.savefig(self.fig_dir +'/conversion_pred_filter_test' + '.pdf', format='pdf')
        plt.clf()
        
        plt.hist(self.conversion_pred_test, bins=100,  edgecolor='black', alpha=0.5,  color='orange')
        plt.savefig(self.fig_dir+'/conversion_pred_test' + '.png', format='png', dpi=300)
        plt.clf()
        
        plt.hist(self.click_pred_test, bins=100,  edgecolor='black', alpha=0.5,  color='orange')
        plt.savefig(self.fig_dir+'/click_pred_test' + '.png', format='png', dpi=300)
        plt.clf()
   
class MultiTaskCallbackPlot0(Callback):
    def __init__(
        self,
        fig_dir='./outputs'
    ):
        super().__init__()
        self.path = fig_dir
        self.auc_metric  = BinaryAUROC()

    def on_train_start(self, trainer, pl_module):
        print("start training")
        
    def dim_zero_cat(self, x):
        """Concatenation along the zero dimension."""
        if isinstance(x, torch.Tensor):
            return x
        x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
        if not x:  # empty list
            raise ValueError("No samples to concatenate")
        return torch.cat(x, dim=0)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # judge if a attribute is not None

        if hasattr(pl_module, 'val_ctr_auc'):
            self.log("val/ctr_auc", pl_module.val_ctr_auc.compute())
            pval_ctr = self.dim_zero_cat(pl_module.val_ctr_auc.preds)
            pval_ctr_target =  self.dim_zero_cat(pl_module.val_ctr_auc.target)
            self.log("val/ctr_logloss", torch.nn.functional.binary_cross_entropy(pval_ctr, pval_ctr_target))
            pl_module.val_ctr_auc.reset()
        if hasattr(pl_module, 'val_cvr_auc'):
            self.log("val/cvr_auc", pl_module.val_cvr_auc.compute())
            pval_cvr = self.dim_zero_cat(pl_module.val_cvr_auc.preds)
            pval_cvr_target =  self.dim_zero_cat(pl_module.val_cvr_auc.target)
            self.log("val/cvr_logloss", torch.nn.functional.binary_cross_entropy(pval_cvr, pval_cvr_target))
            pl_module.val_cvr_auc.reset()
        if hasattr(pl_module, 'val_ctcvr_auc'):
            self.log("val/ctcvr_auc", pl_module.val_ctcvr_auc.compute())
            pval_ctcvr = self.dim_zero_cat(pl_module.val_ctcvr_auc.preds)
            pval_ctcvr_target =  self.dim_zero_cat(pl_module.val_ctcvr_auc.target)
            self.log("val/ctcvr_logloss", torch.nn.functional.binary_cross_entropy(pval_ctcvr, pval_ctcvr_target))
            pl_module.val_ctcvr_auc.reset()
        if hasattr(pl_module, 'val_cvr_unclick_auc'):
            self.log("val/cvr_unclick_auc", pl_module.val_cvr_unclick_auc.compute())
            pval_cvr_unclick = self.dim_zero_cat(pl_module.val_cvr_unclick_auc.preds)
            pval_cvr_unclick_target =  self.dim_zero_cat(pl_module.val_cvr_unclick_auc.target)
            self.log("val/cvr_unclick_logloss", torch.nn.functional.binary_cross_entropy(pval_cvr_unclick, pval_cvr_unclick_target))
            pl_module.cvr_unclick_auc.reset()
        if hasattr(pl_module, 'val_cvr_exposure_auc'):
            self.log("val/cvr_exposure_auc", pl_module.val_cvr_exposure_auc.compute())
            pval_cvr_exposure = self.dim_zero_cat(pl_module.val_cvr_exposure_auc.preds)
            pval_cvr_exposure_target =  self.dim_zero_cat(pl_module.val_cvr_exposure_auc.target)
            self.log("val/cvr_exposure_logloss", torch.nn.functional.binary_cross_entropy(pval_cvr_exposure, pval_cvr_exposure_target))
            pl_module.val_cvr_exposure_auc.reset()
    
    def on_test_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module, 'ctr_auc'):
            self.log("test/ctr_auc", pl_module.ctr_auc.compute())
            pctr = self.dim_zero_cat(pl_module.ctr_auc.preds)
            pctr_target =  self.dim_zero_cat(pl_module.ctr_auc.target)
            ctr_logloss = torch.nn.functional.binary_cross_entropy(pctr, pctr_target)
            self.log("test/ctr_logloss", ctr_logloss)
            self.log("test/ctr_label_mean", pctr_target.mean())
            self.log("test/ctr_pred_mean", pctr.mean())
            self.log("test/ctr_pred_variance", pctr.var())
            pl_module.ctr_auc.reset()
        if hasattr(pl_module, 'cvr_auc'):
            self.log("test/cvr_auc", pl_module.cvr_auc.compute())
            pcvr = self.dim_zero_cat(pl_module.cvr_auc.preds)
            pcvr_target =  self.dim_zero_cat(pl_module.cvr_auc.target)
            self.log("test/cvr_logloss", torch.nn.functional.binary_cross_entropy(pcvr, pcvr_target))
            self.log("test/cvr_label_mean", pcvr_target.mean())
            self.log("test/cvr_pred_mean", pcvr.mean())
            self.log("test/cvr_pred_variance", pcvr.var())
            pl_module.cvr_auc.reset()
        if hasattr(pl_module, 'cvr_unclick_auc'):
            self.log("test/cvr_unclick_auc", pl_module.cvr_unclick_auc.compute())
            pcvr_unclick = self.dim_zero_cat(pl_module.cvr_unclick_auc.preds)
            pcvr_unclick_target =  self.dim_zero_cat(pl_module.cvr_unclick_auc.target)
            self.log("test/cvr_unclick_logloss", torch.nn.functional.binary_cross_entropy(pcvr_unclick, pcvr_unclick_target))
            self.log("test/cvr_unclick_label_mean", pcvr_unclick_target.mean())
            self.log("test/cvr_unclick_pred_mean", pcvr_unclick.mean())
            self.log("test/cvr_unclick_pred_variance", pcvr_unclick.var())
            pl_module.cvr_unclick_auc.reset()
        if hasattr(pl_module, 'cvr_exposure_auc'):
            self.log("test/cvr_exposure_auc", pl_module.cvr_exposure_auc.compute())
            pcvr_exposure = self.dim_zero_cat(pl_module.cvr_exposure_auc.preds)
            pcvr_exposure_target =  self.dim_zero_cat(pl_module.cvr_exposure_auc.target)
            self.log("test/cvr_exposure_logloss", torch.nn.functional.binary_cross_entropy(pcvr_exposure, pcvr_exposure_target))
            self.log("test/cvr_exposure_label_mean", pcvr_exposure_target.mean())
            self.log("test/cvr_exposure_pred_mean", pcvr_exposure.mean())
            self.log("test/cvr_exposure_pred_variance", pcvr_exposure.var())
            pl_module.cvr_exposure_auc.reset()
        if hasattr(pl_module, 'ctcvr_auc'):
            pctcvr = self.dim_zero_cat(pl_module.ctcvr_auc.preds)
            pctcvr_target =  self.dim_zero_cat(pl_module.ctcvr_auc.target)
            pctcvr_unclick = pctcvr[pctr_target == 0]
            pctcvr_target_unclick = pctcvr_target[pctr_target == 0]
            pctcvr_click = pctcvr[pctr_target == 1]
            pctcvr_target_click = pctcvr_target[pctr_target == 1]
            self.log("test/ctcvr_auc", pl_module.ctcvr_auc.compute())
            self.log("test/ctcvr_auc_unclick", self.auc_metric(pctcvr_unclick, pctcvr_target_unclick))
            self.log("test/ctcvr_auc_click", self.auc_metric(pctcvr_click, pctcvr_target_click))
            self.log("test/ctcvr_logloss", torch.nn.functional.binary_cross_entropy(pctcvr, pctcvr_target))
            self.log("test/ctcvr_logloss_unclick", torch.nn.functional.binary_cross_entropy(pctcvr_unclick, pctcvr_target_unclick))
            self.log("test/ctcvr_logloss_click", torch.nn.functional.binary_cross_entropy(pctcvr_click, pctcvr_target_click))
            self.log("test/ctcvr_label_mean", pctcvr_target.mean())
            self.log("test/ctcvr_pred_mean", pctcvr.mean())
            self.log("test/ctcvr_pred_variance", pctcvr.var())
            self.log("test/ctcvr_label_mean_unclick", pctcvr_target_unclick.mean())
            self.log("test/ctcvr_pred_mean_unclick", pctcvr_unclick.mean())
            self.log("test/ctcvr_pred_variance_unclick", pctcvr_unclick.var())
            self.log("test/ctcvr_label_mean_click", pctcvr_target_click.mean())
            self.log("test/ctcvr_pred_mean_click", pctcvr_click.mean())
            self.log("test/ctcvr_pred_variance_click", pctcvr_click.var())
            pl_module.ctcvr_auc.reset()

        
        # pctr = self.dim_zero_cat(pl_module.ctr_auc.preds)
        # target =  self.dim_zero_cat(pl_module.ctr_auc.target)
        # m = target.sum()
        # n = len(target) - target.sum()
        # p = m / (m + n)
        # q = m / (m + 2*n)
        # self.log("pctr_mean", pctr.mean())
        # self.log("m", m)
        # self.log('n', n)
        # self.log('p', p)
        # self.log('q', q)

if __name__ == "__main__":
    config = dict(
        # basic
        global_seed=2020,
        vocabulary_size=vocabulary_size,
        single_embedding_vocabulary_size=737946,
        sigmoid=False,
        # dataloader
        num_workers=8,
        num_workers_test=8,
        persistent_workers=False,
        test_persistent_workers=False,
        shuffle=True,
        # early stop
        using_early_stop=False,
        monitor_metric="val_loss_epoch",
        mode="min",
        earlystop_epoch=1,
        min_delta=0.0002,
        # checkpoint
        using_model_checkpoint=False,
        checkpoint_monitor_metric="val_loss_epoch",
        checkpoint_monitor_mode="min",
        save_top_k=1,
        check_point_path="./out/",
        # model structure and hyperparameters
        total_epoch=1,
        batch_size=8000,
        learning_rate=0.001,
        embedding_size=5,
        weight_decay=1e-6,
        presion="32",
        expert_num=8,
        task_num=3,
        embed_output_dim=90,
        bottom_mlp_dims=[256],
        tower_mlp_dims=[128, 64, 32],
        dropout_expert=[0.3],
        dropout_tower=[0.1, 0.3, 0.3],
        # student hyperparameters
        student_model=CGAM_V3,
        student_model_loss=Loss_Student,
        batch_type="fr",
        counterfact_mode="IPW",
        # contrastive loss
        hard_positive_topk=50,
        hard_negative_topk=50,
        contrastive_loss_proportion=0.4,
        contrastive_loss=ContrastiveLoss,
        mining_target="mining_click_embedding",  # 'mining_click_embedding','mining_conversion', 'mining_click', 'mining_both'
        contrastive_type="CosineSimilarity_ntxent",  #'CosineSimilarity_ntxent','swap_positive_and_negative_triplet', 'basic_triplet', 'inverse_triplet', 'inverse_exp_triplet', 'LpDistance_ntxent', 'negative_set_to_1_LpDistance_ntxent'
        temperature=0.5,
        negative_setting_number=1,
        normalize_embeddings=False,
        # basic loss
        cvr_nonclick_trade_off=0,
        ctr_nonclick_trade_off=0.45,
        ctr_loss_proportion=1,
        cvr_loss_proportion=1,
        ctcvr_loss_proportion=0.1,
        weight_clicked=0.5,
        weight_conversion=0.1,
        p_ctr_down_clip=0.05,
        # wandb
        using_wandb=False,
        project="CVR prediction test",
        name="student-hard_mining_and_ctr_debias",
        group_name="CGAM-IPW-constractive",
        log_model=False,
        profiler=None,  # 'simple' , 'pytorch'
    )
    debug = False
    config["batch_type"] = "fr"
    config["using_wandb"] = False
    dataloaders, config["field_dims"], config["numerical_num"] = get_dataloaders(
        config, debug=debug
    )

    config["test_model_path"] = "./out/ctr_base_model.ckpt"
    config["png_name"] = "ctr_base_model"
    config["figure_dir"] = "./out/figure_train/ctr_base_model"
    # config['test_model_path'] = './out/contrastive_no_cliping_ctr.ckpt'
    # config['png_name'] = 'contrastive_no_cliping_ctr'
    # config['figure_dir'] = './out/figure_train/contrastive_no_cliping_ctr'
    if not os.path.exists(config["figure_dir"]):
        os.makedirs(config["figure_dir"])

    dataloaders["test"] = dataloaders["train"]
    litmodel = LitCGCM.load_from_checkpoint(config["test_model_path"], config=config)
    litcallback = LitCallback(config)
    single_train(config, litmodel, litcallback, dataloaders, train=False, test=True)
