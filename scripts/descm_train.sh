# single
# python  src/train.py -m  data=ccp model.model._target_=src.models.descm.ESCM,src.models.descm.DESCM_Embedding experiment=descm_experiment test=False data.debug=True logger.wandb.project=lightning-hydra-template


# grid search 
# nohup python src/train.py -m data=ccp,fr,us,nl,es model.model._target_=src.models.descm.DESCM_Embedding_DA_1_1 model.loss._target_=src.models.common.Basic_Loss model._target_=src.models.descm.MultiTaskLitModel_DA_click_to_impression experiment=descm_experiment test=True data.debug=True >logs/out.log 2>&1 &

# nohup python src/train.py -m data=ccp,fr,us,nl,es model.loss._target_=src.models.common.Basic_Loss,src.models.common.Impression_CTR_IPW_Loss,src.models.common.Unclick_CTR_IPW_Loss,src.models.common.Click_CTR_IPW_Loss  experiment=descm_experiment test=True data.debug=False >logs/out.log 2>&1 &

python src/train.py -m data=fr experiment=descm_experiment test=True data.debug=True logger=csv

# python src/train.py -m hparams_search=descm_optuna experiment=descm_experiment data.debug=True test=True

nohup python src/train.py \
-m data=ccp,fr,us,nl,es \
model._target_=src.models.common.MultiTaskLitModel \
model.model._target_=src.models.descm.ESCM \
model.loss._target_=src.models.common.IPS_GPL_Loss,src.models.common.DR_GPL_Loss,src.models.common.MRDR_GPL_Loss,src.models.common.DRMSE_GPL_Loss,src.models.common.IPS_VR_Loss,src.models.common.DR_VR_Loss \
trainer.max_epochs=1 trainer.min_epochs=1 \
logger=csv \
experiment=descm_experiment \
test=True data.debug=False >logs/out.log 2>&1 & \


# nohup python src/train.py \
# -m data=es \
# model._target_=src.models.common.MultiTaskLitModel \
# model.model._target_=src.models.descm.ESCM \
# model.loss._target_=src.models.common.DR_Loss \
# trainer.max_epochs=1 trainer.min_epochs=1 \
# experiment=descm_experiment \
# logger=csv \
# test=True data.debug=False >logs/out.log 2>&1 & \

# # ploting multi ccp
# python src/train.py \
# -m data=ccp \
# model._target_=src.models.common.MultiTaskLitModel \
# model.model._target_=src.models.descm.ESCM \
# model.loss._target_=src.models.common.MMoE_Multi_Loss \
# trainer.max_epochs=1 trainer.min_epochs=1 \
# experiment=descm_experiment \
# logger=csv \
# callbacks=multi_task_callbacks_plot \
# callbacks.multi_task_callback_plot.fig_dir=./outputs/multicvr_figures_ccp \
# ckpt_path_test=/media/user/data1/fk/recom/CVR_Prediction_Hydra/logs/train/multiruns/2024-05-09_11-35-16/0/checkpoints/last.ckpt \
# train=False test=True data.debug=False

# # ploting dr ccp
# python src/train.py \
# -m data=ccp \
# model._target_=src.models.common.MultiTaskLitModel \
# model.model._target_=src.models.descm.ESCM \
# model.loss._target_=src.models.common.DR_Loss \
# trainer.max_epochs=1 trainer.min_epochs=1 \
# experiment=descm_experiment \
# logger=csv \
# callbacks=multi_task_callbacks_plot \
# callbacks.multi_task_callback_plot.fig_dir=./outputs/dr_figures_ccp \
# ckpt_path_test=/media/user/data1/fk/recom/CVR_Prediction_Hydra/logs/train/multiruns/2024-05-09_11-35-16/1/checkpoints/last.ckpt \
# train=False test=True data.debug=False


# # ploting multi fr
# python src/train.py \
# -m data=fr \
# model._target_=src.models.common.MultiTaskLitModel \
# model.model._target_=src.models.descm.ESCM \
# model.loss._target_=src.models.common.MMoE_Multi_Loss \
# trainer.max_epochs=1 trainer.min_epochs=1 \
# experiment=descm_experiment \
# logger=csv \
# callbacks=multi_task_callbacks_plot \
# callbacks.multi_task_callback_plot.fig_dir=./outputs/multicvr_figures_fr \
# ckpt_path_test=/media/user/data1/fk/recom/CVR_Prediction_Hydra/logs/train/multiruns/2024-05-09_11-35-16/2/checkpoints/last.ckpt \
# train=False test=True data.debug=False


# # ploting dr fr
# python src/train.py \
# -m data=fr \
# model._target_=src.models.common.MultiTaskLitModel \
# model.model._target_=src.models.descm.ESCM \
# model.loss._target_=src.models.common.DR_Loss \
# trainer.max_epochs=1 trainer.min_epochs=1 \
# experiment=descm_experiment \
# logger=csv \
# callbacks=multi_task_callbacks_plot \
# callbacks.multi_task_callback_plot.fig_dir=./outputs/dr_figures_fr \
# ckpt_path_test=/media/user/data1/fk/recom/CVR_Prediction_Hydra/logs/train/multiruns/2024-05-09_11-35-16/3/checkpoints/last.ckpt \
# train=False test=True data.debug=False

# # ploting multi
# python src/train.py \
# -m data=us \
# model._target_=src.models.common.MultiTaskLitModel \
# model.model._target_=src.models.descm.ESCM \
# model.loss._target_=src.models.common.MMoE_Multi_Loss \
# trainer.max_epochs=1 trainer.min_epochs=1 \
# experiment=descm_experiment \
# logger=csv \
# callbacks=multi_task_callbacks_plot \
# callbacks.multi_task_callback_plot.fig_dir=./outputs/multicvr_figures_us \
# ckpt_path_test=/media/user/data1/fk/recom/CVR_Prediction_Hydra/logs/train/multiruns/2024-05-09_11-35-16/4/checkpoints/last.ckpt \
# train=False test=True data.debug=False


# # ploting dr
# python src/train.py \
# -m data=us \
# model._target_=src.models.common.MultiTaskLitModel \
# model.model._target_=src.models.descm.ESCM \
# model.loss._target_=src.models.common.DR_Loss \
# trainer.max_epochs=1 trainer.min_epochs=1 \
# experiment=descm_experiment \
# logger=csv \
# callbacks=multi_task_callbacks_plot \
# callbacks.multi_task_callback_plot.fig_dir=./outputs/dr_figures_us \
# ckpt_path_test=/media/user/data1/fk/recom/CVR_Prediction_Hydra/logs/train/multiruns/2024-05-09_11-35-16/5/checkpoints/last.ckpt \
# train=False test=True data.debug=False

# # ploting multi
# python src/train.py \
# -m data=nl \
# model._target_=src.models.common.MultiTaskLitModel \
# model.model._target_=src.models.descm.ESCM \
# model.loss._target_=src.models.common.MMoE_Multi_Loss \
# trainer.max_epochs=1 trainer.min_epochs=1 \
# experiment=descm_experiment \
# logger=csv \
# callbacks=multi_task_callbacks_plot \
# callbacks.multi_task_callback_plot.fig_dir=./outputs/multicvr_figures_nl \
# ckpt_path_test=/media/user/data1/fk/recom/CVR_Prediction_Hydra/logs/train/multiruns/2024-05-09_11-35-16/6/checkpoints/last.ckpt \
# train=False test=True data.debug=False


# # ploting dr
# python src/train.py \
# -m data=nl \
# model._target_=src.models.common.MultiTaskLitModel \
# model.model._target_=src.models.descm.ESCM \
# model.loss._target_=src.models.common.DR_Loss \
# trainer.max_epochs=1 trainer.min_epochs=1 \
# experiment=descm_experiment \
# logger=csv \
# callbacks=multi_task_callbacks_plot \
# callbacks.multi_task_callback_plot.fig_dir=./outputs/dr_figures_nl \
# ckpt_path_test=/media/user/data1/fk/recom/CVR_Prediction_Hydra/logs/train/multiruns/2024-05-09_11-35-16/7/checkpoints/last.ckpt \
# train=False test=True data.debug=False


# # plot res ccp
# python src/train.py \
# -m data=ccp \
# model._target_=src.models.descm_res.MultiTaskLitModel_Res \
# model.model._target_=src.models.descm_res.DESCM_Embedding_Res_Cross1 \
# model.loss._target_=src.models.descm_res.Basic_Loss_Res \
# experiment=descm_res_experiment \
# logger=csv \
# callbacks=multi_task_callbacks_plot \
# callbacks.multi_task_callback_plot.fig_dir=./outputs/res_figures_ccp \
# ckpt_path_test=/media/user/data1/fk/recom/CVR_Prediction_Hydra/logs/train/multiruns/res_model/0/checkpoints/last.ckpt \
# train=False test=True data.debug=False

# # plot res fr
# python src/train.py \
# -m data=fr \
# model._target_=src.models.descm_res.MultiTaskLitModel_Res \
# model.model._target_=src.models.descm_res.DESCM_Embedding_Res_Cross1 \
# model.loss._target_=src.models.descm_res.Basic_Loss_Res \
# experiment=descm_res_experiment \
# logger=csv \
# callbacks=multi_task_callbacks_plot \
# callbacks.multi_task_callback_plot.fig_dir=./outputs/res_figures_fr \
# ckpt_path_test=/media/user/data1/fk/recom/CVR_Prediction_Hydra/logs/train/multiruns/res_model/1/checkpoints/last.ckpt \
# train=False test=True data.debug=False


# # plot res
# python src/train.py \
# -m data=nl \
# model._target_=src.models.descm_res.MultiTaskLitModel_Res \
# model.model._target_=src.models.descm_res.DESCM_Embedding_Res_Cross1 \
# model.loss._target_=src.models.descm_res.Basic_Loss_Res \
# experiment=descm_res_experiment \
# logger=csv \
# callbacks=multi_task_callbacks_plot \
# callbacks.multi_task_callback_plot.fig_dir=./outputs/res_figures_nl \
# ckpt_path_test=/media/user/data1/fk/recom/CVR_Prediction_Hydra/logs/train/multiruns/res_model/2/checkpoints/last.ckpt \
# train=False test=True data.debug=False
# # plot res
# python src/train.py \
# -m data=us \
# model._target_=src.models.descm_res.MultiTaskLitModel_Res \
# model.model._target_=src.models.descm_res.DESCM_Embedding_Res_Cross1 \
# model.loss._target_=src.models.descm_res.Basic_Loss_Res \
# experiment=descm_res_experiment \
# logger=csv \
# callbacks=multi_task_callbacks_plot \
# callbacks.multi_task_callback_plot.fig_dir=./outputs/res_figures_us \
# ckpt_path_test=/media/user/data1/fk/recom/CVR_Prediction_Hydra/logs/train/multiruns/res_model/3/checkpoints/last.ckpt \
# train=False test=True data.debug=False