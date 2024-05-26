# test full training
nohup python src/train.py \
-m data=fr \
model._target_=src.models.descm_res.MultiTaskLitModel_Res \
model.model._target_=src.models.descm_res.DESCM_Embedding_Res_Simple \
model.loss._target_=src.models.descm_res.Basic_Loss_Res_Ratio \
+model.loss.trade_off_ctr_loss_0=1 \
trainer.max_epochs=1 trainer.min_epochs=1 \
experiment=descm_res_experiment \
logger=csv \
test=True data.debug=True >logs/out.log 2>&1 & \

# simple 训练
nohup python src/train.py \
-m data=ccp \
model._target_=src.models.descm_res.MultiTaskLitModel_Res \
model.model._target_=src.models.descm_res.DESCM_Embedding_Res_Cross1 \
model.loss._target_=src.models.common.DESCM_Embedding_Res_Wopctr \
+model.model.cross_num=1 \
+model.loss.trade_off_unclick_loss=2 \
trainer.max_epochs=3 trainer.min_epochs=1 \
experiment=descm_res_experiment \
test=True data.debug=False >logs/out.log 2>&1 & \

# simple 训练
nohup python src/train.py \
-m data=es \
model._target_=src.models.descm_res.MultiTaskLitModel_Res \
model.model._target_=src.models.descm_res.DESCM_Embedding_Res_Wodecon \
model.loss._target_=src.models.common.Basic_Loss \
+model.model.cross_num=1 \
trainer.max_epochs=1 trainer.min_epochs=1 \
experiment=descm_res_experiment \
test=True data.debug=False >logs/out.log 2>&1 & \


# ploting
nohup python src/train.py \
-m data=fr \
callbacks=multi_task_callbacks_plot \
model._target_=src.models.descm_res.MultiTaskLitModel_Res \
model.model._target_=src.models.descm_res.DESCM_Embedding_Res_Cross1 \
model.loss._target_=src.models.descm_res.Basic_Loss_Res \
experiment=descm_res_experiment \
logger=csv \
ckpt_path_test=/media/user/data1/fk/recom/CVR_Prediction_Hydra/logs/train/multiruns/res_model/1/checkpoints/last.ckpt \
train=False test=True data.debug=False >logs/out.log 2>&1 &

python src/train.py \
-m data=fr \
model._target_=src.models.descm_res.MultiTaskLitModel_Res \
model.model._target_=src.models.descm_res.DESCM_Embedding_Res_Cross1 \
model.loss._target_=src.models.descm_res.Basic_Loss_Res \
experiment=descm_res_experiment \
logger=csv \
callbacks=multi_task_callbacks_plot \
callbacks.multi_task_callback_plot.fig_dir=./outputs/res_figures \
ckpt_path_test=/media/user/data1/fk/recom/CVR_Prediction_Hydra/logs/train/multiruns/res_model/1/checkpoints/last.ckpt \
train=False test=True data.debug=False