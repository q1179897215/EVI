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
-m data=ccp,fr,nl,us,es \
model._target_=src.models.descm_res.MultiTaskLitModel_Res \
model.model._target_=src.models.descm_res.DESCM_Embedding_Res_Cross1 \
model.loss._target_=src.models.descm_res.Basic_Loss_Res \
+model.model.cross_num=1,2,3 \
trainer.max_epochs=3 trainer.min_epochs=1 \
experiment=descm_res_experiment \
test=True data.debug=False >logs/out.log 2>&1 & \

# simple ratio industral 训练 +model.model.cross_num=1 \
nohup python src/train.py \
-m data=in \
model._target_=src.models.descm_res.MultiTaskLitModel_Res \
model.model._target_=src.models.descm_res.DESCM_Embedding_Res_Simple \
model.loss._target_=src.models.descm_res.Basic_Loss_Res_Unclick \
+model.loss.trade_off_unclick_loss=0.1,0.5,1,2,5,10,100 \
trainer.max_epochs=1 trainer.min_epochs=1 \
data.batch_size=8000 \
experiment=descm_res_experiment \
test=True data.debug=False >logs/out.log 2>&1 &