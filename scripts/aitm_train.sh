nohup python src/train.py \
-m data=ccp,fr,us,nl,es \
model._target_=src.models.common.MultiTaskLitModel \
model.model._target_=src.models.aitm.AITM \
model.loss._target_=src.models.aitm.Loss_AITM \
trainer.max_epochs=1 trainer.min_epochs=1 \
experiment=esmm_experiment \
logger=csv \
test=True data.debug=True >logs/out.log 2>&1 & \


nohup python src/train.py \
-m data=ccp,fr,us,nl,es \
trainer.max_epochs=1 trainer.min_epochs=1 \
experiment=esmm_experiment,aitm_experiment \
test=True data.debug=False >logs/out.log 2>&1 & \