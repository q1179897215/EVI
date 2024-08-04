# single
# python  src/train.py -m  data=ccp model.model._target_=src.models.descm.ESCM,src.models.descm.DESCM_Embedding experiment=descm_experiment test=False data.debug=True logger.wandb.project=lightning-hydra-template


# grid search 
# nohup python src/train.py -m data=ccp,fr,us,nl,es model.model._target_=src.models.descm.DESCM_Embedding_DA_1_1 model.loss._target_=src.models.common.Basic_Loss model._target_=src.models.descm.MultiTaskLitModel_DA_click_to_impression experiment=descm_experiment test=True data.debug=True >logs/out.log 2>&1 &

# nohup python src/train.py -m data=ccp,fr,us,nl,es model.loss._target_=src.models.common.Basic_Loss,src.models.common.Impression_CTR_IPW_Loss,src.models.common.Unclick_CTR_IPW_Loss,src.models.common.Click_CTR_IPW_Loss  experiment=descm_experiment test=True data.debug=False >logs/out.log 2>&1 &

nohup python src/train.py \
-m data=ccp,fr,us,nl,es experiment=dcmt_experiment test=True data.debug=False trainer.max_epochs=1 trainer.min_epochs=1 >logs/out.log 2>&1 & \


python src/train.py -m data=in experiment=descm_ukd_experiment test=True data.debug=False trainer.max_epochs=1 trainer.min_epochs=1

nohup python src/train.py \
-m data=ccp \
experiment=descm_experiment test=True data.debug=False trainer.max_epochs=1 \
model._target_=src.models.descm_ukd.CvrTeacherMultiTaskLitModel \
model.loss._target_=src.models.descm_ukd.CvrTeacherMultiTaskLoss \
model.model._target_=src.models.descm_ukd.CvrTeacherMultiTask >logs/out.log 2>&1 & \

