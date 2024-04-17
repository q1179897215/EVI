# single
# python  src/train.py -m  data=ccp model.model._target_=src.models.descm.ESCM,src.models.descm.DESCM_Embedding experiment=descm_experiment test=False data.debug=True logger.wandb.project=lightning-hydra-template


# grid search 
nohup python src/train.py -m data=ccp,fr,us,nl,es model.model._target_=src.models.descm.DESCM_Embedding_DA_1_1 model.loss._target_=src.models.common.Basic_Loss experiment=descm_experiment test=True data.debug=False >logs/out.log 2>&1 &

# python src/train.py -m data=ccp,fr,us,nl,es model.model._target_=src.models.descm.ESCM,src.models.descm.DESCM_Embedding model.loss._target_=src.models.common.Basic_Loss experiment=descm_experiment test=True data.debug=False 

# python src/train.py -m hparams_search=descm_optuna experiment=descm_experiment data.debug=True test=True