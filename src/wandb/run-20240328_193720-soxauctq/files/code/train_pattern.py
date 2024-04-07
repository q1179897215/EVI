import os

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.models.esmm import ESMMLitModelFactory, MMoELitModelFactory
from tools.dataset import get_DataModule
from tools.utils import LitCallback, LitModel

multi_embedding_vocabulary_size = {
    "101": 238635,
    "121": 98,
    "122": 14,
    "124": 3,
    "125": 8,
    "126": 4,
    "127": 4,
    "128": 3,
    "129": 5,
    "205": 467298,
    "206": 6929,
    "207": 263942,
    "216": 106399,
    "508": 5888,
    "509": 104830,
    "702": 51878,
    "853": 37148,
    "301": 4,
}
using_feature_ids = [str(i) for i in range(330)]
using_feature_ids.pop(295)
keeped_feature_names = [i for i in range(0, len(using_feature_ids))]
default_config = dict(
    # seed
    global_seed=2020,
    # dataloader
    num_workers=8,
    num_workers_test=8,
    persistent_workers=False,
    test_persistent_workers=False,
    shuffle=True,
    shuffle_queue_size=512,
    # express data embedding
    # ali-ccp data embedding
    multi_embedding_vocabulary_size=multi_embedding_vocabulary_size,
    single_embedding_vocabulary_size=737946,
    # Indutrial data embedding
    vocabulary_size=10614790,
    using_feature_ids=using_feature_ids,
    feature_names=keeped_feature_names,
    feature_num=len(using_feature_ids),
    single_feature_len=3,
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
    batch_type="fr",
    # early stop
    using_early_stop=False,
    monitor_metric="val_loss_epoch",
    mode="min",
    earlystop_epoch=1,
    min_delta=0.0002,
    # checkpoint
    using_model_checkpoint=False,
    checkpoint_monitor_metric=None,
    checkpoint_monitor_mode="min",
    save_top_k=1,
    check_point_path="./out/",
    # wandb
    using_wandb=False,
    project="CVR prediction test",
    name="test",
    group_name="test",
    log_model=False,
    profiler=None,  # 'simple' , 'pytorch'
)


def read_lit_model(config):
    lit_model_factories = dict(
        esmm=ESMMLitModelFactory,
        mmoe=MMoELitModelFactory,
    )
    lit_model_factory = lit_model_factories[config["model_name"]]()
    another_config = config.copy()
    return lit_model_factory.get_lit_model(another_config)


def single_train(config, model_pl, litcallback, datamodule):
    pl.seed_everything(config["global_seed"], workers=True)
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    if config["using_wandb"]:
        wandb_logger = WandbLogger(
            name=config["name"],
            project=config["project"],
            config=config,
            reinit=True,
            group=config["group_name"],
            log_model=config["log_model"],
            save_code=True,
            settings=wandb.Settings(code_dir="."),
        )
    else:
        wandb_logger = None

    model_checkpoint = (
        ModelCheckpoint(
            dirpath=config["check_point_path"],
            save_top_k=config["save_top_k"],
            mode=config["checkpoint_monitor_mode"],
            monitor=config["checkpoint_monitor_metric"],
        )
        if config["using_model_checkpoint"]
        else None
    )

    early_stop = (
        EarlyStopping(
            monitor=config["monitor_metric"],
            mode=config["mode"],
            patience=config["earlystop_epoch"],
            min_delta=config["min_delta"],
        )
        if config["using_early_stop"]
        else None
    )

    call_backs = [litcallback, early_stop, model_checkpoint]
    # filter the None
    call_backs = [call_back for call_back in call_backs if call_back is not None]

    trainer = pl.Trainer(
        max_epochs=config["total_epoch"],
        callbacks=call_backs,
        logger=wandb_logger,
        profiler=config["profiler"],
        log_every_n_steps=2,
        deterministic=True,
    )
    # torch.use_deterministic_algorithms(True, warn_only=True)

    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(aitm_pl, datamodule=ali_dm)
    # tuner.lr_find(aitm_pl, datamodule=ali_dm)
    trainer.fit(
        model=model_pl,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )
    trainer.test(model=model_pl, datamodule=datamodule)
    if config["group_name"] is not None:
        wandb.finish()


def sweep_train():
    pl.seed_everything(default_config["global_seed"], workers=True)
    os.environ["WANDB__SERVICE_WAIT"] = "600"

    train = True
    test = True
    val = True
    with wandb.init(
        config=default_config,
        group=default_config["group_name"],
        # name=default_config['name'],
        save_code=True,
        settings=wandb.Settings(code_dir="."),
    ) as run:
        config = wandb.config
        sweep_value = [str(config[key]) for key in sweep_parameters.keys()]
        run.name = "+".join(sweep_value)
        wandb_logger = WandbLogger(
            log_model=default_config["log_model"],
        )

        model_pl = read_lit_model(default_config)
        litcallback = None

        model_checkpoint = (
            ModelCheckpoint(
                dirpath=config["check_point_path"],
                save_top_k=config["save_top_k"],
                mode=config["checkpoint_monitor_mode"],
                monitor=config["checkpoint_monitor_metric"],
            )
            if config["using_model_checkpoint"]
            else None
        )

        early_stop = (
            EarlyStopping(
                monitor=config["monitor_metric"],
                mode=config["mode"],
                patience=config["earlystop_epoch"],
                min_delta=config["min_delta"],
            )
            if config["using_early_stop"]
            else None
        )

        call_backs = [litcallback, early_stop, model_checkpoint]
        # filter the None
        call_backs = [call_back for call_back in call_backs if call_back is not None]

        trainer = pl.Trainer(
            max_epochs=config["total_epoch"],
            callbacks=call_backs,
            logger=wandb_logger,
            profiler=config["profiler"],
            log_every_n_steps=2,
            deterministic=True,
        )
        # torch.use_deterministic_algorithms(True, warn_only=True)

        # tuner = Tuner(trainer)
        # tuner.scale_batch_size(aitm_pl, datamodule=ali_dm)
        # tuner.lr_find(aitm_pl, datamodule=ali_dm)
        trainer.fit(
            model=model_pl,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
        )
        trainer.test(model=model_pl, datamodule=datamodule)


if __name__ == "__main__":
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    pl.seed_everything(default_config["global_seed"], workers=True)

    debug = False
    train_type = "train_single"
    sweep_parameters = dict(
        model_name={"values": ["esmm", "mmoe"]},
    )

    default_config["batch_type"] = "es"
    default_config["model_name"] = "mmoe"
    #############---------------WANDB-------------------##########################
    default_config["using_wandb"] = True
    default_config["project"] = "CVR prediction test"
    default_config["group_name"] = "deconfounder_ctcvr_no_weight_ctr"
    default_config["name"] = "mmoe"
    ############---------------WANDB-------------------###########################

    ############################################################################
    datamodule = get_DataModule(default_config, debug=debug)
    default_config["field_dims"] = datamodule.field_dims
    default_config["numerical_num"] = datamodule.numerical_num

    if train_type == "train_single":
        litmodel = read_lit_model(default_config)
        litcallback = None
        single_train(default_config, litmodel, litcallback, datamodule)
    elif train_type == "train_sweep":
        print("train_sweep")

        sweep_config = {
            "method": "grid",  # grid, random, bayes
            "name": "name",
            "metric": {
                "goal": "maximize",  # maximize,minimize
                "name": "conversion_auc_filter_test",
            },
        }

        sweep_config["parameters"] = sweep_parameters
        sweep_config["name"] = "+".join([str(key) for key in sweep_parameters.keys()])
        sweep_id = wandb.sweep(sweep_config, project=default_config["project"])
        wandb.agent(sweep_id=sweep_id, function=sweep_train, count=100)
        # sweep_train()
