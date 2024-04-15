import hydra
import sys
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from src.models.common import BatchTransform
def test_DESCM(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)

    data_loader = hydra.utils.instantiate(cfg_train.data)
    train_data_loader = data_loader.train_dataloader()
    cfg_train.model.model._target_ = 'src.models.descm.DESCM'
    model = hydra.utils.instantiate(cfg_train.model.model)
    for batch in train_data_loader:
        click, conversion, features = BatchTransform(cfg_train.data.batch_type)(batch)
        model(features)
        break
    
def test_ESCM(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)

    data_loader = hydra.utils.instantiate(cfg_train.data)
    train_data_loader = data_loader.train_dataloader()
    cfg_train.model.model._target_ = 'src.models.descm.ESCM'
    model = hydra.utils.instantiate(cfg_train.model.model)
    for batch in train_data_loader:
        click, conversion, features = BatchTransform(cfg_train.data.batch_type)(batch)
        model(features)
        break