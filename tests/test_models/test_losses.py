import hydra
import torch
import pytest
import sys
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from src.models.common import BatchTransform, IPW_Loss
    
def test_IPW_Loss(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)

    data_loader = hydra.utils.instantiate(cfg_train.data)
    train_data_loader = data_loader.train_dataloader()
    cfg_train.model.model._target_ = 'src.models.descm.DESCM'
    model = hydra.utils.instantiate(cfg_train.model.model)
    loss = IPW_Loss(1,1,0.1)
    for batch in train_data_loader:
        click, conversion, features = BatchTransform(cfg_train.data.batch_type)(batch)
        click_pred, conversion_pred, click_conversion_pred, imp_pred, task_feature = model(features)
        losses = loss(click_pred, conversion_pred, click_conversion_pred, click, conversion)
        break


# @pytest.mark.parametrize("p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion, expected", [
#     # ID: HappyPath-1
#     (torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([1.0]), torch.tensor([1.0]), 1, 1, 1, pytest.approx(1.3863, 0.01)),
#     # ID: HappyPath-2
#     (torch.tensor([0.9]), torch.tensor([0.1]), torch.tensor([0.8]), torch.tensor([1.0]), torch.tensor([0.0]), 0.5, 0.3, 0.2, pytest.approx(0.3154, 0.01)),
#     # ID: EdgeCase-LowPCTR
#     (torch.tensor([1e-8]), torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([1.0]), torch.tensor([1.0]), 1, 1, 1, pytest.approx(16.1181, 0.01)),
#     # ID: EdgeCase-HighPCTR
#     (torch.tensor([1.0 - 1e-8]), torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([1.0]), torch.tensor([1.0]), 1, 1, 1, pytest.approx(1.3863, 0.01)),
#     # ID: ErrorCase-NegativeLossProportion
#     pytest.param(torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([1.0]), torch.tensor([1.0]), -1, 1, 1, 0, marks=pytest.mark.xfail),
# ])
# def test_ipw_loss(p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr, ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion, expected):
#     # Arrange
#     loss_calculator = IPW_Loss(ctr_loss_proportion, cvr_loss_proportion, ctcvr_loss_proportion)

#     # Act
#     loss = loss_calculator(p_ctr, p_cvr, p_ctcvr, y_ctr, y_cvr)

#     # Assert
#     assert loss == expected, f"Expected loss {expected}, but got {loss}"
