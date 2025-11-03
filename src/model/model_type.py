from enum import Enum
from typing import Union

from causalml.inference.tree import UpliftRandomForestClassifier
from lightgbm import LGBMClassifier
from omegaconf import DictConfig, OmegaConf

from model.neuralnet import NeuralNetClassifier


class ModelType(str, Enum):
    ClassTransformLGBM = "lgbm"
    ClassTransformMLP = "mlp"
    UpliftRF = "uplift_rf"


def init_model(
    model_cfg: DictConfig, input_dim: int = None, control_name: str = None
) -> Union[LGBMClassifier, NeuralNetClassifier, UpliftRandomForestClassifier]:
    """Initialize a model based on its configuration.

    Args:
        model_cfg: Model configuration from Hydra
        input_dim: Number of input features (required for neural networks)
        control_name: Name of control group (required for UpliftRF)
    """
    # Convert config to dict and remove the name field
    model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
    model_name = model_cfg.pop("name")

    match model_name:
        case ModelType.ClassTransformLGBM.value:
            model = LGBMClassifier(**model_cfg)
        case ModelType.ClassTransformMLP.value:
            model_cfg["input_dim"] = input_dim
            model = NeuralNetClassifier(**model_cfg)
        case ModelType.UpliftRF.value:
            model_cfg["control_name"] = control_name
            model = UpliftRandomForestClassifier(**model_cfg)
        case _:
            raise ValueError(f"Unsupported model type: {model_name}")

    return model
