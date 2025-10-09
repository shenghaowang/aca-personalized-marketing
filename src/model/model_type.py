from enum import Enum
from typing import Union

from lightgbm import LGBMClassifier

from model.neuralnet import NeuralNetClassifier


class ModelType(str, Enum):
    LGBM = "lgbm"
    MLP = "mlp"


def init_model(
    model_name: str, input_dim: int = None
) -> Union[LGBMClassifier, NeuralNetClassifier]:
    match model_name:
        case ModelType.LGBM.value:
            model = LGBMClassifier(
                random_state=42, scale_pos_weight=5, learning_rate=0.01, n_estimators=50
            )
        case ModelType.MLP.value:
            model = NeuralNetClassifier(input_dim=input_dim)
        case _:
            raise ValueError(f"Unsupported model type: {model_name}")

    return model
