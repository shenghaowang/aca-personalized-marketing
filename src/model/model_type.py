from enum import Enum
from typing import Union

from causalml.inference.tree import UpliftRandomForestClassifier
from lightgbm import LGBMClassifier

from model.neuralnet import NeuralNetClassifier


class ModelType(str, Enum):
    ClassTransformLGBM = "lgbm"
    ClassTransformMLP = "mlp"
    UpliftRF = "uplift_rf"


def get_model_kwargs(model_type: str, feature_cols: list) -> dict:
    """Get the required kwargs for initializing a model based on its type."""
    match model_type:
        case ModelType.ClassTransformLGBM.value | ModelType.ClassTransformMLP.value:
            return {"input_dim": len(feature_cols)}
        case ModelType.UpliftRF.value:
            return {"control_name": "No"}
        case _:
            return {}


def init_model(
    model_name: str, **kwargs
) -> Union[LGBMClassifier, NeuralNetClassifier, UpliftRandomForestClassifier]:
    match model_name:
        case ModelType.ClassTransformLGBM.value:
            model = LGBMClassifier(
                random_state=42, scale_pos_weight=5, learning_rate=0.01, n_estimators=50
            )
        case ModelType.ClassTransformMLP.value:
            model = NeuralNetClassifier(input_dim=kwargs.get("input_dim"))
        case ModelType.UpliftRF.value:
            model = UpliftRandomForestClassifier(
                n_estimators=10,
                control_name=kwargs.get("control_name"),
                random_state=42,
            )
        case _:
            raise ValueError(f"Unsupported model type: {model_name}")

    return model
