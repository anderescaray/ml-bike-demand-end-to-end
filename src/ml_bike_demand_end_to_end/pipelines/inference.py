"""Inference Pipeline."""
from kedro.pipeline import Pipeline, node

from .nodes import load_model, predict

def create_inference_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_model,
                inputs=["params:training.model_type", "params:model_storage"],
                outputs="model",
            ),
            node(
                func=predict,
                inputs=["model", "features"], #output of the feature_eng pipeline
                outputs="predictions",
            ),
        
        ]
    )