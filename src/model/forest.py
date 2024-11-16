from dataclasses import dataclass
from typing import Optional

from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

from model.train import TrainClassifierResponse, train_classifier


@dataclass
class TrainForestResponse(TrainClassifierResponse):
    model: RandomForestClassifier


def train_forest(
    target: str,
    data: DataFrame,
    features: Optional[int] = None,
    grid_search_params: Optional[dict] = None,
    grid_search_scoring: Optional[str] = "f1_macro",
    **kwargs,
) -> TrainForestResponse:
    model = RandomForestClassifier(n_jobs=-1, **kwargs)
    return train_classifier(model, target, data, features, grid_search_params)
