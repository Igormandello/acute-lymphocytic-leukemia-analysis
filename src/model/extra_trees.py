from dataclasses import dataclass
from typing import Optional

from pandas import DataFrame
from sklearn.ensemble import ExtraTreesClassifier

from model.train import TrainClassifierResponse, train_classifier


@dataclass
class TrainExtraTreesResponse(TrainClassifierResponse):
    model: ExtraTreesClassifier


def train_extra_trees(
    target: str,
    data: DataFrame,
    features: Optional[int] = None,
    grid_search_params: Optional[dict] = None,
    **kwargs,
) -> TrainExtraTreesResponse:
    model = ExtraTreesClassifier(n_jobs=-1, **kwargs)
    return train_classifier(model, target, data, features, grid_search_params)
