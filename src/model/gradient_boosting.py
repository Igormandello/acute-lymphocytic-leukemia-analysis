from dataclasses import dataclass
from typing import Optional

from pandas import DataFrame
from sklearn.ensemble import HistGradientBoostingClassifier

from model.train import TrainClassifierResponse, train_classifier


@dataclass
class TrainGradientBoostingResponse(TrainClassifierResponse):
    model: HistGradientBoostingClassifier


def train_gradient_boosting(
    target: str,
    data: DataFrame,
    features: Optional[int] = None,
    grid_search_params: Optional[dict] = None,
    **kwargs,
) -> TrainGradientBoostingResponse:
    model = HistGradientBoostingClassifier(random_state=774, **kwargs)
    return train_classifier(model, target, data, features, grid_search_params)
