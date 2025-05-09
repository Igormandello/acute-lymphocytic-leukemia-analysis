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
    polynomial_degree: Optional[int] = None,
    **kwargs,
) -> TrainGradientBoostingResponse:
    model = HistGradientBoostingClassifier(**kwargs)
    return train_classifier(model, target, data, features, grid_search_params, polynomial_degree=polynomial_degree)
