from dataclasses import dataclass
from typing import Optional

from lightgbm import LGBMClassifier
from pandas import DataFrame

from model.train import TrainClassifierResponse, train_classifier


@dataclass
class TrainLGBMResponse(TrainClassifierResponse):
    model: LGBMClassifier


def train_lgbm(
    target: str,
    data: DataFrame,
    features: Optional[int] = None,
    grid_search_params: Optional[dict] = None,
    polynomial_degree: Optional[int] = None,
    **kwargs,
) -> TrainLGBMResponse:
    model = LGBMClassifier(**kwargs, verbosity=-1)
    return train_classifier(model, target, data, features, grid_search_params, polynomial_degree=polynomial_degree)
