from dataclasses import dataclass
from typing import Optional, TypeVar

from numpy import ndarray
from pandas import DataFrame
from sklearn.base import ClassifierMixin
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from model.dataset import make_dataset

T = TypeVar("T", bound=ClassifierMixin)

@dataclass
class TrainClassifierResponse:
    model: T
    acc: float
    f1: float
    multiclass_f1: ndarray
    train_report: str
    test_report: str
    feature_names: list[str]


def train_classifier(
    classifier: T,
    target: str,
    data: DataFrame,
    features: Optional[int] = None,
    grid_search_params: Optional[dict] = None
) -> TrainClassifierResponse:
    response = make_dataset(data, features, target)

    if grid_search_params:
        classifier = GridSearchCV(
            classifier,
            n_jobs=-1,
            cv=StratifiedKFold(n_splits=5, shuffle=False),
            scoring="f1_macro",
            param_grid=grid_search_params,
        )

    classifier.fit(response.X_train, response.y_train)

    if grid_search_params:
        classifier = classifier.best_estimator_
        print(f"Best params: {classifier.get_params()}")

    prediction = classifier.predict(response.X_test)
    f1 = f1_score(response.y_test, prediction, average="weighted")
    multiclass_f1 = f1_score(response.y_test, prediction, average=None)
    acc = balanced_accuracy_score(response.y_test, prediction)

    train_report = classification_report(response.y_train, classifier.predict(response.X_train), output_dict=False)
    report = classification_report(response.y_test, prediction, output_dict=False)

    return TrainClassifierResponse(
        model=classifier,
        acc=acc,
        f1=f1,
        multiclass_f1=multiclass_f1,
        train_report=train_report,
        test_report=report,
        feature_names=response.feature_names,
    )
