from dataclasses import dataclass
from typing import Optional
import numpy as np
from pandas import DataFrame, Series
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split


@dataclass
class MakeDatasetResponse:
    X_train: np.ndarray
    y_train: Series
    X_test: np.ndarray
    y_test: Series
    feature_names: list[str]

def make_dataset(data: DataFrame, features: Optional[int], target: str) -> MakeDatasetResponse:
    X = data.drop(axis=1, labels=["prognostic", "subtype", "subtype_target"], errors="ignore")
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    feature_names = X.columns

    if features != None:
        kbest = SelectKBest(chi2, k=features)
        X_train = kbest.fit_transform(X_train, y_train)
        X_test = kbest.transform(X_test)
        feature_names = np.array(X.columns)[kbest.get_support(indices=False)]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return MakeDatasetResponse(X_train, y_train, X_test, y_test, feature_names)
