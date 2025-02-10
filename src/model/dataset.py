from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
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
    scaler: StandardScaler

@dataclass
class MakeGroupedDatasetResponse:
    X_train: np.ndarray
    X_train_male: np.ndarray
    X_train_female: np.ndarray
    y_train: Series
    y_train_male: Series
    y_train_female: Series
    X_test: np.ndarray
    X_test_male: np.ndarray
    X_test_female: np.ndarray
    y_test: Series
    y_test_male: Series
    y_test_female: Series
    feature_names: list[str]
    scaler: StandardScaler


def make_dataset(data: DataFrame, features: Optional[int], target: str) -> MakeDatasetResponse:
    X = data.drop(axis=1, labels=["sample_id", "prognostic", "subtype", "sex", "race", "is_white", "subtype_target"], errors="ignore")
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    feature_names = X.columns

    if features != None:
        kbest = SelectKBest(chi2, k=features)
        X_train = kbest.fit_transform(X_train, y_train)
        X_test = kbest.transform(X_test)
        feature_names = np.array(X.columns)[kbest.get_support(indices=False)]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return MakeDatasetResponse(X_train, y_train, X_test, y_test, feature_names, scaler)

def make_grouped_dataset(data: DataFrame, target: str) -> MakeGroupedDatasetResponse:
    X = data.drop(axis=1, labels=["sample_id", "prognostic", "subtype", "race", "is_white", "subtype_target"], errors="ignore")
    X_male = X[X["sex"] == "Male"]
    X_female = X[X["sex"] == "Female"]

    y_male = data[data["sex"] == "Male"][target]
    y_female = data[data["sex"] == "Female"][target]

    X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(X_male.drop(columns=["sex"]), y_male, stratify=y_male)
    X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(X_female.drop(columns=["sex"]), y_female, stratify=y_female)
    feature_names = list(set(X.columns) - set(["sex"]))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(np.vstack([X_train_male, X_train_female]))
    X_train_male = scaler.transform(X_train_male)
    X_train_female = scaler.transform(X_train_female)

    X_test = scaler.transform(np.vstack([X_test_male, X_test_female]))
    X_test_male = scaler.transform(X_test_male)
    X_test_female = scaler.transform(X_test_female)

    y_train = pd.concat([y_train_male, y_train_female])
    y_test = pd.concat([y_test_male, y_test_female])

    return MakeGroupedDatasetResponse(X_train, X_train_male, X_train_female, y_train, y_train_male, y_train_female, X_test, X_test_male, X_test_female, y_test, y_test_male, y_test_female, feature_names, scaler)
