from numpy import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score
from tqdm import tqdm
from joblib import parallel_config

from model.dataset import MakeGroupedDatasetResponse, make_grouped_dataset

from dataclasses import dataclass, field

@dataclass
class MetricsReport:
    name: str
    overall: list = field(default_factory=list)
    male: list = field(default_factory=list)
    female: list = field(default_factory=list)

    def add(self, metric, response: MakeGroupedDatasetResponse, y_pred, y_pred_male, y_pred_female):
        self.overall.append(metric(response.X_test, response.y_test, y_pred))
        self.male.append(metric(response.X_test_male, response.y_test_male, y_pred_male))
        self.female.append(metric(response.X_test_female, response.y_test_female, y_pred_female))

@dataclass
class MetricsResults:
    report: pd.DataFrame
    f1: MetricsReport
    recall: MetricsReport
    accuracy: MetricsReport

def get_avg_std(data: list):
    return f"{np.median(data):.4f} ± {np.std(data):.4f}"

def create_report(*metrics: MetricsReport):
    columns = ["Metric", "Overall", "Male", "Female"]
    data = [[metric.name, get_avg_std(metric.overall), get_avg_std(metric.male), get_avg_std(metric.female)] for metric in metrics]
    return pd.DataFrame(data, columns=columns)


def report_results(df: pd.DataFrame, model_factory, parameters: dict, seed_list: list[int], polynomial_degree: int | None = None):
    f1 = MetricsReport("F1")
    roc_auc = MetricsReport("ROC AUC")
    recall = MetricsReport("Recall")
    accuracy = MetricsReport("Accuracy")

    for seed in tqdm(seed_list):
        random.mtrand._rand.seed(seed)

        model = model_factory(**parameters)
        with parallel_config(n_jobs=-1):
            response = make_grouped_dataset(df, "subtype", polynomial_degree)
            model.fit(response.X_train, response.y_train)

        y_pred = model.predict(response.X_test)
        y_pred_male = model.predict(response.X_test_male)
        y_pred_female = model.predict(response.X_test_female)

        f1.add(lambda _, x, y: f1_score(x, y, average="weighted"), response, y_pred, y_pred_male, y_pred_female)
        recall.add(lambda _, x, y: recall_score(x, y, average="macro"), response, y_pred, y_pred_male, y_pred_female)
        roc_auc.add(lambda x, y_true, _: roc_auc_score(y_true, model.predict_proba(x), multi_class='ovr'), response, y_pred, y_pred_male, y_pred_female)
        accuracy.add(lambda _, x, y: accuracy_score(x, y), response, y_pred, y_pred_male, y_pred_female)

    return MetricsResults(
        report=create_report(f1, recall, roc_auc, accuracy),
        f1=f1,
        recall=recall,
        accuracy=accuracy
    )
