from numpy import random
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm

from model.dataset import make_grouped_dataset, make_grouped_dataset_biased


def report_results(df: pd.DataFrame, parameters: dict, seed_list: list[int], is_biased: bool):
    male_f1s = []
    female_f1s = []
    f1s = []
    for seed in tqdm(seed_list):
        random.mtrand._rand.seed(seed)

        model = HistGradientBoostingClassifier(**parameters)
        response = make_grouped_dataset_biased(df, "subtype") if is_biased else make_grouped_dataset(df, "subtype")
        model.fit(response.X_train, response.y_train)

        f1s.append(f1_score(response.y_test, model.predict(response.X_test), average="weighted"))
        male_f1s.append(f1_score(response.y_test_male, model.predict(response.X_test_male), average="weighted"))
        female_f1s.append(f1_score(response.y_test_female, model.predict(response.X_test_female), average="weighted"))

    print(f"F1: {np.median(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Male F1: {np.median(male_f1s):.4f} ± {np.std(male_f1s):.4f}")
    print(f"Female F1: {np.median(female_f1s):.4f} ± {np.std(female_f1s):.4f}")