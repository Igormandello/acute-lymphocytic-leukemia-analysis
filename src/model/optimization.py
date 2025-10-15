from enum import Enum
from typing import Optional
from joblib import parallel_config
import optuna
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, balanced_accuracy_score, make_scorer, precision_recall_curve, recall_score
from sklearn.model_selection import cross_val_score

from model.dataset import get_dataset

from dataclasses import dataclass, field

weighted_f1 = make_scorer(lambda x, y: f1_score(x, y, average="weighted"), greater_is_better=True)
micro_recall = make_scorer(lambda x, y: recall_score(x, y, average="micro"), greater_is_better=True)

@dataclass
class EvalResults:
    f1_weighted: float
    f1_macro: float
    recall_macro: float
    balanced_accuracy: float
    accuracy: float
    report: str

class SuggestionValueType(Enum):
  INT = 0
  FLOAT = 1
  CATEGORY = 2

@dataclass
class ObjectiveSuggestion:
  value_type: SuggestionValueType
  param: str
  param_range: tuple[float, float] = (0, 0)
  param_options: list[str] = field(default_factory=list)
  transform_param: Optional[callable] = None
  is_log: bool = False

def eval_model(model, dataset: tuple[pd.DataFrame, pd.Series] = None, use_threshold: bool = False, report: bool = False):
  with parallel_config(n_jobs=-1):
    X, y = get_dataset(category="min_tpm_5", dataset="test", target="subtype") if dataset == None else dataset
    if use_threshold and len(y.unique()) > 2:
      raise Exception("Threhsold must only be used in binary classifications")

    if use_threshold:
      probabilities = model.predict_proba(X)[:, 1]
      _, _, thresholds = precision_recall_curve(y, probabilities)
      accuracies = [balanced_accuracy_score(y, probabilities >= threshold) for threshold in thresholds]
      max_acc = max(accuracies)
      threshold = next((i for i, value in zip(thresholds, accuracies) if value == max_acc))
      print(f"Using threshold {threshold}, with max acc {max_acc}")
      prediction = probabilities >= threshold
    else:
      prediction = model.predict(X)

    results = EvalResults(
      f1_weighted=f1_score(y, prediction, average="weighted"),
      f1_macro=f1_score(y, prediction, average="macro"),
      recall_macro=recall_score(y, prediction, average="macro"),
      balanced_accuracy=balanced_accuracy_score(y, prediction),
      accuracy=accuracy_score(y, prediction),
      report=classification_report(y, prediction)
    )

  if report:
    values = dict(results.__dict__)
    values.pop("report")
    print(pd.DataFrame(values, index=[0]))

  return results

def create_study(name: str = None, model_factory: callable = None, suggestions: list[ObjectiveSuggestion] = [], scoring: callable = weighted_f1, trials: int = 5, report_test_results: bool = True, custom_dataset: tuple[pd.DataFrame, pd.Series] = None, n_jobs: int = -1, verbosity = optuna.logging.INFO):
  storage = None
  if name != None:
    storage = optuna.storages.JournalStorage(
      optuna.storages.journal.JournalFileBackend(f"../studies/{name}.log")
    )

  X, y = custom_dataset if custom_dataset != None else get_dataset(category="min_tpm_5", dataset="train", target="subtype")

  study = optuna.create_study(study_name=name, storage=storage, direction="maximize", load_if_exists=True)
  current_trials = len(study.trials)
  if current_trials < trials:
    objective_fn = create_objective(model_factory, scoring, suggestions, X=X, y=y)
    optuna.logging.set_verbosity(verbosity)
    study.optimize(objective_fn, n_trials=trials - current_trials, n_jobs=n_jobs)

  model = model_factory(**study.best_params)
  model.fit(X, y)

  if report_test_results:
    eval_model(model, report=True)

  return study, model

def create_objective(model_factory, scoring: callable, suggestions: list[ObjectiveSuggestion], X: pd.DataFrame, y: pd.Series):
  def objective(trial: optuna.Trial):
    params = {}
    for suggestion in suggestions:
      name = suggestion.param
      value = None
      if suggestion.value_type is SuggestionValueType.CATEGORY:
        value = trial.suggest_categorical(name, choices=suggestion.param_options)
      else:
        suggestion_fn = trial.suggest_int if suggestion.value_type is SuggestionValueType.INT else trial.suggest_float if suggestion.value_type is SuggestionValueType.FLOAT else trial.suggest_categorical
        value = suggestion_fn(name, suggestion.param_range[0], suggestion.param_range[1], log=suggestion.is_log)

      params[name] = value if suggestion.transform_param == None else suggestion.transform_param(value)

    classifier = model_factory(**params)
    with parallel_config(n_jobs=-1):
      return cross_val_score(classifier, X, y, scoring=scoring, cv=5).mean()
  
  return objective

def load_model(name: str, model_factory: callable, dataset: tuple[pd.DataFrame, pd.Series]):
  storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend(f"../studies/{name}.log")
  )

  study = optuna.load_study(study_name=name, storage=storage)
  model = model_factory(**study.best_params)

  X, y = dataset
  model.fit(X, y)

  return model