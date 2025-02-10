from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from colour import Color
from matplotlib.colors import LinearSegmentedColormap
from seaborn.matrix import ClusterGrid

from model.forest import train_forest

colors = ["#00FF00", "#000000", "#FF0000"]
cmap = LinearSegmentedColormap.from_list("gene", [Color(c1).rgb for c1 in colors])


def get_importances(importances: np.ndarray, feature_names: list[str], top: int = 10) -> pd.Series:
    sorted_importances = {
        name: importance
        for importance, name in sorted(zip(importances, feature_names), key=lambda x: x[0], reverse=True)
    }
    return pd.Series(data=sorted_importances)[:top]


def plot_clustermap(cluster_data: pd.DataFrame, col_colors: list[pd.Series]) -> ClusterGrid:
    return sns.clustermap(
        np.log2(cluster_data, out=np.zeros_like(cluster_data), where=cluster_data != 0),
        col_colors=col_colors,
        method="ward",
        z_score=0,
        metric="euclidean",
        cmap=cmap,
        vmin=-4,
        vmax=4,
    )


def plot_legend(g: ClusterGrid, subtypes: pd.Series, lut: dict):
    for label in subtypes.unique():
        g.ax_col_dendrogram.bar(0, 0, color=lut[label], label=label, linewidth=0)

    g.ax_col_dendrogram.legend(title="Subtype", loc="center", ncol=5, bbox_to_anchor=(0.47, 0.8))

def show_cluster(data: pd.DataFrame, metric: str, top: int = 10, features: Optional[int] = None):
  response = train_forest(metric, features=features, data=data)

  print(f"Acc: {response.acc}, F1 Score: {response.f1}")
  importances = get_importances(response.model.feature_importances_, response.feature_names, top=top)
  sns.barplot(x=importances, y=importances.index, orient='h').set(xlabel='Importance', ylabel='Gene', title=f"Importância de cada gene ({metric})")

  subtypes = data["subtype"]
  palette = sns.cubehelix_palette(subtypes.unique().size, light=.9, dark=.1, reverse=True, start=1, rot=-2)
  lut = dict(zip(map(str, subtypes.unique()), palette))
  col_colors = [subtypes.map(lut), data["prognostic"].map({ "POOR": "r", "GOOD": "g", "MODERATE": "y", "UNKNOWN": "black" })]

  cluster_data = data.drop(axis=1, labels=["prognostic", "subtype"])[importances.index].transpose()

  g = plot_clustermap(cluster_data, col_colors=col_colors)
  plot_legend(g, subtypes, lut)

  plt.show()