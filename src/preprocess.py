import pandas as pd

genes_csv = pd.read_csv("data/genes.csv", delimiter=",", decimal=",", header=None, low_memory=False)

# Removing Mean Average Deviation values
df = genes_csv.drop(axis=1, labels=[genes_csv.shape[1] - 1])
df = df.transpose()
df.iloc[0, 0] = "subtype"
df.columns = df.iloc[0, :]

prognostic_map = {
    "ETV6RUNX1": "GOOD",
    "HYPER": "UNKNOWN",
    "PHlike": "POOR",
    "DUX4IGH": "GOOD",
    "KMT2A": "POOR",
    "PAX5": "MODERATE",
    "BCRABL1": "POOR",
    "HYPO": "UNKNOWN",
    "TCF3PBX1": "MODERATE",
    "iAMP21": "POOR",
}

df["prognostic"] = df["subtype"].map(prognostic_map)

df = df.drop(index=0)
df.to_csv("data/genes_transformed.csv", index=False)
