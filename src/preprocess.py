import pandas as pd

male_df = pd.read_csv("data/male.csv")
female_df = pd.read_csv("data/female.csv")
combined_df = pd.concat([male_df, female_df], axis=0).fillna(0)

extra_data_df = pd.read_csv("data/extra_data.tsv", delimiter="\t", keep_default_na=False)
final_df = pd.merge(combined_df, extra_data_df, on="sample_id", how="left")

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

final_df["prognostic"] = final_df["subtype"].map(prognostic_map)
final_df.to_csv("data/genes_extra_data.csv", index=False)
