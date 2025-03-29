import sys
import os
from collections import defaultdict
import pandas as pd
import json

min_tpm_filter = int(sys.argv[1])

save_prefix = f"preprocessed/min_tpm_{min_tpm_filter}/"
os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
print(f"Preprocessing data, filtering out genes with max average TPM (by subtype) < {min_tpm_filter}. Saving to prefix '{save_prefix}'.")

combined_df = pd.read_csv("data/genes.csv").fillna(0)
extra_data_df = pd.read_csv("data/extra_data.tsv", delimiter="\t", keep_default_na=False)
final_df = pd.merge(combined_df, extra_data_df, on="sample_id", how="left")

max_average_by_gene = {}
data_by_gene_by_subtype = defaultdict(lambda: defaultdict(dict))
for gene in set(combined_df.columns) - set(["sample_id"]):
    genes_average = final_df[["subtype", gene]].groupby("subtype").agg(["mean", "std"]).reset_index()
    
    average_by_subtype = {}
    for _, row in genes_average.iterrows():
        subtype = row["subtype"].iloc[0]
        average_by_subtype[subtype] = row[(gene, "mean")]
        data_by_gene_by_subtype[gene][subtype]["mean"] = row[(gene, "mean")]
        data_by_gene_by_subtype[gene][subtype]["stddev"] = row[(gene, "std")]

    max_average_by_gene[gene] = max(average_by_subtype.values())

dropped_genes = [x[0] for x in max_average_by_gene.items() if x[1] < min_tpm_filter]
print(f"Dropped {len(dropped_genes)} genes")
final_df = final_df.drop(columns=dropped_genes, axis=1)

open(f"{save_prefix}/data_by_gene_by_subtype.json", "w").write(json.dumps(data_by_gene_by_subtype))

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
final_df.to_csv(f"{save_prefix}/genes.csv", index=False)
