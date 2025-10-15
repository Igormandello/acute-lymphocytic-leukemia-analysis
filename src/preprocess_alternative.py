
import sys
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import json
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

min_tpm_filter = int(sys.argv[1])

save_prefix = f"preprocessed/min_tpm_{min_tpm_filter}/"
os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
print(f"Preprocessing data, filtering out genes with max average TPM (by subtype) < {min_tpm_filter}. Saving to prefix '{save_prefix}'.")

combined_df = pd.read_csv("data/genes.csv").fillna(0)
extra_data_df = pd.read_csv("data/extra_data.tsv", delimiter="\t", keep_default_na=False)
final_df = pd.merge(combined_df, extra_data_df, on="sample_id", how="left")

target = "subtype"
sex = final_df["sex"]
X = final_df.drop(axis=1, labels=extra_data_df.columns, errors="ignore")
y = final_df[target]
X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(X, y, sex, stratify=y, test_size=0.3, random_state=1337)

train_df = pd.DataFrame(X_train, columns=X.columns, index=y_train.index)
train_df[target] = y

max_average_by_gene = {}
data_by_gene_by_subtype = defaultdict(lambda: defaultdict(dict))
for gene in set(combined_df.columns) - set(["sample_id"]):
    genes_average = train_df[["subtype", gene]].groupby("subtype").agg(["mean", "std"]).reset_index()
    
    average_by_subtype = {}
    for _, row in genes_average.iterrows():
        subtype = row["subtype"].iloc[0]
        average_by_subtype[subtype] = row[(gene, "mean")]
        data_by_gene_by_subtype[gene][subtype]["mean"] = row[(gene, "mean")]
        data_by_gene_by_subtype[gene][subtype]["stddev"] = row[(gene, "std")]

    max_average_by_gene[gene] = max(average_by_subtype.values())

dropped_genes = [x[0] for x in max_average_by_gene.items() if x[1] < min_tpm_filter]
print(f"Dropped {len(dropped_genes)} genes")
open(f"{save_prefix}/data_by_gene_by_subtype.json", "w").write(json.dumps(data_by_gene_by_subtype))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_df = pd.DataFrame(X_train, columns=X.columns, index=y_train.index)
train_df[target] = y_train
train_df["sex"] = sex_train
train_df = train_df.drop(labels=dropped_genes, axis=1)

test_df = pd.DataFrame(X_test, columns=X.columns, index=y_test.index)
test_df[target] = y_test
test_df["sex"] = sex_test
test_df = test_df.drop(labels=dropped_genes, axis=1)

train_df.to_csv(f"{save_prefix}/train.csv", index=False)
test_df.to_csv(f"{save_prefix}/test.csv", index=False)

print("Dataset statistics:")
print("-------------------------")
print("Train:")
print(f"Size: {len(train_df)}")
print(f"Distribution: {train_df[target].value_counts()}")
print("-------------------------")
print("Test:")
print(f"Size: {len(test_df)}")
print(f"Distribution: {test_df[target].value_counts()}")
