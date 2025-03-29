import pandas as pd

df = pd.read_excel("data/genes.xlsx").set_index("gene_id", drop=True).T
df.index.name = "sample_id"
df.to_csv("data/genes.csv")
