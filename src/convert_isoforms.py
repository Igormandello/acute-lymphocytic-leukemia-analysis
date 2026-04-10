import pandas as pd

df = pd.read_csv("data/transcripts_isoforms.tsv", sep="\t").set_index("transcript_id", drop=True).T
df.index.name = "sample_id"
df.to_csv("data/transcripts_isoforms.csv")
