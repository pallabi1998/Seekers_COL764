import pandas as pd
from rank_bm25 import BM25Okapi
from collections import Counter

CNT = 1
df_read = pd.read_csv(f'vocab_idf_{CNT}.csv')

idf_dict = df_read.to_dict()
print("converted")
print(len(idf_dict))