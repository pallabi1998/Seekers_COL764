import pandas as pd
from rank_bm25 import BM25Okapi
from collections import Counter

CNT = 2
import pickle

with open('bm25_model_105947593.pkl', 'rb') as file:
    bm25 = pickle.load(file)

print("Loaded model")
#vocab = set(word for doc_freqs_dict in bm25.doc_freqs for word in doc_freqs_dict.keys())
idf =  bm25.idf
print(len(idf))
# Create a Counter for word frequencies based on document frequencies
#word_frequencies = Counter({word: sum(doc_freqs_dict.get(word, 0) for doc_freqs_dict in bm25.doc_freqs) for word in vocab})


df = pd.DataFrame(idf, index=[0])
print("converted")
# Write the DataFrame to a CSV file
df.to_csv(f'vocab_idf_{CNT}.csv', index=False)