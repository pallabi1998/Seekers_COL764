import pandas as pd
from rank_bm25 import BM25Okapi
from collections import Counter
import string
import re
import numpy as np

CNT = 2
import pickle


def remove_punctuation(text):
    text = text.replace('|' , " or ")
    return re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)

qfname = "./query_file.txt"
with open(qfname, 'r') as qfile:
    queries_t = qfile.readlines()

queries = []
doc_scores = []
for q in queries_t:
    queries.append(w for w in remove_punctuation(q).split(" ") if not w.isspace())
    doc_scores.append([])


with open("pid_last.pkl", 'rb') as pfile:
    pids= pickle.load(pfile)

print(len(pids), pids[0], pids[-1])

fnames = [ 'bm25_model_229868210.pkl', 'bm25_model_506154247.pkl', 'bm25_model_809630815.pkl', 'bm25_model_1127853430.pkl', 'bm25_model_last.pkl'] #'bm25_model_105947593.pkl',
#bm25_models = []


for fid, f in enumerate(fnames):
    with open(f, 'rb') as file:
        bm25_model= pickle.load(file)
        print(fid)
        #with open(f"./outputs/bm25_{fid}", 'w') as outfile:
        for qid, tokenized_query in enumerate(queries):
                doc_score = bm25_model.get_scores(tokenized_query)
                doc_scores[qid].append(doc_score)
                # a = np.argsort(doc_score)
                # for i in range(200):
                #     outfile.write(f"{qid} Q0 {pids[a[i]]} {i} {doc_score} 1")
        del bm25_model


