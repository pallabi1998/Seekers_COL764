import pandas as pd
from rank_bm25 import BM25Okapi
from collections import Counter
import string
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

CNT = 2
import pickle





import nltk
from nltk.corpus import stopwords
import math


from nltk.translate.bleu_score import sentence_bleu


inp_file = "./RCD2020FIRETASK/trec_formatted_with_p_tags.txt"



import xml
import xml.etree.ElementTree as ET

tree = ET.parse(inp_file)
root = tree.getroot()



    
titles = []
texts = []

for top_element in root.findall("top"):
    #print(top_element)
    title_element = top_element.find("title")
    desc_element = top_element.find("desc")
    text_element = ""
    for p_elem in desc_element.findall("p"):
        text_element += p_elem.text + " "
    titles.append(title_element.text)
    texts.append(text_element.strip())

def remove_punctuation(text):
    text = text.replace('|' , " or ")
    return re.sub(r'[{}]'.format(re.escape(string.punctuation)), ' ', text)





with open("pid_last.pkl", 'rb') as pfile:
    pids= pickle.load(pfile)

print(len(pids), pids[0], pids[-1])

fnames = [ 'bm25_model_229868210.pkl', 'bm25_model_506154247.pkl', 'bm25_model_809630815.pkl', 'bm25_model_1127853430.pkl', 'bm25_model_last.pkl'] #'bm25_model_105947593.pkl',
#bm25_models = []



final_scores = []
text_words = []
for text1, title in zip(texts, titles):
    scores = []
    text = remove_punctuation(text1).split()
    text2 = [lemmatizer.lemmatize(t) for t in text]
    text_words.append(text)
    for t in text:
        scores.append(0.0)
    final_scores.append(scores)


for fid, f in enumerate(fnames):
    with open(f, 'rb') as file:
        bm25_model= pickle.load(file)
        print(fid)
        idf =  bm25_model.idf
        for qid, text1 in enumerate(texts):
            text = remove_punctuation(text1).split()
            for tid, t in enumerate(text):       
                if t in idf and len(t)>4:
                    final_scores[qid][tid] += idf[t]

                elif len(t)>4:
                    #final_scores[qid][tid] += 0.1 # 0
                    print(t)

                # a = np.argsort(doc_score)
                # for i in range(200):
                #     outfile.write(f"{qid} Q0 {pids[a[i]]} {i} {doc_score} 1")
        del bm25_model


with open(f"./outputs/lemfinalqnswer_freq.txt", 'w') as outfile:
    
    for qid in range(len(texts)):
        
        a = np.argsort(final_scores[qid])
        #print(a, text_words[qid][a[0]])
        
        outfile.write(text_words[qid][a[-1]]+'\n')

