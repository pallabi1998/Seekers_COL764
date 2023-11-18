
import nltk
from nltk.corpus import stopwords
import math

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


inp_file = "./trec_formatted_with_p_tags.txt"

res_file = './finalqnswer_freq.txt'

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


#print(titles, texts)


with open(res_file, 'r') as rfile:
    results = rfile.readlines()








total_bleu = 0
total_jacc = 0
total_meteor = 0
total_cos = 0

def get_cosine(str1, str2):
    tokens1 = tokenizer(str1, return_tensors='pt')
    tokens2 = tokenizer(str2, return_tensors='pt')

    # Get BERT embeddings
    with torch.no_grad():
        outputs1 = model(**tokens1)
        outputs2 = model(**tokens2)

    # Extract the embeddings from the output
    embeddings1 = outputs1.last_hidden_state.mean(dim=1).numpy()
    embeddings2 = outputs2.last_hidden_state.mean(dim=1).numpy()

    # Calculate cosine similarity
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return similarity





def jaccard_similarity(str1, str2):
    # Tokenize the strings into sets of words
    set1 = set(str1.split())
    set2 = set(str2.split())

    # Calculate the Jaccard similarity coefficient
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2)) 


    if union == 0:
        return 0  # Avoid division by zero if both sets are empty
    else:
        return intersection / union

print("===============================================================")
cnt = 0
for text, title in zip(texts, titles):
    
    
    if(cnt<25):
        cnt+=1
        continue
    
    candidate = results[cnt]
    #candidate = results[cnt-25]
    
    total_jacc += jaccard_similarity(title, candidate)
    smooth_func = SmoothingFunction().method3
    bleu_score = sentence_bleu(title.split(), candidate.split(), smoothing_function=smooth_func)
    total_cos += get_cosine(candidate, title)
    total_bleu += bleu_score
    cnt += 1
    print(title, candidate, bleu_score, jaccard_similarity(title, candidate))

print(total_bleu/25, total_jacc/25, total_cos/25)

