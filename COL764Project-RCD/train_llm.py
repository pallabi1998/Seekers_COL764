import random
import math
import time
import copy
import os
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.optim as optim
import transformers
#from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import nltk
from nltk.corpus import stopwords
import math


from nltk.translate.bleu_score import sentence_bleu


inp_file = "./train_data.xml"



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
   
dataset = []   
for i in range(len(texts)):
    dataset.append(f"Given Dialogue - {texts[i]} Difficult words are - {titles[i]}")


n_epochs = 1

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4)

DEBUG = False

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


class RCDDataset(Dataset):
    
    def __init__(self, ds_path):
        self.rcd = ds_path
        
    def __len__(self):
        return len(self.rcd)
        
    def __getitem__(self, idx):
        # returning question
         
        return self.rcd[idx]
    
def train_model(model, dataloader):
    
    model.train()
    loss = 0
    n = len(dataloader.dataset)
    
    for batch in dataloader:
        #questions, answers = batch
        tok_labels = tokenizer(list(batch), padding=True, truncation=True, return_tensors='pt')
        tok_labels = {k : v.to(device) for k,v in tok_labels.items()}        
        
        optimizer.zero_grad()
        
        outputs = model(**tok_labels, labels=tok_labels['input_ids'])
        
        batch_loss = outputs.loss
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.detach()*len(batch)
        
    return loss.cpu().item()/n

def predict_model(model, dataloader):
    loss, accuracy = 0, 0
    n = len(dataloader.dataset)
    genres = torch.zeros(n, dtype=torch.long).to(device)

    with torch.no_grad():
        for batch in dataloader:
            questions, answers = batch
            tok_titles = tokenizer(list(questions), padding=True, truncation=True, return_tensors='pt')
            tok_titles = {k : v.to(device) for k,v in tok_titles.items()}        
            outputs = model(**tok_titles).logits
            _, preds = torch.max(outputs, 1)
            print(preds.cpu())
    #return pd.DataFrame({'Genre': genres.cpu()}).rename_axis('Id')

def run(dataloader, savepath='.'):

    train_stats = []

    print(f"Training Model")
    for i in range(1, n_epochs+1):
        print(f"\nEpoch {i}:")

        loss = train_model(model, dataloader)
        print(f"    Loss = {loss}")

        train_stats.append(loss)

    torch.save(model, f'{savepath}/gpt2.pt')

    return train_stats




def topk(probs, n=5):
    # The scores are initially softmaxed to convert to probabilities
    probs = torch.softmax(probs, dim= -1)
    
    # PyTorch has its own topk method, which we use here
    tokensProb, topIx = torch.topk(probs, k=n)
    
    # The new selection pool (9 choices) is normalized
    tokensProb = tokensProb / torch.sum(tokensProb)

    # Send to CPU for numpy handling
    tokensProb = tokensProb.cpu().detach().numpy()

    # Make a random choice from the pool based on the new prob distribution
    choice = np.random.choice(n, 1, p = tokensProb)
    tokenId = topIx[choice][0]

    return int(tokenId)

def model_infer(model, tokenizer, init_token, max_length=20, n=3):
    # Preprocess the init token (task designator)
    init_id = tokenizer.encode(init_token)
    result = init_id
    init_input = torch.tensor(init_id).unsqueeze(0).to(device)

    with torch.set_grad_enabled(False):
        # Feed the init token to the model
        output = model(init_input)

        # Flatten the logits at the final time step
        logits = output.logits[0,-1]

        # Make a top-k choice and append to the result
        result.append(topk(logits, n=n))
        
        # For max_length times:
        for i in range(max_length):
            # Feed the current sequence to the model and make a choice
            input = torch.tensor(result).unsqueeze(0).to(device)
            output = model(input)
            logits = output.logits[0,-1]
            res_id = topk(logits)

            # If the chosen token is EOS, return the result
            if res_id == tokenizer.eos_token_id:
                break
            else: # Append to the sequence 
                result.append(res_id)
    # IF no EOS is generated, return after the max_len
    return tokenizer.decode(result, skip_special_tokens=True, clean_up_tokenization_spaces=True)

csg_ds=RCDDataset(dataset)
train_dl = DataLoader(csg_ds, batch_size=5)
train_model(model,train_dl)

ans = model_infer(model, tokenizer, f"""All right. It's not Sunday. We don't need a sermon.
to be able to behave like gentlemen. 
it. 
close the window. It was blowing on my neck. 
t seems to me that it's up to us to convince this gentleman (indicating NO. 8) that we're right and he's wrong. Maybe if we each took a minute or two, you know, if we sort of try it on for size.
table. 
guilty. I thought it was obvious. I mean nobody proved otherwise.is on the prosecution. The defendant doesn't have to open his mouth. That's in the Constitution. The Fifth Amendment. You've heard of it. 
I . . . what I meant . . . well, anyway, I think he was guilty. 
man who lived on the second floor right underneath the room where the murder took place. At ten minutes after twelve on the night of the killing he heard loud noises in the upstairs apartment. He said it sounded like a fight. Then he heard the kid say to his father, "I'm gonna kill you.!” A second later he heard a body falling, and he ran to the door of his apartment, looked out, and saw the kid running down the stairs and out of the house. Then he called the police. They found the father with a knife in his chest. 
movies. That's a little ridiculous, isn't it? He couldn't even remember what pictures he saw. 
right. 
testimony don't prove it, then nothing does.""")

print(ans)