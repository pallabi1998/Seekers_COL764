import re
import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
#import beautifulsoup4 as bs4
from bs4 import BeautifulSoup


train_fp = "/scratch/cse/msr/csy227517/IR/Project/train_data.xml"


def read_query(train_fp):
    with open(train_fp, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, "xml")
    topic_tags = bs_data.find_all("top", number=True)
    print(len(topic_tags))
    qcontent = []
    for topic in topic_tags:
        topic_number = topic["num"]
        topic_content = topic.get_text()
        topic_contents = topic_content.split("\n")
        qcontent.append(topic_contents[1])  # 0 for null, 1 for query, 2 for ques, 3 for desc

    return qcontent


print(read_query(train_fp))