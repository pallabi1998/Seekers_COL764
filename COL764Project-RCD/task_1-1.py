from sumy.summarizers.luhn import LuhnSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer 
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.kl import KLSummarizer

from nltk.corpus import wordnet
from nltk import word_tokenize

def get_hypernym_count(word):
    synsets = wordnet.synsets(word)
    hypernyms = set()
    for synset in synsets:
        for hypernym in synset.hypernyms():
            hypernyms.add(hypernym.name())
    return len(hypernyms)

def sort_by_hypernyms(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    
    # Get hypernym counts for each word
    word_hypernym_counts = [(word, get_hypernym_count(word)) for word in tokens]
    
    # Sort words based on hypernym counts
    sorted_words = [word for word, _ in sorted(word_hypernym_counts, key=lambda x: x[1], reverse=True)]
    
    return sorted_words

import spacy
import pytextrank
import pke
import nltk
from nltk.corpus import stopwords
import math


from nltk.translate.bleu_score import sentence_bleu

summarizer_luhn = LuhnSummarizer()
summarizer_kl = KLSummarizer()
text = """
<desc>
<p>All right. It's not Sunday. We don't need a sermon.</p>
<p>to be able to behave like gentlemen. </p>
<p>it. </p>
<p>close the window. It was blowing on my neck. </p>
<p>it seems to me that it's up to us to convince this gentleman (indicating NO. 8) that we're right and he's wrong. Maybe if we each took a minute or two, you know, if we sort of try it on for size.</p>
<p>table. </p>
<p>guilty. I thought it was obvious. I mean nobody proved otherwise.is on the prosecution. The defendant doesn't have to open his mouth. That's in the Constitution. The Fifth Amendment. You've heard of it. </p>
<p>I . . . what I meant . . . well, anyway, I think he was guilty. </p>
<p>man who lived on the second floor right underneath the room where the murder took place. At ten minutes after twelve on the night of the killing he heard loud noises in the upstairs apartment. He said it sounded like a fight. Then he heard the kid say to his father, "I'm gonna kill you.!” A second later he heard a body falling, and he ran to the door of his apartment, looked out, and saw the kid running down the stairs and out of the house. Then he called the police. They found the father with a knife in his chest. </p>
<p>movies. That's a little ridiculous, isn't it? He couldn't even remember what pictures he saw. </p>
<p>right. </p>
<p>testimony don't prove it, then nothing does. </p>
</desc>
"""

inp_file = "./RCD2020FIRETASK/trec_formatted_with_p_tags.txt"
inp_file = "./newData.xml"

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


print(titles, texts)







#exit(0)

text_length = len(text)
parser = PlaintextParser.from_string(text, Tokenizer("english"))
#print(dir(parser))
#print(parser.document)
summary1 = summarizer_luhn(parser.document, 2)
print(summary1)

summary2 = summarizer_kl(parser.document, 2)
print(summary2)


nlp = spacy.load("en_core_web_sm")

#tr = pytextrank.TextRank()
# @spacy.Language.component("textrank")
# def my_custom_component_wrapper(doc):
#     tr.analyze(doc)
#     return doc


#tr.load_stopwords()
#nlp.add_pipe("textrank", last = True)
nlp.add_pipe("textrank")
summary_sent = ""
for s in summary1:
    #print(dir(s))
    summary_sent += " "+ s._text
#print(summary_sent)
#nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
extracted_doc = nlp(summary_sent)
print(extracted_doc._.phrases)
print("\n \n")
for p in extracted_doc._.phrases:
    if p.count>=1 and p.count<=3 and p.rank>=0.02:
        print(p.text)



new_text = """
Person 1: Hi! How's your day going?

Person 2: Hey there! It's been pretty good. How about you?

Person 1: Not bad, thanks for asking. Anything exciting happening for you today?

Person 2: Well, I'm meeting some friends for dinner later. So that should be fun. How about you?

Person 1: Sounds like a great plan! I'm just going to relax at home and catch up on some reading about dinasours. Enjoy your dinner!

Person 2: Thanks! Have a relaxing evening. Let's chat again soon!

Person 1: Definitely! Take care, and talk to you soon.
"""

extractor = pke.unsupervised.YAKE()
extractor.load_document(
    input = new_text, language ='en', normalization = None
)

stoplist = stopwords.words("english")
num = 10
extractor.candidate_selection(n=num)
window = 7
use_stems = False
threshold = 0.8
extractor.candidate_weighting(
    window = window, use_stems = use_stems
)
keyphrases = extractor.get_n_best(n=10, threshold=threshold)
print("\nKeyphrases:\n")
print(keyphrases)


def get_rank(phrase):
    return phrase[1]

def get_rank2(phrase):
    return phrase.rank


total_bleu = 0
total_jacc = 0
total_bleu2 = 0
total_jacc2 = 0



import nltk
from nltk.corpus import wordnet
from nltk.corpus import brown
from collections import Counter

nltk.download('brown')

cwords = wordnet.words()


cword_freq = Counter(cwords)



from transformers import BertTokenizer, BertForMaskedLM
from transformers import pipeline

# Load the BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Create a fill-mask pipeline
nlp_fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Function to generate a summary using BERT
def generate_summary(paragraph, num_words=8):
    # Split the paragraph into sentences
    sentences = paragraph.split(". ")

    # Initialize the summary
    summary = []

    # Generate summaries for each sentence
    for sentence in sentences:
        # Generate masked tokens for the sentence
        masked_sentence = sentence + " [MASK]"

        # Use BERT to predict the masked word
        masked_result = nlp_fill_mask(masked_sentence)
        predicted_word = masked_result[0]["token_str"]

        # Append the predicted word to the summary
        summary.append(predicted_word)

        # Stop when the summary reaches the desired number of words
        if len(summary) >= num_words:
            break

    return " ".join(summary)









def jaccard_similarity(str1, str2):
    # Tokenize the strings into sets of words
    set1 = set(str1.split())
    set2 = set(str2.split())

    # Calculate the Jaccard similarity coefficient
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2)) 
    #union = len(set1) + len(set2) - intersection

    if union == 0:
        return 0  # Avoid division by zero if both sets are empty
    else:
        return intersection / union


answer = []

for text1, title in zip(texts, titles):
    
    text = str(text1)
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    text_length = len(text.split("."))
    summary_t = summarizer_kl(parser.document, text_length*0.8)
    summary_sent = ""
    for s in summary_t:
        summary_sent += " "+ s._text
    extracted_doc = nlp(summary_sent)
    selected_phrases = []
    for p in extracted_doc._.phrases:
        if p.count>=1 and p.count<=3 and p.rank>=0.02 and len(p.text.split())>=3 and len(p.text.split())<10  :
            selected_phrases.append([p.text, p.rank])
    sorted_phrases = sorted(selected_phrases, key=get_rank)
    # reference = title.split()
    
    if len(sorted_phrases) == 0:
        sorted_phrases = sorted(extracted_doc._.phrases, key=get_rank2)
        candidate = sorted_phrases[-1].text
    else:
        candidate = sorted_phrases[-1][0]
    # print(sorted_phrases[-1], title)
    # total_jacc += jaccard_similarity(title, candidate)
    # bleu_score = sentence_bleu(title.split(), candidate.split())
    # total_bleu += bleu_score


    # extractor = pke.unsupervised.YAKE()
    # extractor.load_document(
    #     input = text, language ='en', normalization = None
    # )
    # num = 10
    # extractor.candidate_selection(n=num)
    # window = 7
    # use_stems = False
    # threshold = 0.8
    # extractor.candidate_weighting(
    #     window = window, use_stems = use_stems
    # )
    # keyphrases = extractor.get_n_best(n=1, threshold=threshold)[0][0]
    



    # words_t = text1.split()
    # ext_wind = 0
    # min_score = math.inf
    # for i in range(len(words_t)-7):
    #     score = 0
    #     for j in range(2):
    #         score += cword_freq[words_t[i+j]]
    #         if words_t[i+j] not in cword_freq:
    #             score -=10
    #     if score < min_score:
    #         ext_wind=i
    
    # keyphrases = ""
    # for i in range(6):
    #     keyphrases += " " + words_t[i+ext_wind]


    # #keyphrases = generate_summary(text1)       
    # print(keyphrases, ext_wind)
    #sorted_words = sort_by_hypernyms(text1)
    #keyphrases = sorted_words[0] #+" "+ sorted_words[1]
    keyphrases = candidate
    print(title, keyphrases)

    

    total_jacc2 += jaccard_similarity(title, keyphrases)
    bleu_score2 = sentence_bleu(title.split(), keyphrases.split())
    total_bleu2 += bleu_score2
    print(bleu_score2, jaccard_similarity(title, keyphrases))
    answer.append(keyphrases)
    
with open("qnswer_file_dcu", "w") as f:
    for a in answer:
        f.write(a+'\n')

print(total_bleu/len(texts), total_jacc/len(texts))
print(total_bleu2/len(texts), total_jacc2/len(texts))


