from sumy.summarizers.luhn import LuhnSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer 
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.kl import KLSummarizer


import spacy
import pytextrank
import pke
import nltk
from nltk.corpus import stopwords


summarizer_luhn = LuhnSummarizer()
summarizer_kl = KLSummarizer()
text = """
Proactively retrieving relevant information to contextualize conversations has potential applications in better understanding the
conversational content between communicating parties. Since in
contrast to traditional IR, there is no explicitly formulated userquery, an important research challenge is to first identify the candidate segments of text that may require contextualization for a
better comprehension of their content, and then make use of these
identified segments to formulate a query and eventually retrieve the
potentially relevant information to augment a conversation. In this
paper, we propose a generic unsupervised framework that involves
shifting overlapping windows of terms through a conversation
and estimating scores indicating the likelihood of the existence of
an information need within these segments. Within our proposed
framework, we investigate a query performance prediction (QPP)
based approach for scoring these candidate term windows with the
hypothesis that a term window that indicates a higher specificity is
likely to be indicative of a potential information need requiring contextualization.
"""
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