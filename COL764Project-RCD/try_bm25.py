from rank_bm25 import BM25Okapi
from collections import Counter

corpus = [
    "Hello kitty!",
    "It is quite windy in London",
    "How is the weather today?",
    "windy windy windy day hi in a windi city",
    
    
]

corpus2 = [
    "Welcome to New York! it's been waiting for you London Boy",
    "London is a windy place",
    "what news do you have about the windy london?",
    "what a beautiful cat",
    "This is a purring cat",
    "Katniss Everdeen",
    "The kitty party!",
    "There's a kitten over there",
    "Oh my baby thanks for the cat"
]

tokenized_corpus = [doc.split(" ") for doc in corpus]
tokenized_corpus2 = [doc.split(" ") for doc in corpus2]

bm25 = BM25Okapi(tokenized_corpus)
bm252 = BM25Okapi(tokenized_corpus2)
print(dir(bm252))
print(bm25.doc_freqs, bm25.corpus_size, bm25.doc_len, bm25.idf, bm25.avgdl)

# <rank_bm25.BM25Okapi at 0x1047881d0>

query = "windy London"
tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)


print(doc_scores)
print( bm252.get_scores(tokenized_query))
print(bm25.get_top_n(tokenized_query, corpus, n=5))








query = "baby cat"
tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)

print(doc_scores)

print(bm25.get_top_n(tokenized_query, corpus, n=5))
# ['It is quite windy in London']

vocab = set(word for doc_freqs_dict in bm25.doc_freqs for word in doc_freqs_dict.keys())

# Create a Counter for word frequencies based on document frequencies
word_frequencies = Counter({word: sum(doc_freqs_dict.get(word, 0) for doc_freqs_dict in bm25.doc_freqs) for word in vocab})

# Print the vocabulary and word frequencies
print("Vocabulary:", list(vocab))
print("Word Frequencies:", dict(word_frequencies))

