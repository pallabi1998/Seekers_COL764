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

# Example usage
sentence = "I went to the bank to deposit money."
sorted_words = sort_by_hypernyms(sentence)
print(sorted_words)
