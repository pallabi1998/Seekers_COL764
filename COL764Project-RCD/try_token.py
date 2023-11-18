from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Tokenize a sentence
sentence = "This is an example sentence. It is running over! I donno what i'll do"
passages_text = [sentence, sentence, sentence, sentence]
tokens = tokenizer.tokenize(sentence)
tokenized_corpus = tokenizer( passages_text, padding=False, truncation=False )
# Print the tokens
print("Original Sentence:", sentence)
print("Tokens:", tokens)
print(tokenized_corpus)
