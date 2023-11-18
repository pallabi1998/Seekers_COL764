from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode sentences
sentence1 = "This is the first sentence."
sentence2 = "Here comes the second sentence."

tokens1 = tokenizer(sentence1, return_tensors='pt')
tokens2 = tokenizer(sentence2, return_tensors='pt')

# Get BERT embeddings
with torch.no_grad():
    outputs1 = model(**tokens1)
    outputs2 = model(**tokens2)

# Extract the embeddings from the output
embeddings1 = outputs1.last_hidden_state.mean(dim=1).numpy()
embeddings2 = outputs2.last_hidden_state.mean(dim=1).numpy()

# Calculate cosine similarity
similarity = cosine_similarity(embeddings1, embeddings2)[0][0]

print(f"Cosine Similarity: {similarity} between {embeddings1.shape} and {embeddings2.shape}")
