import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import load_model

# Sample data (you should replace this with your own dataset)
text_data = [
    "This is the input text for summarization.",
    "The LSTM model will create a summary.",
    "You can customize and improve this model."
]

summary_data = [
    "Text summarization using LSTM is demonstrated.",
    "LSTM creates concise summaries of input text.",
    "Customizing the model can lead to better results."
]

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
tokenizer.fit_on_texts(summary_data)

vocab_size = len(tokenizer.word_index) + 1

# Convert text and summary data to sequences
text_sequences = tokenizer.texts_to_sequences(text_data)
summary_sequences = tokenizer.texts_to_sequences(summary_data)

# Pad sequences to the same length
max_sequence_length = max(len(seq) for seq in text_sequences)

text_sequences = pad_sequences(text_sequences, maxlen=max_sequence_length, padding='post')
summary_sequences = pad_sequences(summary_sequences, maxlen=max_sequence_length, padding='post')

# Create the LSTM-based summarization model
model = Sequential()
model.add(Embedding(vocab_size, 128))
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=sparse_categorical_crossentropy,
    metrics=['accuracy']
)

# Train the model (you should replace this with your training data)
model.fit(text_sequences, summary_sequences, epochs=10)

# Save the trained model for later use
model.save('text_summarization_model.h5')

# To generate summaries, load the model and provide input text for prediction
# (Assuming you have already trained the model and loaded it)
loaded_model = load_model('text_summarization_model.h5')

input_text = "Your input text goes here."
input_sequence = tokenizer.texts_to_sequences([input_text])
input_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post')
predicted_summary = loaded_model.predict(input_sequence)

# Decode the predicted summary
predicted_summary_text = tokenizer.sequences_to_texts([np.argmax(predicted_summary, axis=1)])[0]
print("Predicted Summary:", predicted_summary_text)





for text1, title in zip(texts, titles):
    
    
    words_t = text.split()
    ext_wind = 0
    min_score = math.inf
    for i in range(len(words_t)-7):
        score = 0
        for j in range(2):
            score += cword_freq[words_t[i+j]]
            if words_t[i+j] not in cword_freq:
                score -=10
        if score < min_score:
            ext_wind=i
    
    keyphrases = ""
    for i in range(6):
        keyphrases += " " + words_t[i+ext_wind]


    

    

print(total_bleu/len(texts), total_jacc/len(texts))
print(total_bleu2/len(texts), total_jacc2/len(texts))


