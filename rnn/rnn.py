import sys
import string
import re
sys.path.append("..")
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from data_loading import get_spam_dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Remove 'Subject:' and any non-letter characters
    text = re.sub(r'Subject:', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Convert text to lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back to string
    text = ' '.join(tokens)

    return text

df = get_spam_dataset()

# Assuming `df` is your DataFrame with emails
df['Body'] = df['Body'].apply(preprocess_text)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['Body'], df['Label'], test_size=0.2, random_state=42)

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
max_len = 50 # Adjust based on your dataset
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# Define the RNN model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_len),
    LSTM(100, return_sequences=True),
    Dropout(0.5),
    LSTM(50),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_data=(X_test_pad, y_test))

# Evaluate the model
_, accuracy = model.evaluate(X_test_pad, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the model
model.save('spam_detection_rnn_model.h5')

#Plot the accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()