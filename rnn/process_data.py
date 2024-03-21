import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('wordnet')

def init_preprocessing():
    global ENGLISH_STOP_WORDS, lemmatizer
    # Set of English stopwords
    ENGLISH_STOP_WORDS = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Performs initial cleaning of the text.
    """
    text = re.sub(r'Subject:', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize_and_lemmatize(text):
    """
    Tokenizes and lemmatizes the text, removing stopwords.
    """
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in ENGLISH_STOP_WORDS]
    return lemmatized_tokens

def advanced_preprocess_text(text):
    """
    Applies cleaning, tokenization, lemmatization, and stopwords removal.
    """
    cleaned_text = clean_text(text)
    preprocessed_tokens = tokenize_and_lemmatize(cleaned_text)
    return ' '.join(preprocessed_tokens)

def prepare_sequences(texts, max_len=50):
    """
    Prepares text sequences for LSTM model input.
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences, tokenizer
