import sys
import string
import re
sys.path.append("..")

from data_loading import get_spam_dataset

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import nltk
import re
import joblib


nltk.download('punkt')
nltk.download('stopwords')

df = get_spam_dataset()
# print(df.info())

def simple_preprocess_text(text):
    # Remove 'Subject:' and any non-letter characters, convert to lowercase
    text = re.sub(r'Subject:', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]

    # Join tokens back to string
    text = ' '.join(tokens)

    return text

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

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Join tokens back to string
    text = ' '.join(tokens)

    return text

# Apply simplified text preprocessing
df['Body'] = df['Body'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df['Body'], df['Label'], test_size=0.2, random_state=42)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, df['Body'], df['Label'], cv=kf, scoring='accuracy')

param_grid = {
    'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidfvectorizer__min_df': [1, 2, 3],
    'tfidfvectorizer__max_df': [0.8, 0.9, 1.0],
    'multinomialnb__alpha': [0.1, 0.5, 1.0]
}

grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
print("Training on grid search")
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

best_model = make_pipeline(TfidfVectorizer(ngram_range=best_params['tfidfvectorizer__ngram_range'],
                                           min_df=best_params["tfidfvectorizer__min_df"],
                                           max_df=best_params["tfidfvectorizer__max_df"],
                                           ), MultinomialNB(alpha=best_params['multinomialnb__alpha']))
print("Training the model with best parameters")
best_model.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# best_model.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
train_accuracy = accuracy_score(y_train, best_model.predict(X_train))

report = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Train accuracy: {train_accuracy}')
print(report)

joblib.dump(best_model, 'naive_bayes_model.pkl')