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

# Apply simplified text preprocessing
df['Body'] = df['Body'].apply(simple_preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df['Body'], df['Label'], test_size=0.2, random_state=42)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
train_accuracy = accuracy_score(y_train, model.predict(X_train))

report = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Train accuracy: {train_accuracy}')
print(report)

