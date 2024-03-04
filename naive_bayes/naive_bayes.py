import sys

sys.path.append("..")

from data_loading import get_spam_dataset

df = get_spam_dataset()
# print(df.info())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['Body'], df['Label'], test_size=0.2, random_state=42)

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score


predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
train_accuracy = accuracy_score(y_train, model.predict(X_train))

report = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Train accuracy: {train_accuracy}')
print(report)