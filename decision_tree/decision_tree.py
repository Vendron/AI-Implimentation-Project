import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV,  KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys

sys.path.append("..")

from data_loading import get_spam_dataset

combined_df = get_spam_dataset()
print(combined_df.info())
X_train, X_test, y_train, y_test = train_test_split(combined_df["Body"], combined_df["Label"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


param_grid = {
    'max_depth': np.arange(1, 20),
    'min_samples_split': [2, 5, 10],
}

clf = DecisionTreeClassifier()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(clf, param_grid, cv=kf, scoring='accuracy', return_train_score=True)
# grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', return_train_score=True)
grid_search.fit(X_train_tfidf, y_train)

best_estimator = grid_search.best_estimator_
print(best_estimator)
best_estimator.fit(X_train_tfidf, y_train)
y_train_pred = best_estimator.predict(X_train_tfidf)
y_test_pred = best_estimator.predict(X_test_tfidf)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(classification_report(y_test, y_test_pred))


mean_test_scores = grid_search.cv_results_['mean_test_score']
mean_train_scores = grid_search.cv_results_['mean_train_score']

plt.figure(figsize=(10, 6))
plt.plot(param_grid['max_depth'], mean_test_scores[:len(param_grid['max_depth'])], label='Test Accuracy')
plt.plot(param_grid['max_depth'], mean_train_scores[:len(param_grid['max_depth'])], label='Train Accuracy')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.title('Test of Decision Tree')
plt.legend()
plt.grid(True)
plt.show()

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)