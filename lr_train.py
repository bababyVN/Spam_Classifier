import pandas as pd
import os
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Path to the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Data/combined_spam_dataset.csv')

# Load the dataset
combined_data = pd.read_csv(file_path)

# Check structure
print(combined_data.head())
print(combined_data.dtypes)

# Preprocessing

stop_words = set(ENGLISH_STOP_WORDS)
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

combined_data['text'] = combined_data['text'].apply(preprocess_text)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(combined_data['text']).toarray()
y = combined_data['label'].values

# Split dataset (80% train, 10% validation, 10% test)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)  # ~10% of original

# Logistic Regression with GridSearchCV
param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"]
}

logreg = LogisticRegression()
grid_search = GridSearchCV(logreg, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV score (f1):", grid_search.best_score_)

# Retrain best model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Evaluation
train_preds = best_model.predict(X_train)
val_preds = best_model.predict(X_val)
test_preds = best_model.predict(X_test)

print("Train F1:", f1_score(y_train, train_preds))
print("Val F1:", f1_score(y_val, val_preds))
print("Test F1:", f1_score(y_test, test_preds))

print("Classification Report (Test):\n", classification_report(y_test, test_preds))

# Save model and vectorizer
joblib.dump(best_model, os.path.join(script_dir, 'Model\Lr\logreg_spam_model.pkl'))
joblib.dump(tfidf_vectorizer, os.path.join(script_dir, 'Model\Lr\\tfidf_vectorizer_1000.pkl'))

# test the model
sample_text = ["Win a free iPhone now!", "Let's meet tomorrow."]
sample_processed = [preprocess_text(text) for text in sample_text]
sample_vector = tfidf_vectorizer.transform(sample_processed).toarray()
predictions = best_model.predict(sample_vector)

for text, pred in zip(sample_text, predictions):
    print(f"Text: {text}\nPrediction: {'Spam' if pred == 1 else 'Ham'}\n") 