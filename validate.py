import re
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words = set(ENGLISH_STOP_WORDS)
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(script_dir, 'Model', 'Knn', 'knn_spam_classifier.pkl')
vectorizer_path = os.path.join(script_dir, 'Model', 'Knn', 'tfidf_knn_vectorizer.pkl')
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)




# load the combined dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Data\combined_spam_dataset.csv')
combined_data = pd.read_csv(file_path)

total = 50000
sample_texts = combined_data['text'][:total]
label = combined_data['label'][:total]
# === Sample test data ===
processed_texts = [preprocess_text(text) for text in sample_texts]
X_test = vectorizer.transform(processed_texts)
# === Make predictions ===
predictions = model.predict(X_test)

    # === Display results ===
correct = 0
for label, pred in zip(label, predictions):
    if label == pred:
        correct += 1
print(correct/total)

'''
sample_texts = [""]
processed_texts = [preprocess_text(text) for text in sample_texts]
X_test = vectorizer.transform(processed_texts)
X_test = X_test.toarray()
# === Make predictions ===
predictions = model.predict(X_test)

# === Display results ===
for text, pred in zip(sample_texts, predictions):
    label = "Spam" if pred == 1 else "Ham"
    print(f"Text: '{text}' => Prediction: {label}")
'''