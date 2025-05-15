import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Data\combined_spam_dataset.csv')

# load the combined dataset
combined_data = pd.read_csv(file_path)

# verify structure
print(combined_data.head())
print(combined_data.dtypes)
stop_words = set(ENGLISH_STOP_WORDS)

def preprocess_text(text):
    if not isinstance(text, str):
        return ''  # or return str(text) if you prefer to keep float values as text
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)  # simple word tokenizer
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)
combined_data['text'] = combined_data['text'].apply(preprocess_text)


# transform text to vectors

# TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(combined_data['text']).toarray()
y = combined_data['label'].values

# split data into train, test, validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

print()

# train SVM model
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

# evaluate model
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# save model
joblib.dump(svm_model, 'Model\Svm\svm_spam_classifier.pkl')
joblib.dump(tfidf, 'Model\Svm\\tfidf_svm_vectorizer.pkl')

# test the model
sample_text = ["Win a free iPhone now!", "Let's meet tomorrow."]
sample_processed = [preprocess_text(text) for text in sample_text]
sample_vector = tfidf.transform(sample_processed).toarray()
predictions = svm_model.predict(sample_vector)

for text, pred in zip(sample_text, predictions):
    print(f"Text: {text}\nPrediction: {'Spam' if pred == 1 else 'Ham'}\n")