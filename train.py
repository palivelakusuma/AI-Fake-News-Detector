import pandas as pd
import pickle
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Combine
data = pd.concat([fake, true], axis=0)

# Shuffle
data = data.sample(frac=1, random_state=42)

# Keep needed columns
data = data[["text", "label"]]

# Clean function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    return text

# Apply cleaning
data["text"] = data["text"].apply(clean_text)

# Split
x = data["text"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7
)

xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Model
model = LogisticRegression(max_iter=1000)

# Train
model.fit(xv_train, y_train)

# Predict
predictions = model.predict(xv_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save
with open("model/fake_news_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("Model saved successfully!")