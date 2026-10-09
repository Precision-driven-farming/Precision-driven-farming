"""
Member 8 - Farmer Chatbot NLP Layer
Role: TF-IDF + Logistic Regression Intent Classifier
Sits on top of Member 9 spraying_pipeline.py output
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import os

INTENTS_PATH = "intents.json"

def load_intents():
    with open(INTENTS_PATH, 'r') as f:
        data = json.load(f)
    texts = []
    labels = []
    for item in data:
        if item["tag"]!= "unknown":
            for pattern in item["patterns"]:
                texts.append(pattern)
                labels.append(item["tag"])
    return texts, labels, data

# Load training data
my_texts, my_tags, my_intents_data = load_intents()

print(f"[REAL] Loaded {len(my_texts)} farmer questions for training")

# TF-IDF Vectorizer - converts farmer words to numbers
# This is lightweight for mobile use 
my_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1,2) # unigram + bigram for farmer language
)

X_all = my_vectorizer.fit_transform(my_texts)

# Train test split to show accuracy 
X_train, X_test, y_train, y_test = train_test_split(X_all, my_tags, test_size=0.2, random_state=42)

# Logistic Regression Model 
my_model = LogisticRegression(max_iter=1000, multi_class='auto')
my_model.fit(X_train, y_train)

# Evaluate - for documentation
y_pred = my_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[REAL] Model accuracy: {acc*100:.2f}% - for docs")

def predict_intent(user_input: str):
    """
    Predicts farmer intent with confidence
    Returns: intent tag, confidence score
    Safety: Always adds human verification note outside
    """
    if not user_input or len(user_input.strip()) < 2:
        return "unknown", 0.0

    # Transform user input
    vec = my_vectorizer.transform([user_input.lower()])

    # Predict
    intent = my_model.predict(vec)[0]
    probas = my_model.predict_proba(vec)[0]
    confidence = float(max(probas))

    # Low confidence -> unknown for safety
    if confidence < 0.4:
        return "unknown", confidence

    return intent, confidence

def get_response_for_intent(intent_tag):
    """Get response from intents.json"""
    for item in my_intents_data:
        if item["tag"] == intent_tag:
            return item["responses"][0]
    return "I didn't understand. Please ask about spray zones."

# Test for my own verification
if __name__ == "__main__":
    test_q = "where to spray?"
    tag, conf = predict_intent(test_q)
    print(f"Q: {test_q} -> Intent: {tag} Confidence: {conf}")
