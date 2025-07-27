import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake['label'] = 0  # FAKE
true['label'] = 1  # REAL

# Combine and shuffle
data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)

print("✅ Dataset loaded successfully!")
print("📋 Available columns:", data.columns)
print("📝 Sample data:\n", data[['text', 'label']].head())

# Features and labels
X = data['text']
y = data['label']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vect = tfidf.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.25, random_state=42)

# Model training
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {round(score*100, 2)}%")

# Live prediction
while True:
    user_input = input("\n🔍 Enter your own news article below to test (type 'exit' to quit):\n")
    if user_input.lower() == 'exit':
        break
    user_vect = tfidf.transform([user_input])
    prediction = model.predict(user_vect)[0]
    print("🧠 Prediction:", "REAL ✅" if prediction == 1 else "FAKE ❌")
