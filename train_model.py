import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Step 1: Load cleaned data
df = pd.read_csv('data/cleaned_data.csv')


df = df.dropna(subset=['text'])  # ⬅️ Fix NaN issue
# Step 2: Set features and labels
X = df['text']
y = df['target']

# Step 3: Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Step 5: Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("\n🔍 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 7: Save the model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model training complete! Files saved as 'model.pkl' and 'vectorizer.pkl'")
print("NaN in text column:", df['text'].isnull().sum())
