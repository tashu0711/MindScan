import pickle
import numpy as np

# Step 1: Load saved model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Step 2: Label mapping (you can edit this as per actual label mapping)
label_map = {
    0: 'anxiety',
    1: 'bipolar',
    2: 'depression',
    3: 'ocd',
    4: 'ptsd'
}

# Step 3: Take user input
text = input("📝 Enter a Reddit post or message:\n> ")

# Step 4: Vectorize input text
X = vectorizer.transform([text])

# Step 5: Predict label
pred = model.predict(X)[0]
print(f"\n🧠 Model Prediction: {label_map[pred].upper()}")
