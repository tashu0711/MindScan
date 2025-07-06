import streamlit as st
import pickle

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Mapping label numbers to disorder names
label_map = {
    0: 'Anxiety',
    1: 'Bipolar Disorder',
    2: 'Depression',
    3: 'OCD (Obsessive-Compulsive Disorder)',
    4: 'PTSD (Post-Traumatic Stress Disorder)'
}

# Streamlit UI
st.title("🧠 MindScan - Mental Health Predictor")

st.markdown("Enter a post, message, or description and get an estimated mental health prediction based on ML model.")

user_input = st.text_area("✍️ Enter your text here:", height=150)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess and predict
        text_vector = vectorizer.transform([user_input])
        prediction = model.predict(text_vector)[0]
        label_name = label_map[prediction]

        st.success(f"🧠 Prediction: **{label_name}**")
