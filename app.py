# import streamlit as st
# import pickle

# # Load model and vectorizer
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)

# with open('vectorizer.pkl', 'rb') as f:
#     vectorizer = pickle.load(f)

# # Mapping label numbers to disorder names
# label_map = {
#     0: 'Anxiety',
#     1: 'Bipolar Disorder',
#     2: 'Depression',
#     3: 'OCD (Obsessive-Compulsive Disorder)',
#     4: 'PTSD (Post-Traumatic Stress Disorder)'
# }

# # Streamlit UI
# st.title("🧠 MindScan - Mental Health Predictor")

# st.markdown("Enter a post, message, or description and get an estimated mental health prediction based on ML model.")

# user_input = st.text_area("✍️ Enter your text here:", height=150)

# if st.button("🔍 Predict"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         # Preprocess and predict
#         text_vector = vectorizer.transform([user_input])
#         prediction = model.predict(text_vector)[0]
#         label_name = label_map[prediction]

#         st.success(f"🧠 Prediction: **{label_name}**")


import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Label Map
label_map = {
    0: 'Anxiety',
    1: 'Bipolar Disorder',
    2: 'Depression',
    3: 'OCD',
    4: 'PTSD'
}

# Page config
st.set_page_config(page_title="MindScan", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🧠 MindScan</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>AI-powered Mental Health Predictor</h4><hr>", unsafe_allow_html=True)

# ----- Prediction Section -----
with st.form("predict_form"):
    user_input = st.text_area("✍️ Enter your thoughts here:", height=150)
    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]
        probs = model.predict_proba(vector)[0]

        predicted_label = label_map[prediction]
        st.success(f"🎯 **Predicted Mental Health Condition: `{predicted_label}`**")

        # 📊 Bar Chart
        st.markdown("---")
        st.markdown("### 📊 Class Probabilities")
        prob_df = pd.DataFrame({
            'Disorder': [label_map[i] for i in range(len(probs))],
            'Probability': probs
        }).sort_values(by="Probability", ascending=True)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.barh(prob_df['Disorder'], prob_df['Probability'], color='#4A90E2')
        ax.set_xlabel("Confidence Score")
        ax.set_xlim(0, 1)
        ax.set_facecolor('#f5f5f5')
        fig.patch.set_facecolor('#f5f5f5')

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}', va='center')

        st.pyplot(fig)

# ----- Feedback Section -----
st.markdown("---")
st.markdown("## 💬 Share Feedback")

if "clear_feedback" not in st.session_state:
    st.session_state.clear_feedback = False


with st.form("feedback_form"):
    default_text = "" if st.session_state.clear_feedback else st.session_state.get("feedback_text", "")
    feedback_text = st.text_area("🙏 How was your experience using MindScan?", height=100, key="feedback_text", value=default_text)
    rating = st.slider("⭐ Rate this app:", 1, 5, 3)
    fb_submit = st.form_submit_button("📩 Submit Feedback")


if fb_submit:
    if feedback_text.strip():
        feedback_data = pd.DataFrame([[feedback_text.strip(), rating]],
                                     columns=["Feedback", "Rating"])

        if os.path.exists("feedback.csv"):
            feedback_data.to_csv("feedback.csv", mode='a', header=False, index=False)
        else:
            feedback_data.to_csv("feedback.csv", index=False)

        st.success("✅ Thanks for your feedback!")

        # 🧹 Set flag to clear input and rerun
        st.session_state.clear_feedback = True
        st.experimental_rerun()

    else:
        st.warning("Please enter some feedback before submitting.")
