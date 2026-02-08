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

# SVG Icons (inline, 22px, teal #00B4D8)
SVG_BRAIN = '<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#00B4D8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a5 5 0 0 1 4.9 4.1A4 4 0 0 1 20 10a4 4 0 0 1-1.3 7.6A5 5 0 0 1 12 22a5 5 0 0 1-6.7-4.4A4 4 0 0 1 4 10a4 4 0 0 1 3.1-3.9A5 5 0 0 1 12 2z"/><path d="M12 2v20"/><path d="M8 6c2 1 4 1 6 0"/><path d="M7 10.5c2.5 1 5.5 1 8 0"/><path d="M8 18c2-1 4-1 6 0"/></svg>'

SVG_EDIT = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00B4D8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.83 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/></svg>'

SVG_SEARCH = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>'

SVG_SHIELD = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00B4D8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>'

SVG_CHART = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00B4D8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="12" width="4" height="9" rx="1"/><rect x="10" y="7" width="4" height="14" rx="1"/><rect x="17" y="3" width="4" height="18" rx="1"/></svg>'

SVG_MESSAGE = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00B4D8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>'

SVG_SEND = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>'

SVG_INFO = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#8899AA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'

# Page config
st.set_page_config(page_title="MindScan", layout="centered")

# --- Raw CSS ---
st.markdown("""
<style>
    /* Hide streamlit default stuff */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Root variables */
    :root {
        --navy: #1B2A4A;
        --navy-light: #243656;
        --teal: #00B4D8;
        --teal-dark: #0096B4;
        --bg: #0F1923;
        --card-bg: #162133;
        --text-primary: #E8EDF3;
        --text-secondary: #8899AA;
        --border: #2A3A52;
    }

    /* Global background */
    .stApp {
        background-color: var(--bg);
    }

    /* Card container */
    .card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: 40px 20px 10px;
    }
    .app-header h1 {
        color: var(--text-primary);
        font-family: 'Segoe UI', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 12px 0 4px;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: var(--text-secondary);
        font-size: 1rem;
        margin: 0;
        font-weight: 400;
    }

    /* Section titles */
    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--text-primary);
        font-family: 'Segoe UI', sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 16px;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #0a2a2a, #0d3340);
        border: 1px solid #00B4D840;
        border-radius: 10px;
        padding: 20px 24px;
        display: flex;
        align-items: center;
        gap: 14px;
        margin-top: 12px;
    }
    .result-box .label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin: 0;
    }
    .result-box .value {
        color: #00B4D8;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 2px 0 0;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: var(--border);
        margin: 32px 0;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 20px;
        margin-top: 12px;
    }
    .app-footer p {
        color: var(--text-secondary);
        font-size: 0.78rem;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
    }

    /* Streamlit form buttons */
    .stFormSubmitButton > button {
        background: var(--teal) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: background 0.2s !important;
    }
    .stFormSubmitButton > button:hover {
        background: var(--teal-dark) !important;
    }

    /* Text area styling */
    .stTextArea textarea {
        background: var(--navy) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 1px var(--teal) !important;
    }

    /* Slider track */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: var(--teal) !important;
        border-color: var(--teal) !important;
    }
    .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] {
        background: transparent !important;
    }
    .stSlider div[data-baseweb="slider"] > div:first-child > div {
        background: var(--border) !important;
    }
    .stSlider div[data-baseweb="slider"] > div:first-child > div > div {
        background: var(--teal) !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        box-shadow: 0 0 6px rgba(0, 180, 216, 0.4) !important;
    }

    /* Slider value labels */
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"],
    .stSlider [data-baseweb="slider"] [data-testid="stTickBarMin"],
    .stSlider [data-baseweb="slider"] [data-testid="stTickBarMax"] {
        color: var(--text-primary) !important;
        font-size: 0.85rem !important;
    }

    /* Labels */
    .stTextArea label, .stSlider label, .stSelectSlider label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    /* Success/Warning messages */
    .stSuccess {
        background-color: #0a2a2a !important;
        border-left-color: var(--teal) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown(f"""
<div class="app-header">
    {SVG_BRAIN}
    <h1>MindScan</h1>
    <p>AI-powered Mental Health Condition Predictor</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# --- Prediction Section ---
st.markdown(f"""
<div class="section-title">
    {SVG_EDIT}
    <span>Describe your thoughts</span>
</div>
""", unsafe_allow_html=True)

with st.form("predict_form"):
    user_input = st.text_area("Write a post, journal entry, or describe how you're feeling:", height=150, label_visibility="collapsed", placeholder="Write a post, journal entry, or describe how you're feeling...")
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]
        probs = model.predict_proba(vector)[0]

        predicted_label = label_map[prediction]

        st.markdown(f"""
        <div class="result-box">
            {SVG_SHIELD}
            <div>
                <p class="label">Predicted Condition</p>
                <p class="value">{predicted_label}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Probability Chart ---
        st.markdown(f"""
        <div style="margin-top: 28px;">
            <div class="section-title">
                {SVG_CHART}
                <span>Confidence Breakdown</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            'Disorder': [label_map[i] for i in range(len(probs))],
            'Probability': probs
        }).sort_values(by="Probability", ascending=True)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.barh(prob_df['Disorder'], prob_df['Probability'], color='#00B4D8', height=0.55, edgecolor='none')
        ax.set_xlabel("Confidence Score", color='#8899AA', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_facecolor('#162133')
        fig.patch.set_facecolor('#162133')
        ax.tick_params(colors='#8899AA', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#2A3A52')
        ax.spines['left'].set_color('#2A3A52')
        ax.xaxis.label.set_color('#8899AA')

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{width:.0%}', va='center', color='#E8EDF3', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

# --- Divider ---
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --- Feedback Section ---
st.markdown(f"""
<div class="section-title">
    {SVG_MESSAGE}
    <span>Share Feedback</span>
</div>
""", unsafe_allow_html=True)

with st.form("feedback_form"):
    feedback_text = st.text_area("How was your experience using MindScan?", height=100, label_visibility="collapsed", placeholder="How was your experience using MindScan?")
    rating = st.select_slider(
        "Rating",
        options=[1, 2, 3, 4, 5],
        value=3,
        format_func=lambda x: {1: "1 - Poor", 2: "2 - Fair", 3: "3 - Good", 4: "4 - Very Good", 5: "5 - Excellent"}[x]
    )
    fb_submit = st.form_submit_button("Submit Feedback")

if fb_submit:
    if feedback_text.strip():
        feedback_data = pd.DataFrame([[feedback_text.strip(), rating]],
                                     columns=["Feedback", "Rating"])

        if os.path.exists("feedback.csv"):
            feedback_data.to_csv("feedback.csv", mode='a', header=False, index=False)
        else:
            feedback_data.to_csv("feedback.csv", index=False)

        st.success("Thanks for your feedback!")
        st.rerun()
    else:
        st.warning("Please enter some feedback before submitting.")

# --- Footer ---
st.markdown(f"""
<div class="divider"></div>
<div class="app-footer">
    <p>{SVG_INFO} This tool is for informational purposes only. Not a substitute for professional medical advice.</p>
</div>
""", unsafe_allow_html=True)
