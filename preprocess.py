import pandas as pd
import re

# Load raw data
df = pd.read_csv('data/cleaned_data.csv')

# Drop unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Drop rows with missing text
df = df.dropna(subset=['text'])

# Basic text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)       # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)   # remove special chars/numbers
    text = re.sub(r'\s+', ' ', text).strip()   # remove extra whitespace
    return text

df['text'] = df['text'].apply(clean_text)

# Drop empty rows after cleaning
df = df[df['text'].str.strip() != '']

# Save cleaned data
df.to_csv('data/cleaned_data.csv', index=False)
print(f"Preprocessing done! Shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}")
