import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("data/health_data_final.csv")

print("CSV shape:", df.shape)
print("CSV columns:", df.columns.tolist())
print("First few rows:\n", df.head())

# Drop rows where symptoms or disease is missing/blank
df = df.dropna(subset=["symptoms", "disease"])
df = df[df["symptoms"].str.strip() != ""]

# Convert to string just in case
X = df["symptoms"].astype(str)
y = df["disease"].astype(str)

# Build pipeline
pipeline = make_pipeline(TfidfVectorizer(stop_words=None), LogisticRegression(max_iter=1000))

# Debug: Check a few rows before fitting
print("Sample symptoms data:", X.head().tolist())
print("Sample labels:", y.head().tolist())

# Train
pipeline.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("[INFO] Model trained and saved to model.pkl")
