import os
import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib
from huggingface_hub import HfApi

# --- CONFIG ---
SYMBOL = "RELIANCE.NS" 
# CHANGE THIS to your actual Hugging Face username
REPO_ID = "YOUR_HF_USERNAME/stock-predictor-model" 
MODEL_FILE = "model.pkl"

def train():
    print(f"Fetching data for {SYMBOL}...")
    stock = yf.Ticker(SYMBOL)
    df = stock.history(period="max")

    # 1. Create Indicators
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["Return"] = df["Close"].pct_change()

    # 2. Target: 1 if Price UP tomorrow, else 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    features = ["SMA_10", "SMA_50", "Return", "Open", "High", "Low", "Close"]
    X = df[features]
    y = df["Target"]

    # 3. Train XGBoost
    print("Training XGBoost Model...")
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X, y)

    # 4. Accuracy Check
    recent_X = X.iloc[-30:] 
    recent_y = y.iloc[-30:]
    acc = accuracy_score(recent_y, model.predict(recent_X))
    print(f"Recent Accuracy: {acc:.2%}")

    # 5. Save & Upload
    if acc > 0.5:
        joblib.dump(model, MODEL_FILE)
        print("Uploading to Cloud...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=MODEL_FILE,
            path_in_repo=MODEL_FILE,
            repo_id=REPO_ID,
            repo_type="model",
            token=os.getenv("HF_TOKEN")
        )
        print("Upload Success!")
    else:
        print("Model weak. Skipping update.")

if __name__ == "__main__":
    train()
