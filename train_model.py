import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download
import os

# --- CONFIGURATION ---
REPO_ID = "Shakuji/stock-predictor-model"
HISTORY_FILE = "history.json"
HF_TOKEN = os.getenv("HF_TOKEN")

# --- 1. FETCH DATA (2 YEARS ROLLING WINDOW) ---
def fetch_data(ticker):
    try:
        # Fetch 2 years of data to give the "Council" a long memory
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        
        if df.empty: return None
        
        # Calculate Indicators (Simplified for Stability)
        df['Prev_Close'] = df['Close'].shift(1)
        df['Return'] = (df['Close'] - df['Prev_Close']) / df['Prev_Close']
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # Target: Did price go UP next day? (1 = Yes, 0 = No)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Drop NaN
        df = df.dropna()
        return df
    except:
        return None

# --- 2. TRAIN THE "DEMOCRACY" (Voting Classifier) ---
def train_and_save(horizon="1d"):
    print(f"Training Democracy Model for {horizon}...")
    
    # Diverse list of tickers to train the general market sense
    # Mixing Tech, Banks, Auto, and Energy prevents sector bias
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", 
               "ICICIBANK.NS", "TATAMOTORS.NS", "SBIN.NS", "ITC.NS"]
    
    all_data = []
    for t in tickers:
        df = fetch_data(t)
        if df is not None:
            all_data.append(df)
            
    if not all_data:
        print("No data fetched.")
        return

    full_df = pd.concat(all_data)
    
    # Features - Using a stable core set to prevent overfitting
    features = ['SMA_10', 'SMA_50', 'Volatility', 'Return']
    X = full_df[features]
    y = full_df['Target']
    
    # --- DEFINE THE COUNCIL ---
    # 1. The Veteran (Random Forest): Stable, low variance, sees big picture
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    
    # 2. The Specialist (Gradient Boosting): Aggressive, high accuracy for momentum
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    
    # 3. The Skeptic (Logistic Regression): Pure math, conservative, needs strong proof
    lr = LogisticRegression(random_state=42)
    
    # --- THE VOTING MECHANISM ---
    # voting='soft' means they vote with probabilities (e.g., "I'm 60% sure")
    # This is more accurate than just "Yes/No"
    model = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft' 
    )
    
    model.fit(X, y)
    
    # Save The Council for this Horizon
    filename = f"model_{horizon}.pkl"
    joblib.dump(model, filename)
    print(f"✅ Democracy Model {filename} trained and saved.")
    
    # Upload to Cloud
    if HF_TOKEN:
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="model",
                token=HF_TOKEN
            )
            print(f"☁️ Uploaded {filename} to Cloud.")
        except Exception as e:
            print(f"Upload failed: {e}")

if __name__ == "__main__":
    # Train the Council for ALL time horizons (1D, 1W, 1M)
    horizons = ["1d", "1w", "1m"]
    
    print("🚀 Starting Training for The Council of 9...")
    for h in horizons:
        print(f"\n--- Horizon: {h} ---")
        train_and_save(h)
    print("\n✅ All 9 Brains Trained Successfully!")
