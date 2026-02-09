import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from huggingface_hub import HfApi
import os

# --- CONFIG: THE MEGA BASKET (Restored) ---
TRAINING_BASKET = [
    # BANKS & FINANCE
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BAJFINANCE.NS",
    # IT & TECH
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    # AUTO
    "TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "EICHERMOT.NS",
    # METALS & ENERGY
    "TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
    # CONSUMER & PHARMA
    "ITC.NS", "HINDUNILVR.NS", "ASIANPAINT.NS", "TITAN.NS", "SUNPHARMA.NS", "CIPLA.NS",
    # INDEX
    "^NSEI"
]

REPO_ID = "Shakuji/stock-predictor-model"
HF_TOKEN = os.getenv("HF_TOKEN")
HORIZONS = {"1d": 1, "1w": 5, "1m": 20}

# --- MATH INDICATORS (Full Suite Restored) ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_features(df):
    # 1. Trend (MACD)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 2. Volatility (Bollinger Bands & ATR)
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma_20 + (std_20 * 2)
    df['BB_Lower'] = sma_20 - (std_20 * 2)
    
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(window=14).mean()

    # 3. Candlestick Patterns
    body_size = abs(df['Close'] - df['Open'])
    full_range = df['High'] - df['Low']
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    df['Is_Doji'] = (body_size <= (full_range * 0.1)).astype(int)
    df['Is_Hammer'] = ((lower_shadow >= (body_size * 2)) & (upper_shadow <= (body_size * 0.5))).astype(int)
    
    # 4. Memory Lags
    for lag in range(1, 6): 
        df[f'Return_Lag_{lag}'] = df['Close'].pct_change(lag)

    return df

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="2y") # 2 Years Rolling Window
        if len(df) < 200: return None
        
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["RSI"] = calculate_rsi(df["Close"])
        df["Volatility"] = df["Close"].rolling(window=10).std()
        
        df = add_features(df)
        df = df.dropna()
        return df
    except:
        return None

def train():
    all_data = []
    print("🚀 Collecting Data from 30+ Market Leaders...")
    
    for ticker in TRAINING_BASKET:
        df = get_stock_data(ticker)
        if df is not None:
            all_data.append(df)
            
    if not all_data:
        print("❌ No data collected.")
        return

    full_df = pd.concat(all_data)
    print(f"✅ Dataset Ready: {len(full_df)} rows.")

    # Full Feature List
    features = ["SMA_10", "SMA_50", "RSI", "Volatility", "MACD", "MACD_Signal", 
                "BB_Upper", "BB_Lower", "ATR", "Is_Doji", "Is_Hammer"]
    features += [f'Return_Lag_{i}' for i in range(1, 6)]
    
    api = HfApi()

    # --- TRAIN THE COUNCIL FOR EACH HORIZON ---
    for name, days in HORIZONS.items():
        print(f"\n🧠 Training 'Council of 9' for {name.upper()} (Target: {days} days)...")
        
        # Prepare Target
        df_train = full_df.copy()
        df_train["Target"] = (df_train["Close"].shift(-days) > df_train["Close"]).astype(int)
        df_train = df_train.dropna()
        
        X = df_train[features]
        y = df_train["Target"]

        # --- THE DEMOCRACY (Voting Classifier) ---
        # 1. The Veteran (Random Forest)
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        
        # 2. The Specialist (Gradient Boosting)
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        
        # 3. The Skeptic (Logistic Regression)
        lr = LogisticRegression(random_state=42, max_iter=1000)
        
        # Voting Mechanism
        model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            voting='soft' 
        )
        
        model.fit(X, y)
        
        # Save & Upload
        filename = f"model_{name}.pkl"
        joblib.dump(model, filename)
        
        print(f"☁️ Uploading {filename}...")
        try:
            api.upload_file(
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="model",
                token=HF_TOKEN
            )
        except Exception as e:
            print(f"Upload failed: {e}")

if __name__ == "__main__":
    train()
