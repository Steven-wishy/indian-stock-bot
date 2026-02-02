import os
import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib
from huggingface_hub import HfApi
import numpy as np

# --- CONFIG ---
# 1. We train on 10 major stocks to learn general patterns
TRAINING_BASKET = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
    "TATAMOTORS.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS"
]
# 2. We also fetch the Market Index to understand "Market Mood"
MARKET_INDEX = "^NSEI" 

REPO_ID = "Shakuji/stock-predictor-model" 
MODEL_FILE = "model.pkl"

# --- HELPER: Advanced Pattern Recognition ---
def add_patterns(df):
    # Base measurements
    body_size = abs(df['Close'] - df['Open'])
    full_range = df['High'] - df['Low']
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # 1. Doji (Indecision)
    df['Is_Doji'] = (body_size <= (full_range * 0.1)).astype(int)
    
    # 2. Hammer (Bullish Reversal)
    df['Is_Hammer'] = (
        (lower_shadow >= (body_size * 2)) & 
        (upper_shadow <= (body_size * 0.5))
    ).astype(int)
    
    # 3. Shooting Star (Bearish Reversal - Inverted Hammer at top)
    df['Is_ShootingStar'] = (
        (upper_shadow >= (body_size * 2)) &
        (lower_shadow <= (body_size * 0.5))
    ).astype(int)
    
    # 4. Bullish Engulfing (Strong Buy)
    prev_open = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)
    df['Is_Engulfing'] = (
        (df['Close'] > df['Open']) & 
        (prev_close < prev_open) & 
        (df['Close'] > prev_open) & 
        (df['Open'] < prev_close)
    ).astype(int)

    # 5. Marubozu (Strong Momentum - No shadows)
    df['Is_Marubozu'] = (
        (body_size > (full_range * 0.9)) # Body is 90% of the candle
    ).astype(int)
    
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_stock_data(symbol, market_df=None):
    print(f"📥 Fetching {symbol}...")
    stock = yf.Ticker(symbol)
    df = stock.history(period="2y")
    
    if len(df) < 200: return None
    
    # --- TECHNICAL INDICATORS ---
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["RSI"] = calculate_rsi(df["Close"])
    df["Volatility"] = df["Close"].rolling(window=10).std()
    df["Return"] = df["Close"].pct_change()
    
    # --- MARKET CONTEXT (The "Nifty Filter") ---
    # We join the Nifty 50 return to the stock data
    if market_df is not None:
        df = df.join(market_df, rsuffix="_Market")
        # Fill missing market days with 0 change
        df["Return_Market"] = df["Return_Market"].fillna(0)
    else:
        # If Nifty fetch failed, assume neutral market
        df["Return_Market"] = 0
    
    # --- PATTERN RECOGNITION ---
    df = add_patterns(df)
    
    # Target: 1 if Price UP tomorrow
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()
    return df

def train():
    # 1. Fetch Market Context First
    print(f"📊 Fetching Market Index ({MARKET_INDEX})...")
    nifty = yf.Ticker(MARKET_INDEX)
    nifty_df = nifty.history(period="2y")
    nifty_df["Return"] = nifty_df["Close"].pct_change()
    # We only need the market return column
    market_signal = nifty_df[["Return"]]
    
    all_data = []
    print("🚀 Starting Pro-Tier Training...")
    
    for ticker in TRAINING_BASKET:
        df = get_stock_data(ticker, market_signal)
        if df is not None:
            all_data.append(df)
            
    full_df = pd.concat(all_data)
    print(f"✅ Training on {len(full_df)} rows of data.")
    
    # The Full Feature List
    features = [
        "SMA_10", "SMA_50", "RSI", "Volatility", "Return", 
        "Return_Market", # <--- Nifty Impact
        "Is_Doji", "Is_Hammer", "Is_ShootingStar", "Is_Engulfing", "Is_Marubozu"
    ]
    
    X = full_df[features]
    y = full_df["Target"]
    
    # Deeper Trees for complex patterns
    model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=7, random_state=42)
    model.fit(X, y)
    
    acc = accuracy_score(y.iloc[-1000:], model.predict(X.iloc[-1000:]))
    print(f"🏆 Final Model Accuracy: {acc:.2%}")
    
    joblib.dump(model, MODEL_FILE)
    
    api = HfApi()
    api.upload_file(
        path_or_fileobj=MODEL_FILE,
        path_in_repo=MODEL_FILE,
        repo_id=REPO_ID,
        repo_type="model",
        token=os.getenv("HF_TOKEN")
    )
    print("🎉 Upload Success!")

if __name__ == "__main__":
    train()
