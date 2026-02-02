from flask import Flask, request
import yfinance as yf
import joblib
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# CHANGE THIS
REPO_ID = "Shakuji/stock-predictor-model"
MODEL_FILE = "model.pkl"

def get_prediction(ticker):
    # Download latest brain
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
    model = joblib.load(model_path)

    # Get Data
    stock = yf.Ticker(ticker)
    df = stock.history(period="3mo")
    if len(df) < 60: return "Not enough data", 0

    # Indicators
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["Return"] = df["Close"].pct_change()

    latest = df.iloc[-1:].copy()
    features = ["SMA_10", "SMA_50", "Return", "Open", "High", "Low", "Close"]

    # Predict
    pred = model.predict(latest[features])[0]
    prob = model.predict_proba(latest[features])[0][pred]
    return ("UP 🚀" if pred == 1 else "DOWN 🔻"), prob

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    ticker = ""
    if request.method == "POST":
        ticker = request.form.get("ticker").upper()
        if not ticker.endswith(".NS"): ticker += ".NS"
        try:
            direction, conf = get_prediction(ticker)
            result = f"<h2>{ticker}: {direction}</h2><p>Confidence: {conf*100:.1f}%</p>"
        except Exception as e:
            result = f"<p>Error: {str(e)}</p>"

    return f"""
    <div style="text-align:center; padding-top:50px;">
        <h1>🇮🇳 Stock AI</h1>
        <form method="POST">
            <input type="text" name="ticker" value="{ticker.replace('.NS','')}" placeholder="TATASTEEL">
            <button>Predict</button>
        </form>
        {result}
    </div>
    """

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
