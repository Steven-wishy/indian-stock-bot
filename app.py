from flask import Flask, request, render_template_string
import yfinance as yf
import joblib
import json
import pandas as pd
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# --- CONFIGURATION ---
REPO_ID = "Shakuji/stock-predictor-model" 
MODEL_FILE = "model.pkl" 

def get_prediction_and_data(ticker):
    try:
        # 1. Download Model (Public)
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
        model = joblib.load(model_path)
        
        # 2. Get Data
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo") 
        
        # Error Handling
        if df.empty: 
            return None, f"Could not find data for '{ticker}'. <br>Hint: For Oracle India, use 'OFSS'. For Tata, use 'TATASTEEL'.", None, None
        
        if len(df) < 60: 
            return None, "Insufficient data (Need 60+ days of history)", None, None

        # 3. Prepare Graph Data
        dates = df.index.strftime('%d %b').tolist()[-90:] 
        prices = df['Close'].round(2).tolist()[-90:]
        
        # 4. Calculate Indicators
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["Return"] = df["Close"].pct_change()
        
        latest = df.iloc[-1:].copy()
        features = ["SMA_10", "SMA_50", "Return", "Open", "High", "Low", "Close"]
        
        # --- THE FIX: Create Named DataFrame ---
        input_df = pd.DataFrame(latest[features].values, columns=features)
        
        # 5. Predict
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][pred]
        
        direction = "BULLISH" if pred == 1 else "BEARISH"
        return direction, prob, dates, prices
        
    except Exception as e:
        err_msg = str(e)
        if "feature names" in err_msg:
            return None, "Model Mismatch: Please run 'Daily Retrain' in GitHub Actions.", None, None
        return None, f"System Error: {err_msg}", None, None

# --- UI TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Oracle /// Market Intelligence</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #050505; --surface: #121212; --border: #333333; --text-main: #ffffff; --text-muted: #888888; --accent: #FFFFFF; --green: #34D399; --red: #F87171; --font: 'Inter', sans-serif; }
        * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
        body { background-color: var(--bg); color: var(--text-main); font-family: var(--font); margin: 0; display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
        .app-container { width: 100%; max-width: 480px; padding: 24px; display: flex; flex-direction: column; gap: 32px; }
        header { margin-top: 20px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }
        h1 { font-weight: 300; font-size: 24px; letter-spacing: -0.5px; margin: 0; color: var(--text-muted); }
        h1 span { color: var(--text-main); font-weight: 600; }
        .input-group { position: relative; width: 100%; }
        input { width: 100%; background: transparent; border: none; border-bottom: 1px solid var(--border); color: var(--text-main); font-family: var(--font); font-size: 32px; padding: 10px 0; outline: none; transition: border 0.3s; border-radius: 0; }
        input:focus { border-bottom-color: var(--accent); }
        input::placeholder { color: #333; }
        button { margin-top: 20px; width: 100%; padding: 16px; background: var(--text-main); color: var(--bg); border: none; font-family: var(--font); font-weight: 600; font-size: 14px; letter-spacing: 0.5px; text-transform: uppercase; cursor: pointer; transition: opacity 0.2s; }
        button:active { opacity: 0.8; }
        .result-panel { animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1); }
        .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
        .metric { background: var(--surface); padding: 16px; border: 1px solid var(--border); }
        .label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 8px; display: block; }
        .value { font-size: 18px; font-weight: 400; }
        .bullish { color: var(--green); }
        .bearish { color: var(--red); }
        .chart-container { position: relative; height: 250px; width: 100%; margin-top: 20px; }
        .loader { height: 2px; width: 100%; background: #333; position: relative; overflow: hidden; display: none; margin-top: 20px; }
        .loader::after { content: ''; position: absolute; left: -50%; height: 100%; width: 50%; background: var(--text-main); animation: load 1s infinite ease-in-out; }
        .error-box { color: var(--red); margin-top: 20px; font-size: 14px; line-height: 1.5; background: rgba(248, 113, 113, 0.1); padding: 15px; border-left: 2px solid var(--red); }
        @keyframes load { 0% { left: -50%; } 100% { left: 100%; } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>
    <div class="app-container">
        <header><h1>ORACLE <span>/// AI</span></h1></header>
        <form method="POST" onsubmit="document.querySelector('.loader').style.display = 'block'">
            <div class="input-group">
                <input type="text" name="ticker" placeholder="TICKER" value="{{ ticker }}" autocomplete="off" autocorrect="off">
            </div>
            <div class="loader"></div>
            <button type="submit">Analyze Asset</button>
        </form>
        {% if error %}
            <div class="error-box">/// ERROR: {{ error | safe }}</div>
        {% endif %}
        {% if prediction %}
        <div class="result-panel">
            <div class="metric-grid">
                <div class="metric"><span class="label">Forecast</span><span class="value {{ 'bullish' if prediction == 'BULLISH' else 'bearish' }}">{{ prediction }}</span></div>
                <div class="metric"><span class="label">Confidence</span><span class="value">{{ confidence }}%</span></div>
            </div>
            <span class="label">Price Action (90 Days)</span>
            <div class="chart-container"><canvas id="stockChart"></canvas></div>
        </div>
        <script>
            const dates = {{ dates | safe }};
            const prices = {{ prices | safe }};
            const direction = "{{ prediction }}";
            const mainColor = direction === 'BULLISH' ? '#34D399' : '#F87171';
            const ctx = document.getElementById('stockChart').getContext('2d');
            let gradient = ctx.createLinearGradient(0, 0, 0, 250);
            gradient.addColorStop(0, mainColor + '40');
            gradient.addColorStop(1, 'rgba(0,0,0,0)');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [{
                        data: prices,
                        borderColor: mainColor,
                        backgroundColor: gradient,
                        borderWidth: 1.5,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        fill: true,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
                    scales: { x: { display: false }, y: { display: true, position: 'right', grid: { color: '#333', borderDash: [2, 4] }, ticks: { color: '#666', font: { family: 'Inter', size: 10 } } } },
                    interaction: { mode: 'nearest', axis: 'x', intersect: false }
                }
            });
        </script>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = 0
    error = None
    ticker_val = ""
    dates = []
    prices = []

    if request.method == "POST":
        raw_input = request.form.get("ticker", "").strip()
        if not raw_input:
            error = "Please enter a stock symbol (e.g., ZOMATO)"
        else:
            ticker_val = raw_input.upper()
            if not ticker_val.endswith(".NS") and not ticker_val.endswith(".BO"):
                ticker_val += ".NS"
            
            pred, conf_or_err, hist_dates, hist_prices = get_prediction_and_data(ticker_val)
            
            if pred:
                prediction = pred
                confidence = round(conf_or_err * 100, 1)
                dates = json.dumps(hist_dates)
                prices = json.dumps(hist_prices)
            else:
                error = conf_or_err

    return render_template_string(HTML_TEMPLATE, 
                                  prediction=prediction, 
                                  confidence=confidence, 
                                  dates=dates, 
                                  prices=prices, 
                                  error=error, 
                                  ticker=ticker_val.replace(".NS",""))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
