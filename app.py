from flask import Flask, request, render_template_string
import yfinance as yf
import joblib
import json
import pandas as pd
from huggingface_hub import hf_hub_download
import numpy as np

app = Flask(__name__)

REPO_ID = "Shakuji/stock-predictor-model" 
MARKET_INDEX = "^NSEI"

# --- MATH & FEATURE LOGIC ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_features(df):
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma_20 + (std_20 * 2)
    df['BB_Lower'] = sma_20 - (std_20 * 2)
    
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(window=14).mean()

    body_size = abs(df['Close'] - df['Open'])
    full_range = df['High'] - df['Low']
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    df['Is_Doji'] = (body_size <= (full_range * 0.1)).astype(int)
    df['Is_Hammer'] = ((lower_shadow >= (body_size * 2)) & (upper_shadow <= (body_size * 0.5))).astype(int)

    for lag in range(1, 6):
        df[f'Return_Lag_{lag}'] = df['Close'].pct_change(lag)
    return df

def generate_reasoning(row):
    reasons = []
    # RSI
    if row['RSI'] > 70: reasons.append("⚠️ <b>RSI Overbought:</b> Risk of Pullback.")
    elif row['RSI'] < 30: reasons.append("💎 <b>RSI Oversold:</b> Potential Value Zone.")
    else: reasons.append("✅ <b>RSI Neutral:</b> Safe zone.")
    
    # MACD
    if row['MACD'] > row['MACD_Signal']: reasons.append("🚀 <b>MACD Bullish:</b> Momentum is Positive.")
    else: reasons.append("📉 <b>MACD Bearish:</b> Momentum is Negative.")
    
    # Bollinger
    if row['Close'] > row['BB_Upper']: reasons.append("⚠️ <b>Bollinger High:</b> Price extended above bands.")
    elif row['Close'] < row['BB_Lower']: reasons.append("🛡️ <b>Bollinger Low:</b> Price at support band.")

    return "<br>".join(reasons)

def calculate_position_size(wallet, risk_per_trade, price, atr):
    # Professional Risk Formula: Risk / (2 * ATR)
    risk_amount = wallet * (risk_per_trade / 100)
    sl_distance = 2 * atr
    
    if sl_distance == 0: return 0, 0, 0
    
    qty = int(risk_amount / sl_distance)
    
    # Safety Cap: Max 25% of wallet on one stock
    max_alloc = wallet * 0.25
    if (qty * price) > max_alloc:
        qty = int(max_alloc / price)
        
    capital = qty * price
    return qty, capital, sl_distance

def get_prediction_and_data(ticker, horizon="1d", graph_range="3mo", wallet=100000, risk=2):
    try:
        model_filename = f"model_{horizon}.pkl"
        model_path = hf_hub_download(repo_id=REPO_ID, filename=model_filename)
        model = joblib.load(model_path)
        
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y") 
        if df.empty or len(df) < 60: return None

        # Filter Graph Data
        graph_days = 90
        if graph_range == "1mo": graph_days = 30
        elif graph_range == "6mo": graph_days = 180
        elif graph_range == "1y": graph_days = 365
        
        # Calculate Features
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["RSI"] = calculate_rsi(df["Close"])
        df["Volatility"] = df["Close"].rolling(window=10).std()
        df = add_features(df)
        
        latest = df.iloc[-1:].copy()
        
        # Predict
        features = ["SMA_10", "SMA_50", "RSI", "Volatility", "MACD", "MACD_Signal", 
                    "BB_Upper", "BB_Lower", "ATR", "Is_Doji", "Is_Hammer"]
        features += [f'Return_Lag_{i}' for i in range(1, 6)]
        
        input_df = pd.DataFrame(latest[features].values, columns=features)
        prob_score = model.predict_proba(input_df)[0][1]

        # Traffic Light Logic
        if 0.45 <= prob_score <= 0.55:
            direction = "NEUTRAL"
            term = "Wait / Unsure"
            final_prob = 50.0
        elif prob_score > 0.55:
            direction = "BULLISH"
            term = "Likely UP"
            final_prob = prob_score * 100
        else:
            direction = "BEARISH"
            term = "Likely DOWN"
            final_prob = (1 - prob_score) * 100
            
        reasoning = generate_reasoning(latest.iloc[0])

        # Portfolio Calc
        current_price = latest['Close'].values[0]
        current_atr = latest['ATR'].values[0]
        qty, capital, sl_dist = calculate_position_size(float(wallet), float(risk), current_price, current_atr)
        
        # Set Targets
        if direction == "BULLISH":
            sl_price = current_price - sl_dist
            target_price = current_price + (sl_dist * 1.5)
        else:
            sl_price = current_price + sl_dist
            target_price = current_price - (sl_dist * 1.5)
            
        dates = df.index.strftime('%d %b').tolist()[-graph_days:] 
        prices = df['Close'].round(2).tolist()[-graph_days:]

        return {
            "direction": direction,
            "prob": round(final_prob, 1),
            "dates": json.dumps(dates),
            "prices": json.dumps(prices),
            "term": term,
            "reasoning": reasoning,
            "atr": round(current_atr, 2),
            "last_price": round(current_price, 2),
            "qty": qty,
            "capital": round(capital, 2),
            "wallet_pct": round((capital / float(wallet)) * 100, 1),
            "stop_loss": round(sl_price, 2),
            "target": round(target_price, 2)
        }
        
    except Exception as e:
        return {"error": str(e)}

# --- UI TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oracle V3 /// Portfolio Architect</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #050505; --surface: #121212; --border: #333; --text: #fff; --text-muted: #888; --green: #34D399; --red: #F87171; --yellow: #FBBF24; --blue: #3B82F6; }
        body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        .app { width: 100%; max-width: 600px; display: flex; flex-direction: column; gap: 20px; }
        .header { border-bottom: 1px solid var(--border); padding-bottom: 15px; }
        h1 { font-weight: 300; margin: 0; font-size: 24px; } h1 span { font-weight: 600; }
        
        .controls { display: grid; gap: 10px; background: var(--surface); padding: 15px; border: 1px solid var(--border); }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        input, button, select { background: #000; border: 1px solid var(--border); color: #fff; padding: 10px; width: 100%; box-sizing: border-box; }
        button { background: #fff; color: #000; font-weight: 600; cursor: pointer; border: none; }
        
        .result-card { background: var(--surface); border: 1px solid var(--border); padding: 20px; animation: slideUp 0.5s ease; }
        .metric-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #222; padding-bottom: 15px; }
        .big-val { font-size: 28px; font-weight: 600; }
        .sub-val { font-size: 13px; color: var(--text-muted); }
        .bullish { color: var(--green); } .bearish { color: var(--red); } .neutral { color: var(--yellow); }

        .portfolio-box { background: rgba(59, 130, 246, 0.1); border: 1px solid var(--blue); padding: 15px; margin-bottom: 20px; }
        .plan-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; text-align: center; margin-top: 10px; }
        .plan-item div:first-child { font-size: 11px; color: var(--text-muted); }
        .plan-item div:last-child { font-size: 16px; font-weight: 600; }

        .chart-box { height: 250px; position: relative; margin-top: 20px; }
        .error { color: var(--red); padding: 10px; background: rgba(248,113,113,0.1); }
        @keyframes slideUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>
    <div class="app">
        <div class="header">
            <h1>ORACLE <span>/// V3</span></h1>
        </div>

        <form method="POST" class="controls">
            <div style="font-size:12px; color:#888; margin-bottom:5px;">ASSET</div>
            <input type="text" name="ticker" placeholder="TICKER (e.g. RELIANCE)" value="{{ ticker }}">
            
            <div style="font-size:12px; color:#888; margin:10px 0 5px;">STRATEGY & WALLET</div>
            <div class="row">
                <select name="horizon">
                    <option value="1d" {% if horizon == '1d' %}selected{% endif %}>Tomorrow (1D)</option>
                    <option value="1w" {% if horizon == '1w' %}selected{% endif %}>Next Week (5D)</option>
                    <option value="1m" {% if horizon == '1m' %}selected{% endif %}>Next Month (20D)</option>
                </select>
                <input type="number" name="wallet" placeholder="Wallet (₹)" value="{{ wallet }}">
            </div>
            
            <button type="submit" style="margin-top:10px;">ANALYZE</button>
            <input type="hidden" name="risk" value="2">
        </form>

        {% if result and not result.error %}
        <div class="result-card">
            <div class="metric-row">
                <div>
                    <div class="sub-val">DIRECTION</div>
                    <div class="big-val {{ result.direction.lower() }}">{{ result.direction }}</div>
                    <div class="sub-val">{{ result.term }} ({{ result.prob }}%)</div>
                </div>
                <div style="text-align:right;">
                    <div class="sub-val">CURRENT PRICE</div>
                    <div class="big-val">₹{{ result.last_price }}</div>
                    <div class="sub-val">ATR: {{ result.atr }}</div>
                </div>
            </div>

            <div class="portfolio-box">
                <div style="color:var(--blue); font-size:12px; font-weight:600; letter-spacing:1px;">AI TRADE ARCHITECT</div>
                <div class="plan-grid">
                    <div class="plan-item">
                        <div>BUY QTY</div>
                        <div style="color:#fff;">{{ result.qty }}</div>
                    </div>
                    <div class="plan-item">
                        <div>CAPITAL</div>
                        <div style="color:#fff;">₹{{ result.capital }}</div>
                    </div>
                    <div class="plan-item">
                        <div>ALLOCATION</div>
                        <div style="color:#fff;">{{ result.wallet_pct }}%</div>
                    </div>
                </div>
                <div class="plan-grid" style="border-top:1px solid rgba(59,130,246,0.3); margin-top:10px; padding-top:10px;">
                    <div class="plan-item">
                        <div>STOP LOSS</div>
                        <div style="color:var(--red);">₹{{ result.stop_loss }}</div>
                    </div>
                    <div class="plan-item">
                        <div>TARGET</div>
                        <div style="color:var(--green);">₹{{ result.target }}</div>
                    </div>
                    <div class="plan-item">
                        <div>RISK/REWARD</div>
                        <div style="color:#fff;">1 : 1.5</div>
                    </div>
                </div>
            </div>

            <div class="chart-box">
                <canvas id="mainChart"></canvas>
            </div>
            
            <div style="font-size:12px; color:#666; margin-top:15px; border-top:1px solid #333; padding-top:10px;">
                {{ result.reasoning | safe }}
            </div>
        </div>
        
        <script>
            const ctx = document.getElementById('mainChart').getContext('2d');
            const dates = {{ result.dates | safe }};
            const prices = {{ result.prices | safe }};
            const stopLoss = {{ result.stop_loss }};
            const target = {{ result.target }};
            const direction = "{{ result.direction }}";
            
            // Visual Projection (Blue Dotted Line)
            const lastPrice = {{ result.last_price }};
            const atr = {{ result.atr }};
            let futureLabels = [];
            let futureData = [];
            let moveFactor = direction === 'BULLISH' ? 1 : (direction === 'BEARISH' ? -1 : 0);
            
            // Project 5 days
            for(let i=1; i<=5; i++) {
                futureLabels.push("Proj +" + i + "d");
                futureData.push(lastPrice + (moveFactor * atr * 0.5 * i));
            }
            
            const allLabels = dates.concat(futureLabels);
            const histData = prices.concat(new Array(5).fill(null));
            const projData = new Array(dates.length - 1).fill(null).concat([lastPrice]).concat(futureData);

            let mainColor = direction === 'BULLISH' ? '#34D399' : (direction === 'BEARISH' ? '#F87171' : '#FBBF24');

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: allLabels,
                    datasets: [
                        {
                            label: 'Price',
                            data: histData,
                            borderColor: mainColor,
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.1
                        },
                        {
                            label: 'Projection',
                            data: projData,
                            borderColor: '#3B82F6',
                            borderWidth: 2,
                            borderDash: [5,5],
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { 
                        annotation: {
                            annotations: {
                                line1: {
                                    type: 'line',
                                    yMin: stopLoss,
                                    yMax: stopLoss,
                                    borderColor: '#F87171',
                                    borderWidth: 1,
                                    borderDash: [2,2],
                                    label: { content: 'STOP', enabled: true, color: '#F87171', position: 'start' }
                                }
                            }
                        },
                        legend: { display: false } 
                    },
                    scales: { 
                        x: { display: false }, 
                        y: { position: 'right', grid: { color: '#333' } } 
                    }
                }
            });
        </script>
        {% elif result.error %}
            <div class="error">System Error: {{ result.error }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    ticker = "RELIANCE.NS"
    horizon = "1d"
    wallet = "100000"
    result = None

    if request.method == "POST":
        ticker = request.form.get("ticker", "RELIANCE.NS").upper()
        if not ticker.endswith(".NS") and not ticker.endswith(".BO") and "-" not in ticker:
            ticker += ".NS"
            
        wallet = request.form.get("wallet", "100000")
        horizon = request.form.get("horizon", "1d")
        
        # Risk is hardcoded to 2% (Standard Professional Risk)
        result = get_prediction_and_data(ticker, horizon, "3mo", float(wallet), 2.0)

    return render_template_string(HTML_TEMPLATE, ticker=ticker.replace(".NS",""), horizon=horizon, wallet=wallet, result=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
