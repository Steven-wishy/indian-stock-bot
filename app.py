from flask import Flask, request, render_template_string, redirect, url_for
import yfinance as yf
import joblib
import json
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi
import numpy as np
import os
import time
from datetime import datetime, timedelta
import pytz 

app = Flask(__name__)

# --- CONFIGURATION ---
REPO_ID = "Shakuji/stock-predictor-model" 
MARKET_INDEX = "^NSEI"
HISTORY_FILE = "history.json"
HF_TOKEN = os.getenv("HF_TOKEN")

# --- 1. MATH & INDICATOR LOGIC (FULL SUITE) ---
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
    
    # 4. Standard Moving Averages & Volatility
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()

    # 5. Memory Lags (CRITICAL FOR V8)
    for lag in range(1, 6):
        df[f'Return_Lag_{lag}'] = df['Close'].pct_change(lag)
        
    return df

def generate_reasoning(row):
    reasons = []
    if row['RSI'] > 70: reasons.append("⚠️ <b>RSI Overbought:</b> Risk of Pullback.")
    elif row['RSI'] < 30: reasons.append("💎 <b>RSI Oversold:</b> Potential Value Zone.")
    else: reasons.append("✅ <b>RSI Neutral:</b> Safe zone.")
    
    if row['MACD'] > row['MACD_Signal']: reasons.append("🚀 <b>MACD Bullish:</b> Momentum is Positive.")
    else: reasons.append("📉 <b>MACD Bearish:</b> Momentum is Negative.")
    
    if row['Close'] > row['BB_Upper']: reasons.append("⚠️ <b>Bollinger High:</b> Price extended.")
    elif row['Close'] < row['BB_Lower']: reasons.append("🛡️ <b>Bollinger Low:</b> Price at support.")

    return "<br>".join(reasons)

def calculate_position_size(wallet, risk_per_trade, price, atr):
    risk_amount = wallet * (risk_per_trade / 100)
    sl_distance = 2 * atr
    if sl_distance == 0: return 0, 0, 0
    qty = int(risk_amount / sl_distance)
    max_alloc = wallet * 0.25
    if (qty * price) > max_alloc: qty = int(max_alloc / price)
    capital = qty * price
    return qty, capital, sl_distance

# --- 2. CLOUD HISTORY & VERIFICATION SYSTEM ---

def load_history():
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=HISTORY_FILE, repo_type="model")
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return []

def save_history_to_cloud(history_data):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history_data, f, indent=4)
    if HF_TOKEN:
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=HISTORY_FILE,
                path_in_repo=HISTORY_FILE,
                repo_id=REPO_ID,
                repo_type="model",
                token=HF_TOKEN
            )
        except Exception as e:
            print(f"Cloud Upload Failed: {e}")

def verify_predictions(history):
    updated = False
    
    # Set Timezone to IST (Indian Standard Time)
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(ist)
    
    CHECK_FILE = "last_check.txt"
    LOCKOUT_FILE = "lockout.txt"
    COOLDOWN_MINUTES = 2 
    LOCKOUT_DURATION = 60 

    # 1. EMERGENCY LOCKOUT CHECK
    if os.path.exists(LOCKOUT_FILE):
        try:
            with open(LOCKOUT_FILE, 'r') as f:
                lockout_time = datetime.strptime(f.read().strip(), '%Y-%m-%d %H:%M:%S')
                if (datetime.now() - lockout_time).total_seconds() / 60 < LOCKOUT_DURATION:
                    return history
        except: pass

    # 2. COOLDOWN CHECK
    if os.path.exists(CHECK_FILE):
        try:
            with open(CHECK_FILE, 'r') as f:
                last_time = datetime.strptime(f.read().strip(), '%Y-%m-%d %H:%M:%S')
                if (datetime.now() - last_time).total_seconds() / 60 < COOLDOWN_MINUTES:
                    return history
        except: pass

    # 3. VERIFICATION LOGIC
    check_count = 0
    
    for entry in history:
        if check_count >= 2: break
        
        if entry.get('Status') == "Pending":
            try:
                pred_date = datetime.strptime(entry['Date'], '%Y-%m-%d')
                days_passed = (now_ist.replace(tzinfo=None) - pred_date).days
                
                # Determine Target Date
                required_days = 1
                if entry['Horizon'] == '1w': required_days = 5
                elif entry['Horizon'] == '1m': required_days = 20
                
                # PROFESSIONAL CHECK:
                is_target_day = (days_passed == required_days)
                market_closed = (now_ist.hour >= 16)

                if days_passed > required_days or (is_target_day and market_closed):
                    
                    time.sleep(1) # Polite Delay
                    
                    ticker = entry['Ticker']
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1mo") 
                    
                    if not hist.empty:
                        result_price = hist.iloc[-1]['Close']
                        entry['Exit_Price'] = round(result_price, 2)
                        
                        if entry['Direction'] == "BULLISH":
                            if result_price > entry['Entry_Price']: entry['Status'] = "✅ WIN"
                            else: entry['Status'] = "❌ LOSS"
                        elif entry['Direction'] == "BEARISH":
                            if result_price < entry['Entry_Price']: entry['Status'] = "✅ WIN"
                            else: entry['Status'] = "❌ LOSS"
                        else:
                            entry['Status'] = "⚪ NEUTRAL"
                        updated = True
                        check_count += 1
                        
            except Exception as e:
                error_msg = str(e)
                if "Too Many Requests" in error_msg or "429" in error_msg:
                    with open(LOCKOUT_FILE, 'w') as f:
                        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    break
                break

    # 4. RESET TIMER
    with open(CHECK_FILE, 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    if updated:
        save_history_to_cloud(history)
    return history

def log_prediction(ticker, horizon, direction, prob, price, target, stop_loss):
    history = load_history()
    today = datetime.now().strftime('%Y-%m-%d')
    
    for entry in history:
        if entry['Date'] == today and entry['Ticker'] == ticker and entry['Horizon'] == horizon:
            return 
            
    new_entry = {
        'Date': today,
        'Ticker': ticker,
        'Horizon': horizon,
        'Direction': direction,
        'Confidence': f"{prob}%",
        'Entry_Price': price,
        'Exit_Price': "-",
        'Target': target,
        'Stop_Loss': stop_loss,
        'Status': "Pending" 
    }
    history.insert(0, new_entry)
    save_history_to_cloud(history)

# --- 3. FLASK ROUTES ---

@app.route("/", methods=["GET", "POST"])
def home():
    ticker = "RELIANCE.NS"
    horizon = "1d"
    wallet = "100000"
    result = None
    
    # Run Smart Referee
    history = load_history()
    history = verify_predictions(history) 

    if request.method == "POST":
        ticker = request.form.get("ticker", "RELIANCE.NS").upper()
        if not ticker.endswith(".NS") and not ticker.endswith(".BO") and "-" not in ticker:
            ticker += ".NS"
            
        wallet = request.form.get("wallet", "100000")
        horizon = request.form.get("horizon", "1d")
        
        result = get_prediction_and_data(ticker, horizon, "3mo", float(wallet), 2.0)
        history = load_history() 

    return render_template_string(HTML_TEMPLATE, ticker=ticker.replace(".NS",""), horizon=horizon, wallet=wallet, result=result, history=history)

@app.route("/delete", methods=["POST"])
def delete_entries():
    items_to_delete = request.form.getlist("selected_logs")
    if items_to_delete:
        history = load_history()
        new_history = [
            entry for entry in history 
            if f"{entry['Date']}|{entry['Ticker']}|{entry['Horizon']}" not in items_to_delete
        ]
        save_history_to_cloud(new_history)
    return redirect(url_for('home'))

# --- 4. PREDICTION ENGINE ---
def get_prediction_and_data(ticker, horizon="1d", graph_range="3mo", wallet=100000, risk=2):
    try:
        model_filename = f"model_{horizon}.pkl"
        model_path = hf_hub_download(repo_id=REPO_ID, filename=model_filename)
        model = joblib.load(model_path)
        
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y") 
        if df.empty or len(df) < 60: return None

        graph_days = 90
        if graph_range == "1mo": graph_days = 30
        elif graph_range == "6mo": graph_days = 180
        elif graph_range == "1y": graph_days = 365
        
        # Calculate Full Suite of Indicators
        df["RSI"] = calculate_rsi(df["Close"])
        df = add_features(df)
        latest = df.iloc[-1:].copy()
        
        # --- FEATURE SELECTION FOR TRUE V8 MODEL ---
        # This list MUST match train_model.py
        features = ["SMA_10", "SMA_50", "RSI", "Volatility", "MACD", "MACD_Signal", 
                    "BB_Upper", "BB_Lower", "ATR", "Is_Doji", "Is_Hammer"]
        features += [f'Return_Lag_{i}' for i in range(1, 6)]
        
        input_df = pd.DataFrame(latest[features].values, columns=features)
        
        # PROBABILITY SCORE (Voting Classifier)
        prob_score = model.predict_proba(input_df)[0][1]

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
        current_price = latest['Close'].values[0]
        current_atr = latest['ATR'].values[0]
        qty, capital, sl_dist = calculate_position_size(float(wallet), float(risk), current_price, current_atr)
        
        if direction == "BULLISH":
            sl_price = current_price - sl_dist
            target_price = current_price + (sl_dist * 1.5)
        else:
            sl_price = current_price + sl_dist
            target_price = current_price - (sl_dist * 1.5)
            
        log_prediction(ticker, horizon, direction, round(final_prob, 1), round(current_price, 2), round(target_price, 2), round(sl_price, 2))

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

# --- 5. HTML FRONTEND ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oracle V8</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #050505; --surface: #121212; --border: #333; --text: #fff; --text-muted: #888; --green: #34D399; --red: #F87171; --yellow: #FBBF24; --blue: #3B82F6; }
        body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        .app { width: 100%; max-width: 650px; display: flex; flex-direction: column; gap: 20px; }
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
        
        /* History & Filter Styles */
        .history-section { margin-top: 40px; }
        .history-controls { display: flex; gap: 10px; margin-bottom: 15px; flex-wrap: wrap; align-items: center;}
        .history-controls input, .history-controls select { padding: 8px; font-size: 12px; flex: 1; min-width: 100px; }
        .del-btn { background: var(--red); color: #fff; border: none; padding: 8px 15px; cursor: pointer; font-weight: 600; font-size: 12px; flex: 0 0 auto; min-width: auto; }
        
        .history-table { width: 100%; border-collapse: collapse; font-size: 12px; }
        .history-table th { text-align: left; color: var(--text-muted); padding: 8px; border-bottom: 1px solid var(--border); cursor:pointer; }
        .history-table td { padding: 8px; border-bottom: 1px solid #222; }
        .status-win { color: var(--green); font-weight: bold; }
        .status-loss { color: var(--red); font-weight: bold; }
        .status-pending { color: var(--text-muted); font-style: italic; }

        @keyframes slideUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>
    <div class="app">
        <div class="header">
            <h1>ORACLE <span>/// V8</span></h1>
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
            const lastPrice = {{ result.last_price }};
            const atr = {{ result.atr }};
            
            let futureLabels = [];
            let futureData = [];
            let moveFactor = direction === 'BULLISH' ? 1 : (direction === 'BEARISH' ? -1 : 0);
            
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
                        { label: 'Price', data: histData, borderColor: mainColor, borderWidth: 2, pointRadius: 0, tension: 0.1 },
                        { label: 'Projection', data: projData, borderColor: '#3B82F6', borderWidth: 2, borderDash: [5,5], pointRadius: 0 }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false }, annotation: { annotations: { line1: { type: 'line', yMin: stopLoss, yMax: stopLoss, borderColor: '#F87171', borderWidth: 1, borderDash: [2,2], label: { content: 'STOP', enabled: true, color: '#F87171', position: 'start' } } } } },
                    scales: { x: { display: false }, y: { position: 'right', grid: { color: '#333' } } }
                }
            });
        </script>
        {% elif result.error %}
            <div class="error">System Error: {{ result.error }}</div>
        {% endif %}

        <div class="history-section">
            <div style="font-size:14px; font-weight:600; margin-bottom:10px; color:#888;">LOG MANAGER</div>
            
            <form action="/delete" method="POST">
                <div class="history-controls">
                    <input type="text" id="searchBox" onkeyup="filterHistory()" placeholder="Search Ticker...">
                    <select id="statusFilter" onchange="filterHistory()">
                        <option value="all">All Status</option>
                        <option value="WIN">Wins Only</option>
                        <option value="LOSS">Losses Only</option>
                        <option value="Pending">Pending</option>
                    </select>
                    <input type="date" id="startDate" onchange="filterHistory()" placeholder="From Date">
                    <input type="date" id="endDate" onchange="filterHistory()" placeholder="To Date">
                    
                    <button type="submit" class="del-btn" onclick="return confirm('Delete selected logs?')">DELETE SELECTED</button>
                </div>

                <table class="history-table" id="historyTable">
                    <thead>
                        <tr>
                            <th width="30">✔</th>
                            <th onclick="sortTable(1)">Date ⬍</th>
                            <th onclick="sortTable(2)">Ticker ⬍</th>
                            <th>Signal</th>
                            <th>Entry</th>
                            <th>Exit</th>
                            <th onclick="sortTable(6)">Result ⬍</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in history %}
                        <tr class="history-row">
                            <td><input type="checkbox" name="selected_logs" value="{{ row.Date }}|{{ row.Ticker }}|{{ row.Horizon }}"></td>
                            <td class="date-cell" style="color:#666;">{{ row.Date }}</td>
                            <td class="ticker-cell">{{ row.Ticker }}</td>
                            <td class="{{ 'bullish' if row.Direction == 'BULLISH' else 'bearish' if row.Direction == 'BEARISH' else 'neutral' }}">{{ row.Direction }}</td>
                            <td>{{ row.Entry_Price }}</td>
                            <td>{{ row.Exit_Price }}</td>
                            <td class="status-cell {{ 'status-win' if 'WIN' in row.Status else 'status-loss' if 'LOSS' in row.Status else 'status-pending' }}">{{ row.Status }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </form>
        </div>
    </div>

    <script>
        function filterHistory() {
            var input = document.getElementById("searchBox");
            var filter = input.value.toUpperCase();
            var statusFilter = document.getElementById("statusFilter").value;
            
            // DATE FILTER LOGIC
            var startDateVal = document.getElementById("startDate").value;
            var endDateVal = document.getElementById("endDate").value;
            var startDate = startDateVal ? new Date(startDateVal) : null;
            var endDate = endDateVal ? new Date(endDateVal) : null;

            var table = document.getElementById("historyTable");
            var tr = table.getElementsByTagName("tr");

            for (var i = 1; i < tr.length; i++) {
                var dateTd = tr[i].getElementsByClassName("date-cell")[0];
                var tickerTd = tr[i].getElementsByClassName("ticker-cell")[0];
                var statusTd = tr[i].getElementsByClassName("status-cell")[0];
                
                if (dateTd && tickerTd && statusTd) {
                    var dateValue = new Date(dateTd.textContent || dateTd.innerText);
                    var txtValue = tickerTd.textContent || tickerTd.innerText;
                    var statusValue = statusTd.textContent || statusTd.innerText;
                    
                    var matchesSearch = txtValue.toUpperCase().indexOf(filter) > -1;
                    var matchesStatus = (statusFilter === "all") || (statusValue.indexOf(statusFilter) > -1);
                    
                    var matchesDate = true;
                    if (startDate) { if (dateValue < startDate) matchesDate = false; }
                    if (endDate) { if (dateValue > endDate) matchesDate = false; }
                    
                    if (matchesSearch && matchesStatus && matchesDate) {
                        tr[i].style.display = "";
                    } else {
                        tr[i].style.display = "none";
                    }
                }       
            }
        }
        function sortTable(n) {
          var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
          table = document.getElementById("historyTable");
          switching = true;
          dir = "asc"; 
          while (switching) {
            switching = false;
            rows = table.rows;
            for (i = 1; i < (rows.length - 1); i++) {
              shouldSwitch = false;
              x = rows[i].getElementsByTagName("TD")[n];
              y = rows[i + 1].getElementsByTagName("TD")[n];
              if (dir == "asc") {
                if (x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()) { shouldSwitch = true; break; }
              } else if (dir == "desc") {
                if (x.innerHTML.toLowerCase() < y.innerHTML.toLowerCase()) { shouldSwitch = true; break; }
              }
            }
            if (shouldSwitch) {
              rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
              switching = true;
              switchcount ++;      
            } else {
              if (switchcount == 0 && dir == "asc") { dir = "desc"; switching = true; }
            }
          }
        }
    </script>
</body>
</html>
"""
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
