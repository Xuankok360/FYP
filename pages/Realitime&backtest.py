import streamlit as st
import ccxt
import pandas as pd
import requests
import logging
import time
import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from collections import defaultdict
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------- LOGGING ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --------------------- Custom Modules for Model Loading ---------------------
class MLPModule(nn.Module):
    def __init__(self, input_dim, neurons=64, dropout_rate=0.2):
        super(MLPModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, neurons)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(neurons, neurons // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(neurons // 2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

class ReshapeNeuralNetClassifier(NeuralNetClassifier):
    pass

# Register these classes so unpickling works:
import __main__
__main__.MLPModule = MLPModule
__main__.ReshapeNeuralNetClassifier = ReshapeNeuralNetClassifier

# Let PyTorch safely load them:
torch.serialization.add_safe_globals([
    MLPModule,
    torch.nn.modules.linear.Linear,
    torch.nn.modules.dropout.Dropout,
    torch.nn.BCEWithLogitsLoss,
    torch.optim.Adam,
    defaultdict,
    dict
])

# --------------------- Streamlit Config ---------------------
st.set_page_config(
    page_title="Real-Time BTCUSDT Trade Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# --------------------- Custom CSS (Optional) ---------------------
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
        color: #333;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
    }
    div.stButton > button {
        background-color: #2ecc71;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1em;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #27ae60;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 BTCUSDT Trading Simulation")

# --------------------- Sidebar Controls ---------------------
st.sidebar.header("Settings")

simulation_mode = st.sidebar.radio("Simulation Mode", ("Real Time", "Back Test"))

if simulation_mode == "Real Time":
    selected_interval = st.sidebar.radio("Select Interval", ("1d", "4h"))
else:
    csv_folder = "Interval"
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    if csv_files:
        selected_csv = st.sidebar.selectbox("Select Data File", csv_files)
    else:
        st.sidebar.error(f"No CSV files found in {csv_folder}.")
        st.stop()

display_mode = st.sidebar.radio("Model Mode", ["Single Model", "All Models"])
MODEL_FOLDER = "SelectedModel"
available_models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pkl')]
if not available_models:
    st.sidebar.error("No .pkl files found in MODEL_FOLDER.")
    st.stop()

selected_model = None
if display_mode == "Single Model":
    selected_model = st.sidebar.selectbox("Select Model", available_models)

initial_balance = st.sidebar.number_input("Initial Balance", value=10000.0, step=1000.0)
rsi_period = st.sidebar.number_input("RSI Period", value=14, min_value=1, step=1)
sma_window = st.sidebar.number_input("SMA Window", value=20, min_value=1, step=1)
bollinger_multiplier = st.sidebar.number_input("Bollinger Multiplier", value=2.0, min_value=0.1, step=0.1)
threshold = st.sidebar.number_input("Movement Threshold (Ex: 0.005 for 0.5%)", value=0.005, step=0.001, format="%.3f")
fee_rate = st.sidebar.number_input("Trading Fee Rate (Ex: 0.001 for 0.1%)", value=0.001, step=0.0001, format="%.4f")

if simulation_mode == "Real Time":
    st.sidebar.subheader("Update Delay")
    input_days = st.sidebar.number_input("Days", min_value=0, value=0, step=1)
    input_hours = st.sidebar.number_input("Hours", min_value=0, value=0, step=1)
    default_seconds = 10 if selected_interval == "1d" else 5
    input_seconds = st.sidebar.number_input("Seconds", min_value=0, value=default_seconds, step=1)
    real_time_delay = int(input_days * 86400 + input_hours * 3600 + input_seconds)
    if "running" not in st.session_state:
        st.session_state.running = False
    if st.sidebar.button("Start Real-Time"):
        st.session_state.running = True
    if st.sidebar.button("Stop Real-Time"):
        st.session_state.running = False

# --------------------- Data Fetching Functions ---------------------
def fetch_premium_index_klines(symbol="BTCUSDT", interval="1d", limit=31, timeout=10):
    url = "https://fapi.binance.com/fapi/v1/premiumIndexKlines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error fetching premium index klines: {e}")
        return None

def fetch_ohlcv_data_daily():
    ex = ccxt.binanceusdm()
    ohlcv = ex.fetch_ohlcv('BTC/USDT', '1d', None, 31)
    df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df

def fetch_premium_index_data_daily():
    data = fetch_premium_index_klines(symbol="BTCUSDT", interval="1d", limit=31, timeout=10)
    if data is not None:
        columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
                   'close_time', 'quote_asset_volume', 'trades',
                   'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        prem_df = pd.DataFrame(data, columns=columns)
        prem_df['date'] = pd.to_datetime(prem_df['open_time'], unit='ms')
        prem_df = prem_df[['date', 'open', 'high', 'low', 'close', 'close_time', 'trades']]
        prem_df.rename(columns={
            'open': 'premium_open',
            'high': 'premium_high',
            'low': 'premium_low',
            'close': 'coinbase_premium_index',
            'close_time': 'premium_close_time',
            'trades': 'premium_trades'
        }, inplace=True)
        prem_df.set_index('date', inplace=True)
        prem_df.sort_index(inplace=True)
        return prem_df
    else:
        logging.error("Failed to fetch daily premium index data")
        return pd.DataFrame()

def fetch_open_interest_data_daily():
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {"symbol": "BTCUSDT", "period": "1d", "limit": 31}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        oi_df = pd.DataFrame(data)
        if oi_df.empty:
            return oi_df
        if 'timestamp' in oi_df.columns:
            oi_df['date'] = pd.to_datetime(oi_df['timestamp'], unit='ms')
        elif 'time' in oi_df.columns:
            oi_df['date'] = pd.to_datetime(oi_df['time'], unit='ms')
        else:
            return pd.DataFrame()
        oi_df.set_index('date', inplace=True)
        if 'openInterest' in oi_df.columns:
            oi_df.rename(columns={'openInterest': 'oi'}, inplace=True)
        elif 'sumOpenInterest' in oi_df.columns:
            oi_df.rename(columns={'sumOpenInterest': 'oi'}, inplace=True)
        else:
            possible_cols = [col for col in oi_df.columns if 'interest' in col.lower()]
            if possible_cols:
                oi_df.rename(columns={possible_cols[0]: 'oi'}, inplace=True)
            else:
                oi_df['oi'] = None
        oi_df = oi_df[['oi']]
        oi_df.sort_index(inplace=True)
        return oi_df
    except Exception as e:
        logging.error(f"Error fetching daily open interest data: {e}")
        return pd.DataFrame()

def fetch_ohlcv_data_4h():
    ex = ccxt.binanceusdm()
    ohlcv = ex.fetch_ohlcv('BTC/USDT', '4h', None, 31)
    df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df

def fetch_premium_index_data_4h():
    data = fetch_premium_index_klines(symbol="BTCUSDT", interval="4h", limit=31, timeout=10)
    if data is not None:
        columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
                   'close_time', 'quote_asset_volume', 'trades',
                   'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        prem_df = pd.DataFrame(data, columns=columns)
        prem_df['date'] = pd.to_datetime(prem_df['open_time'], unit='ms')
        prem_df = prem_df[['date', 'open', 'high', 'low', 'close', 'close_time', 'trades']]
        prem_df.rename(columns={
            'open': 'premium_open',
            'high': 'premium_high',
            'low': 'premium_low',
            'close': 'coinbase_premium_index',
            'close_time': 'premium_close_time',
            'trades': 'premium_trades'
        }, inplace=True)
        prem_df.set_index('date', inplace=True)
        prem_df.sort_index(inplace=True)
        return prem_df
    else:
        logging.error("Failed to fetch 4h premium index data")
        return pd.DataFrame()

def fetch_open_interest_data_4h():
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {"symbol": "BTCUSDT", "period": "4h", "limit": 31}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        oi_df = pd.DataFrame(data)
        if oi_df.empty:
            return oi_df
        if 'timestamp' in oi_df.columns:
            oi_df['date'] = pd.to_datetime(oi_df['timestamp'], unit='ms')
        elif 'time' in oi_df.columns:
            oi_df['date'] = pd.to_datetime(oi_df['time'], unit='ms')
        else:
            return pd.DataFrame()
        oi_df.set_index('date', inplace=True)
        if 'openInterest' in oi_df.columns:
            oi_df.rename(columns={'openInterest': 'oi'}, inplace=True)
        elif 'sumOpenInterest' in oi_df.columns:
            oi_df.rename(columns={'sumOpenInterest': 'oi'}, inplace=True)
        else:
            possible_cols = [col for col in oi_df.columns if 'interest' in col.lower()]
            if possible_cols:
                oi_df.rename(columns={possible_cols[0]: 'oi'}, inplace=True)
            else:
                oi_df['oi'] = None
        oi_df = oi_df[['oi']]
        oi_df.sort_index(inplace=True)
        return oi_df
    except Exception as e:
        logging.error(f"Error fetching 4h open interest data: {e}")
        return pd.DataFrame()

# --------------------- Helper Functions ---------------------
def calculate_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def run_simulation_lstm(df, tuned_model, rsi_period, sma_window, bollinger_multiplier,
                        threshold, fee_rate, initial_balance):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = ['coinbase_premium_index','volume','RSI','STD20','UpperBand','LowerBand','oi','open','close']

    df['RSI'] = calculate_RSI(df['close'], period=rsi_period)
    df['SMA20'] = df['close'].rolling(window=sma_window).mean()
    df['STD20'] = df['close'].rolling(window=sma_window).std()
    df['UpperBand'] = df['SMA20'] + (bollinger_multiplier * df['STD20'])
    df['LowerBand'] = df['SMA20'] - (bollinger_multiplier * df['STD20']) 

    df['future_return'] = (df['close'].shift(-1) - df['close']) / df['close']
    df['action'] = np.where(df['future_return'] > threshold, 0,
                    np.where(df['future_return'] < -threshold, 1, np.nan))
    df.dropna(subset=['action'], inplace=True)
    df.drop(df.tail(1).index, inplace=True)

    df_raw = df.copy()
    df.dropna(subset=feats, inplace=True)
    sc = StandardScaler()
    df[feats] = sc.fit_transform(df[feats])

    def create_sequences(X, lookback=10):
        X_seq = []
        indices = []
        for i in range(len(X) - lookback):
            X_seq.append(X[i:i+lookback])
            indices.append(i+lookback)
        return np.array(X_seq), np.array(indices)

    lookback = 10
    Xvals = df[feats].values
    X_seq, idx_seq = create_sequences(Xvals, lookback)
    valid = idx_seq < (len(df_raw)-1)
    X_seq = X_seq[valid]
    idx_seq = idx_seq[valid]

    trade_dates = df_raw['date'].iloc[idx_seq].reset_index(drop=True)
    entry_prices = df_raw['open'].iloc[idx_seq].reset_index(drop=True)
    exit_prices = df_raw['open'].iloc[idx_seq+1].reset_index(drop=True)

    tuned_model.eval()
    tuned_model.to(device)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = tuned_model(X_tensor)
    preds = (outputs.cpu().numpy() > 0.5).astype(int).flatten()

    balance = initial_balance
    cumulative_fee = 0
    logs = []
    for i in range(len(trade_dates) - 1):
        d0 = trade_dates.iloc[i]
        d1 = trade_dates.iloc[i+1]
        signal = preds[i]
        entry_price = entry_prices.iloc[i]
        exit_price = exit_prices.iloc[i]
        if entry_price == 0:
            continue
        trade_type = 'Long' if signal == 0 else 'Short'
        pnl = (exit_price - entry_price)/entry_price * balance if trade_type == 'Long' else \
              (entry_price - exit_price)/entry_price * balance
        fee = fee_rate * balance
        net = pnl - fee
        balance += net
        cumulative_fee += fee

        logs.append({
            'entry_date': d0,
            'exit_date': d1,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'fee': fee,
            'result': net,
            'balance': balance,
            'type': trade_type,
            'cumulative_fee': cumulative_fee
        })

    return pd.DataFrame(logs)

def run_simulation_single_model(df, model_path, selected_model,
                                rsi_period, sma_window, bollinger_multiplier,
                                threshold, fee_rate, initial_balance):
    df['RSI'] = calculate_RSI(df['close'], period=rsi_period)
    df['SMA20'] = df['close'].rolling(window=sma_window).mean()
    df['STD20'] = df['close'].rolling(window=sma_window).std()
    df['UpperBand'] = df['SMA20'] + (bollinger_multiplier * df['STD20'])
    df['LowerBand'] = df['SMA20'] - (bollinger_multiplier * df['STD20'])

    df['future_return'] = (df['close'].shift(-1) - df['close']) / df['close']
    df['action'] = np.where(df['future_return'] > threshold, 0,
                    np.where(df['future_return'] < -threshold, 1, np.nan))
    df.dropna(subset=['action'], inplace=True)
    df.drop(df.tail(1).index, inplace=True)

    if "best_lstm_model_1h" in selected_model:
        tuned_model = torch.load(model_path, map_location=torch.device("cpu"))
        return run_simulation_lstm(df, tuned_model,
                                   rsi_period, sma_window, bollinger_multiplier,
                                   threshold, fee_rate, initial_balance)

    model = joblib.load(model_path)

    if "ensemble5min" in selected_model:
        feats = ['RSI', 'SMA20', 'STD20', 'UpperBand', 'LowerBand', 'volume']
    elif "best_rf_model_1h" in selected_model:
        feats = ['coinbase_premium_index','volume','RSI','STD20','UpperBand','LowerBand']
    elif "best_rf_model_5min" in selected_model:
        feats = ['close','RSI','volume','coinbase_premium_index','LowerBand','STD20']
    else:
        feats = ['coinbase_premium_index','volume','RSI','STD20','UpperBand','LowerBand']

    df_scaled = df.copy()
    df_scaled.dropna(subset=feats, inplace=True)
    sc = StandardScaler()
    df_scaled[feats] = sc.fit_transform(df_scaled[feats])

    X = df_scaled[feats]
    if hasattr(model, 'feature_names_in_'):
        X = X[model.feature_names_in_]

    X_np = X.to_numpy().astype(np.float32)
    preds = model.predict(X_np)
    df_pred = df.loc[X.index].copy()
    df_pred['pred_signal'] = preds
    df_pred['trade_date'] = df_pred['date'].dt.date
    grouped = df_pred.groupby('trade_date')
    dates_list = sorted(grouped.groups.keys())

    balance = initial_balance
    cumulative_fee = 0
    logs = []
    for i in range(len(dates_list) - 1):
        d0 = dates_list[i]
        d1 = dates_list[i+1]
        data0 = grouped.get_group(d0)
        data1 = grouped.get_group(d1)
        entry_price = data0.iloc[0]['open']
        signal = data0.iloc[0]['pred_signal']
        exit_price = data1.iloc[0]['open']
        if entry_price == 0:
            continue
        trade_type = 'Long' if signal == 0 else 'Short'
        pnl = (exit_price - entry_price)/entry_price * balance if trade_type == 'Long' else \
              (entry_price - exit_price)/entry_price * balance
        fee = fee_rate * balance
        net = pnl - fee
        balance += net
        cumulative_fee += fee

        logs.append({
            'entry_date': d0,
            'exit_date': d1,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'fee': fee,
            'result': net,
            'balance': balance,
            'type': trade_type,
            'cumulative_fee': cumulative_fee
        })

    return pd.DataFrame(logs)

def compute_metrics_from_df(df_metric):
    if df_metric.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    df_ = df_metric.copy().sort_values("entry_date")
    df_.reset_index(drop=True, inplace=True)
    start_date = pd.to_datetime(df_["entry_date"].iloc[0])
    end_date = pd.to_datetime(df_["exit_date"].iloc[-1])
    days_held = max(1, (end_date - start_date).days)

    init_eq = df_["balance"].iloc[0]
    final_eq = df_["balance"].iloc[-1]
    eq_peak = df_["balance"].max()

    annual_ret = (final_eq / init_eq)**(365.0/days_held) - 1 if days_held > 0 else 0
    df_["Trade Return"] = df_["balance"].pct_change()
    vol = df_["Trade Return"].std() * np.sqrt(365)
    sharpe = annual_ret / vol if vol != 0 else np.nan

    df_["CumMax"] = df_["balance"].cummax()
    df_["Drawdown"] = (df_["balance"] - df_["CumMax"]) / df_["CumMax"]
    max_dd = abs(df_["Drawdown"].min()) * 100
    calmar = (annual_ret * 100) / max_dd if max_dd > 0 else np.nan
    return sharpe, max_dd, final_eq, eq_peak, calmar

# --------------------- Data Aggregation ---------------------
def get_combined_data(selected_interval):
    if selected_interval == "1d":
        ohlcv_df = fetch_ohlcv_data_daily()
        prem_df = fetch_premium_index_data_daily()
        oi_df = fetch_open_interest_data_daily()
    else:
        ohlcv_df = fetch_ohlcv_data_4h()
        prem_df = fetch_premium_index_data_4h()
        oi_df = fetch_open_interest_data_4h()

    combined_df = pd.merge(ohlcv_df, prem_df, how='outer', left_index=True, right_index=True)
    combined_df = pd.merge(combined_df, oi_df, how='outer', left_index=True, right_index=True)
    combined_df.sort_index(inplace=True)
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={"index": "date"}, inplace=True)
    return combined_df

# --------------------- Main Simulation Logic ---------------------
def run_simulation():
    csv_folder = "Interval"
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
    
    if simulation_mode == "Real Time":
        csv_filename = "combined_realtimedata_1day.csv" if selected_interval == "1d" else "combined_realtimedata_4h.csv"
        csv_path_update = os.path.join(csv_folder, csv_filename)
        combined_df = get_combined_data(selected_interval)
        st.write("### Latest Fetched Data:")
        st.dataframe(combined_df)
        if os.path.exists(csv_path_update):
            existing_df = pd.read_csv(csv_path_update, parse_dates=['date'])
            new_data = combined_df[~combined_df['date'].isin(existing_df['date'])]
            if not new_data.empty:
                combined_all = pd.concat([existing_df, new_data], ignore_index=True)
            else:
                combined_all = existing_df
            combined_all.sort_values(by='date', inplace=True)
            combined_all.reset_index(drop=True, inplace=True)
        else:
            combined_all = combined_df.copy()
        combined_all.to_csv(csv_path_update, index=False)
        combined_df = pd.read_csv(csv_path_update, parse_dates=['date'])
    else:
        csv_path_update = os.path.join(csv_folder, selected_csv)
        if os.path.exists(csv_path_update):
            combined_df = pd.read_csv(csv_path_update, parse_dates=['date'])
        else:
            combined_df = get_combined_data("1d")
            combined_df.to_csv(csv_path_update, index=False)
    
    combined_df.sort_values('date', inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    if not pd.api.types.is_datetime64_any_dtype(combined_df['date']):
        combined_df['date'] = pd.to_datetime(combined_df['date'])
    num_cols = ['open','high','low','close','volume','coinbase_premium_index','oi']
    for c in num_cols:
        if c in combined_df.columns:
            combined_df[c] = pd.to_numeric(combined_df[c], errors='coerce')
    combined_df.dropna(subset=['open','close'], inplace=True)
    combined_df.sort_values('date', inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df

def simulation_results(combined_df):
    if display_mode == "Single Model":
        st.subheader(f"Simulation: Single Model => {selected_model}")
        model_path = os.path.join(MODEL_FOLDER, selected_model)
        trade_df = run_simulation_single_model(
            df=combined_df.copy(),
            model_path=model_path,
            selected_model=selected_model,
            rsi_period=rsi_period,
            sma_window=sma_window,
            bollinger_multiplier=bollinger_multiplier,
            threshold=threshold,
            fee_rate=fee_rate,
            initial_balance=initial_balance
        )
        if trade_df.empty:
            st.warning("No trades generated. Possibly not enough data or no signals.")
        else:
            st.subheader("Trade Details")
            st.dataframe(trade_df)

            sharpe, mdd, fbal, peak_bal, calmar = compute_metrics_from_df(trade_df)
            st.write("**Final Balance:**", f"{fbal:.2f}")
            st.write("**Sharpe Ratio:**", f"{sharpe:.3f}")
            st.write("**Max Drawdown (%):**", f"{mdd:.2f}")
            st.write("**Calmar Ratio:**", f"{calmar:.3f}")

            # ---- Equity Curve Interactive Chart ----
            fig_eq = px.line(trade_df, x='entry_date', y='balance', title="Equity Curve")
            fig_eq.update_layout(xaxis_title="Date", yaxis_title="Balance")
            st.plotly_chart(fig_eq, use_container_width=True)

            # ---- BTC Price + Equity Curve with Color-coded Trade Segments ----
            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            # BTC Price trace
            fig_dual.add_trace(
                go.Scatter(x=combined_df['date'], y=combined_df['close'], name="BTC Price", line=dict(color="blue")),
                secondary_y=False)
            # Iterate through each trade segment and color by trade type
            for i in range(len(trade_df)-1):
                trade = trade_df.iloc[i]
                # Color green if Long, red if Short
                color = "green" if trade["type"].lower() == "long" else "red"
                fig_dual.add_trace(
                    go.Scatter(
                        x=[trade_df['entry_date'].iloc[i], trade_df['entry_date'].iloc[i+1]],
                        y=[trade_df['balance'].iloc[i], trade_df['balance'].iloc[i+1]],
                        mode="lines+markers",
                        line=dict(color=color),
                        name=f"{trade['type']} Trade Segment"
                    ),
                    secondary_y=True
                )
            fig_dual.update_layout(title_text="BTC Price & Equity Curve")
            fig_dual.update_xaxes(title_text="Date")
            fig_dual.update_yaxes(title_text="BTC Price", secondary_y=False)
            fig_dual.update_yaxes(title_text="Balance", secondary_y=True)
            st.plotly_chart(fig_dual, use_container_width=True)

            # ---- Calendar Heatmap of Daily PnL for Single Model ----
            # Here we group by entry_date (you can change to exit_date if preferred)
            daily_pnl = trade_df.groupby('entry_date')['result'].sum().reset_index()
            daily_pnl['date'] = pd.to_datetime(daily_pnl['entry_date'])
            daily_pnl.set_index('date', inplace=True)
            daily_pnl['day'] = daily_pnl.index.day
            daily_pnl['month'] = daily_pnl.index.month
            pivot_data = daily_pnl.pivot_table(index='month', columns='day', values='result', aggfunc='sum')
            
            # Create heatmap with annotations for each cell
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='RdYlGn'
            ))
            
            annotations = []
            for i, month in enumerate(pivot_data.index):
                for j, day in enumerate(pivot_data.columns):
                    value = pivot_data.iloc[i, j]
                    text = "" if pd.isna(value) else str(round(value, 2))
                    annotations.append(dict(
                        x=day,
                        y=month,
                        text=text,
                        showarrow=False,
                        font=dict(color="black")
                    ))
            fig_heatmap.update_layout(
                title="Calendar Heatmap of Daily PnL (Single Model)",
                xaxis_title="Day",
                yaxis_title="Month",
                annotations=annotations
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

    else:
        st.subheader("Simulation: All Models")
        equity_curves = {}
        trade_results_all = {}
        model_metrics = []
        for mod_file in available_models:
            st.write(f"Running Model: {mod_file}")
            model_path = os.path.join(MODEL_FOLDER, mod_file)
            trade_df_mod = run_simulation_single_model(
                df=combined_df.copy(),
                model_path=model_path,
                selected_model=mod_file,
                rsi_period=rsi_period,
                sma_window=sma_window,
                bollinger_multiplier=bollinger_multiplier,
                threshold=threshold,
                fee_rate=fee_rate,
                initial_balance=initial_balance
            )
            trade_results_all[mod_file] = trade_df_mod
            if not trade_df_mod.empty:
                final_ = trade_df_mod['balance'].iloc[-1]
                sharpe, mdd, fin, peak, calmar = compute_metrics_from_df(trade_df_mod)
                model_metrics.append({
                    "Model": mod_file,
                    "Sharpe Ratio": sharpe,
                    "Max Drawdown (%)": mdd,
                    "Final Balance": fin,
                    "Highest": peak,
                    "Calmar Ratio": calmar
                })
                dates_ = pd.to_datetime(trade_df_mod['entry_date'])
                eq_ = trade_df_mod['balance']
                equity_curves[mod_file] = (dates_, eq_)
        if model_metrics:
            df_metrics = pd.DataFrame(model_metrics)
            df_metrics["Sharpe Rank"] = df_metrics["Sharpe Ratio"].rank(ascending=False)
            df_metrics["MDD Rank"] = df_metrics["Max Drawdown (%)"].rank(ascending=True)
            df_metrics["Composite Score"] = df_metrics["Sharpe Rank"] + df_metrics["MDD Rank"]
            df_metrics.sort_values("Composite Score", inplace=True)
            st.subheader("Trade Metrics & Ranking")
            st.dataframe(df_metrics)

            st.subheader(f"Best Model: {df_metrics.iloc[0]['Model']}")
            best_model_name = df_metrics.iloc[0]['Model']
            best_model_path = os.path.join(MODEL_FOLDER, best_model_name)
            try:
                best_model = joblib.load(best_model_path)
                if hasattr(best_model, 'feature_importances_'):
                    importances = best_model.feature_importances_
                    feature_names = best_model.feature_names_in_ if hasattr(best_model, 'feature_names_in_') else [f"f{i}" for i in range(len(importances))]
                    fig_bar = px.bar(x=feature_names, y=importances,
                                     title=f"Feature Importances for {best_model_name}",
                                     labels={"x": "Features", "y": "Importance"})
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info(f"Best Model {best_model_name} does not support feature importances.")
            except Exception as e:
                st.error(f"Error loading best model {best_model_name}: {e}")
        else:
            st.info("No trade metrics available; check simulation results.")
        
        # ---- Equity Curves for All Models Interactive Chart ----
        fig_all = go.Figure()
        for mod_file, (dt_, eq_) in equity_curves.items():
            fig_all.add_trace(go.Scatter(x=dt_, y=eq_,
                                         mode='lines+markers',
                                         name=f"{mod_file} Final: {eq_.iloc[-1]:.2f}"))
        fig_all.update_layout(title="Equity Curves for All Models",
                              xaxis_title="Date", yaxis_title="Balance")
        st.plotly_chart(fig_all, use_container_width=True)

        # ---- BTC Price & Equity Curves for All Models Interactive Chart ----
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dual.add_trace(
            go.Scatter(x=combined_df['date'], y=combined_df['close'], name="BTC Price", line=dict(color="blue")),
            secondary_y=False)
        for mod_file, (dt_, eq_) in equity_curves.items():
            fig_dual.add_trace(
                go.Scatter(x=dt_, y=eq_, mode='lines+markers', name=mod_file),
                secondary_y=True)
        fig_dual.update_layout(title="BTC Price & Equity (All Models)")
        fig_dual.update_xaxes(title_text="Date")
        fig_dual.update_yaxes(title_text="BTC Price", secondary_y=False)
        fig_dual.update_yaxes(title_text="Balance", secondary_y=True)
        st.plotly_chart(fig_dual, use_container_width=True)
        
        # ---- PnL by Trade Type Bar Chart Interactive ----
        summary = {}
        for model_intvl, df_trades in trade_results_all.items():
            df_trades = df_trades.copy()
            type_sums = df_trades.groupby('type')['result'].sum()
            long_pnl = type_sums.get('Long', 0.0)
            short_pnl = type_sums.get('Short', 0.0)
            summary[model_intvl] = (long_pnl, short_pnl)
        model_labels = list(summary.keys())
        long_vals = [summary[k][0] for k in model_labels]
        short_vals = [summary[k][1] for k in model_labels]
        fig_bar = go.Figure(data=[
            go.Bar(name='Long', x=model_labels, y=long_vals),
            go.Bar(name='Short', x=model_labels, y=short_vals)
        ])
        fig_bar.update_layout(barmode='group',
                              title="PnL by Trade Type (Long vs. Short)",
                              xaxis_title="Model", yaxis_title="Total PnL")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # ---- Calendar Heatmap of Daily PnL Interactive ----
        st.subheader("Calendar Heatmap of Daily PnL")
        selection = st.selectbox("Select model for calendar heatmap", list(trade_results_all.keys()))
        df = trade_results_all[selection].copy()
        if 'exit_date' in df.columns:
            df['exit_date'] = pd.to_datetime(df['exit_date'])
            daily_pnl = df.groupby('exit_date')['result'].sum().reset_index()
            daily_pnl['date'] = daily_pnl['exit_date']
        else:
            df['date'] = pd.to_datetime(df['trade_date'])
            daily_pnl = df.groupby('date')['result'].sum().reset_index()
        daily_pnl.set_index('date', inplace=True)
        daily_pnl['day'] = daily_pnl.index.day
        daily_pnl['month'] = daily_pnl.index.month
        pivot_data = daily_pnl.pivot_table(index='month', columns='day', values='result', aggfunc='sum')
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn'
        ))
        
        annotations = []
        for i, month in enumerate(pivot_data.index):
            for j, day in enumerate(pivot_data.columns):
                value = pivot_data.iloc[i, j]
                text = "" if pd.isna(value) else str(round(value, 2))
                annotations.append(dict(
                    x=day,
                    y=month,
                    text=text,
                    showarrow=False,
                    font=dict(color="black")
                ))
        fig_heatmap.update_layout(
            title=f"Calendar Heatmap of Daily PnL - {selection}",
            xaxis_title="Day",
            yaxis_title="Month",
            annotations=annotations
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

# --------------------- Main ---------------------
def main():
    st.title("BTCUSDT Trading Simulation Dashboard")
    combined_df = run_simulation()
    if simulation_mode == "Real Time":
        if st.session_state.running:
            simulation_results(combined_df)
            progress_bar = st.progress(0)
            countdown_text = st.empty()
            for i in range(real_time_delay):
                progress = int((i + 1) * 100 / real_time_delay)
                progress_bar.progress(progress)
                countdown_text.text(f"Next update in {real_time_delay - i} seconds...")
                time.sleep(1)
            st.rerun()
        else:
            st.info("Real-Time Data Fetch is stopped. Click 'Start Real-Time' in the sidebar to begin.")
    else:
        if st.button("Run Back Test Simulation"):
            simulation_results(combined_df)

if __name__ == "__main__":
    main()
