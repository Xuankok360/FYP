import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import joblib
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
from torch.nn import BCEWithLogitsLoss
from torch.optim.adam import Adam
from collections import defaultdict
import sys

# -------------------------------------------------------------------------
# Define custom classes at module level (for safe unpickling)
# -------------------------------------------------------------------------
class MLPModule(nn.Module):
    def __init__(self, input_dim, neurons=64, dropout_rate=0.2):
        super(MLPModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, neurons)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(neurons, int(neurons / 2))
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(int(neurons / 2), 1)  # Output logits

    def forward(self, X, **kwargs):
        X = torch.relu(self.fc1(X))
        X = self.dropout1(X)
        X = torch.relu(self.fc2(X))
        X = self.dropout2(X)
        return self.fc3(X)

# Dummy classifier to support unpickling
class ReshapeNeuralNetClassifier(NeuralNetClassifier):
    pass

# Inject them into __main__
sys.modules['__main__'].MLPModule = MLPModule
sys.modules['__main__'].ReshapeNeuralNetClassifier = ReshapeNeuralNetClassifier

# Safe globals for torch
torch.serialization.add_safe_globals([
    MLPModule,
    ReshapeNeuralNetClassifier,
    torch.nn.modules.linear.Linear,
    torch.nn.modules.dropout.Dropout,
    BCEWithLogitsLoss,
    Adam,
    defaultdict,
    dict
])

# -------------------------------------------------------------------------
# Global file paths
# -------------------------------------------------------------------------
INTERVAL_FILES = {
    "5min": r"app\Interval\5min_oi_pi.csv",
    "1h":   r"app\Interval\1h_oi_pi.csv",
    "1d":   r"app\Interval\1d_oi_pi.csv"
}

MODEL_FILES = {
    "ensemble5min":           r"app\SelectedModel\ensemble5min_60%.pkl",
    "LogisticRegressionML1h": r"app\SelectedModel\best_log_reg_model_1h_51%.pkl",
    "RandomForest1h":         r"app\SelectedModel\best_rf_model_1h_51%.pkl",
    "RandomForest5min":       r"app\SelectedModel\best_rf_model_5min_58%.pkl"
}

# -------------------------------------------------------------------------
# Helper: Calculate RSI
# -------------------------------------------------------------------------
def calculate_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -------------------------------------------------------------------------
# Additional performance metrics
# -------------------------------------------------------------------------
def calculate_performance_metrics(trade_df, initial_balance):
    """Compute Sharpe ratio, Max drawdown, Profit factor, Overall return."""
    final_balance = trade_df['balance'].iloc[-1]
    overall_return = (final_balance - initial_balance) / initial_balance

    # Per-trade returns for Sharpe
    trade_df['balance_before'] = trade_df['balance'] - trade_df['result']
    trade_df['trade_return'] = trade_df['result'] / trade_df['balance_before']
    returns = trade_df['trade_return'].dropna()

    if len(returns) > 1 and returns.std() != 0:
        sharpe_ratio = returns.mean() / returns.std()
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    roll_max = trade_df['balance'].cummax()
    drawdown = (trade_df['balance'] - roll_max) / roll_max
    max_drawdown = drawdown.min()

    # Profit factor
    profits = trade_df.loc[trade_df['result'] > 0, 'result'].sum()
    losses = trade_df.loc[trade_df['result'] < 0, 'result'].sum()
    if losses != 0:
        profit_factor = profits / abs(losses)
    else:
        profit_factor = np.inf if profits > 0 else 1.0

    return {
        'Sharpe': sharpe_ratio,
        'MaxDD': max_drawdown,
        'ProfitFactor': profit_factor,
        'OverallReturn': overall_return
    }

# -------------------------------------------------------------------------
# Visualization helpers
# -------------------------------------------------------------------------
def plot_heatmap(combined_metrics, title="Performance Metrics Heatmap"):
    """
    combined_metrics is dict: { "Model-Interval": { 'Sharpe':..., 'MaxDD':..., 'ProfitFactor':..., 'OverallReturn':... }, ... }
    We'll show MaxDD and OverallReturn in % (multiply by 100).
    """
    df_perf = pd.DataFrame(combined_metrics).T
    columns = ['Sharpe', 'MaxDD', 'ProfitFactor', 'OverallReturn']
    df_perf = df_perf[columns]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Convert to numpy for display and adjust specific columns to percentage
    data_matrix = df_perf.values.copy()
    for i in range(data_matrix.shape[0]):
        data_matrix[i, 1] = data_matrix[i, 1] * 100         # MaxDD -> percentage
        data_matrix[i, 3] = data_matrix[i, 3] * 100         # OverallReturn -> percentage

    # Set up diverging normalization centered at 0:
    norm = TwoSlopeNorm(vcenter=0, vmin=np.min(data_matrix), vmax=np.max(data_matrix))

    # Use a diverging colormap such as "RdYlGn"
    im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn', norm=norm)

    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(df_perf.index)))
    ax.set_xticklabels(columns)
    ax.set_yticklabels(df_perf.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell (percentages for columns 1 and 3)
    for i in range(len(df_perf.index)):
        for j in range(len(columns)):
            val = data_matrix[i, j]
            if j == 1 or j == 3:
                text_val = f"{val:.2f}%"
            else:
                text_val = f"{val:.2f}"
            ax.text(j, i, text_val, ha="center", va="center", color="black")

    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Value")
    plt.tight_layout()
    st.pyplot(fig)

def plot_scatter_with_regression(combined_metrics, x_metric='MaxDD', y_metric='Sharpe'):
    df_perf = pd.DataFrame(combined_metrics).T
    if x_metric not in df_perf.columns or y_metric not in df_perf.columns:
        return

    x = df_perf[x_metric].values
    y = df_perf[y_metric].values

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.scatter(x, y)

    if len(x) > 1:
        coeffs = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = coeffs[0] * x_line + coeffs[1]
        ax.plot(x_line, y_line)

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f"All Models/Intervals: {y_metric} vs. {x_metric}")
    st.pyplot(fig)

def plot_boxplots_of_trade_returns(all_trade_dfs, title="Distribution of Trade Returns"):
    """
    all_trade_dfs: { "Model-Interval": trades_df, ... }
    We'll gather trade_return from each and create boxplots.
    """
    returns_data = []
    labels = []
    for model_intvl, df_trades in all_trade_dfs.items():
        if 'trade_return' in df_trades.columns:
            valid_returns = df_trades['trade_return'].dropna().values
            if len(valid_returns) > 0:
                returns_data.append(valid_returns)
                labels.append(model_intvl)

    if not returns_data:
        return

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(returns_data, labels=labels)
    ax.set_title(title)
    ax.set_ylabel("Per-Trade Return")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -------------------------------------------------------------------------
# Additional Visualizations
# -------------------------------------------------------------------------
def rolling_sharpe_and_return(all_trade_dfs, window=10):
    """
    Plot rolling Sharpe and rolling cumulative return for each model-interval in one figure.
    `window`: number of trades in the rolling window.
    """
    st.subheader("Rolling Sharpe & Return Line Chart")
    st.write(f"Using a rolling window of {window} trades.")

    # We'll do a separate line for each model-interval
    fig, ax = plt.subplots(figsize=(10, 5))

    for model_intvl, df_trades in all_trade_dfs.items():
        df_trades = df_trades.copy()
        df_trades['trade_idx'] = range(len(df_trades))

        # Compute rolling Sharpe over the last N trades
        df_trades['trade_return'] = df_trades['result'] / (df_trades['balance'] - df_trades['result'])
        rolling_sharpe = df_trades['trade_return'].rolling(window).apply(
            lambda x: x.mean()/x.std() if x.std() != 0 else np.nan, raw=False
        )
        ax.plot(df_trades['trade_idx'], rolling_sharpe, label=f"{model_intvl} Rolling Sharpe")

    ax.set_xlabel("Trade Index")
    ax.set_ylabel("Rolling Sharpe")
    ax.set_title("Rolling Sharpe Over Last N Trades")
    ax.legend()
    st.pyplot(fig)

    # Rolling cumulative return
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for model_intvl, df_trades in all_trade_dfs.items():
        df_trades = df_trades.copy()
        df_trades['trade_idx'] = range(len(df_trades))
        df_trades['cumulative_return'] = (df_trades['balance'] - 10000)/10000
        ax2.plot(df_trades['trade_idx'], df_trades['cumulative_return'], label=model_intvl)

    ax2.set_xlabel("Trade Index")
    ax2.set_ylabel("Cumulative Return")
    ax2.set_title("Rolling Cumulative Return by Trade Index")
    ax2.legend()
    st.pyplot(fig2)

def pnl_by_trade_type_bar_chart(all_trade_dfs):
    """
    Show total PnL for long vs short trades across all model-intervals.
    We'll produce a bar chart with grouping by model-interval.
    """
    st.subheader("PnL by Trade Type Bar Chart")

    summary = {}
    for model_intvl, df_trades in all_trade_dfs.items():
        df_trades = df_trades.copy()
        type_sums = df_trades.groupby('type')['result'].sum()
        long_pnl = type_sums.get('Long', 0.0)
        short_pnl = type_sums.get('Short', 0.0)
        summary[model_intvl] = (long_pnl, short_pnl)

    model_labels = list(summary.keys())
    long_vals = [summary[k][0] for k in model_labels]
    short_vals = [summary[k][1] for k in model_labels]

    x = np.arange(len(model_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, long_vals, width, label='Long')
    bars2 = ax.bar(x + width/2, short_vals, width, label='Short')

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha="right")
    ax.set_ylabel("Total PnL")
    ax.set_title("PnL by Trade Type (Long vs. Short)")
    ax.legend()
    st.pyplot(fig)

def calendar_heatmap_of_daily_pnl(all_trade_dfs):
    """
    Example calendar heatmap of daily PnL for a selected model-interval.
    """
    st.subheader("Calendar Heatmap of Daily PnL")

    all_keys = list(all_trade_dfs.keys())
    if not all_keys:
        st.write("No trade data found.")
        return

    selection = st.selectbox("Select model-interval for calendar heatmap", all_keys)
    df = all_trade_dfs[selection].copy()

    # We need daily PnL. We'll group trades by day, summing net_result
    if 'exit_date' in df.columns:
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        daily_pnl = df.groupby('exit_date')['result'].sum().reset_index()
        daily_pnl['date'] = daily_pnl['exit_date']
    else:
        daily_pnl = df.copy()
        daily_pnl['date'] = pd.to_datetime(df['trade_date'])
        daily_pnl = daily_pnl.groupby('date')['result'].sum().reset_index()

    daily_pnl.set_index('date', inplace=True)
    daily_pnl['day'] = daily_pnl.index.day
    daily_pnl['month'] = daily_pnl.index.month
    pivot_data = daily_pnl.pivot(index='month', columns='day', values='result')

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot_data, aspect='auto', cmap='RdYlGn')
    ax.set_title(f"Calendar Heatmap of Daily PnL - {selection}")

    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels(pivot_data.index)
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels(pivot_data.columns)

    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            val = pivot_data.iloc[i, j]
            if pd.notnull(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color="black")

    plt.colorbar(im, ax=ax, label="Daily PnL")
    plt.tight_layout()
    st.pyplot(fig)

def feature_importance_demo():
    """
    A placeholder to show how you'd display feature importances if available
    (e.g., random forest or logistic regression).
    """
    st.subheader("Feature Importance (Placeholder)")
    st.write("If your model objects expose `.feature_importances_` (RandomForest) or `.coef_` (LogReg), you can show them here.")
    st.write("For ensemble5min or MLP-based models, there's no direct built-in 'feature importance'. You could use e.g. SHAP for deeper analysis.")

def drawdown_area_plot(all_trade_dfs):
    """
    Plot a drawdown area chart for each model-interval in subplots.
    """
    st.subheader("Drawdown Area Plot")
    n_plots = len(all_trade_dfs)
    if n_plots == 0:
        st.write("No data to plot.")
        return

    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 4*n_plots), sharex=False)

    if n_plots == 1:
        axs = [axs]  # make it iterable

    idx = 0
    for model_intvl, df_trades in all_trade_dfs.items():
        df = df_trades.copy()
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df.sort_values('entry_date', inplace=True)

        eq = df['balance'].values
        running_max = np.maximum.accumulate(eq)
        dd = (eq - running_max) / running_max  # negative or zero

        axs[idx].fill_between(df['entry_date'], dd, 0, where=(dd < 0), color='red', alpha=0.3)
        axs[idx].plot(df['entry_date'], dd, color='red')
        axs[idx].set_title(f"Drawdown for {model_intvl}")
        axs[idx].set_ylabel("Drawdown (%)")
        axs[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[idx].xaxis.set_major_locator(mdates.AutoDateLocator())
        idx += 1

    plt.tight_layout()
    st.pyplot(fig)

# -------------------------------------------------------------------------
# Model runner functions
# -------------------------------------------------------------------------
def run_ensemble5min():
    model_name = "Ensemble5min"
    model_path = MODEL_FILES["ensemble5min"]
    best_model = joblib.load(model_path)

    features_to_use = ['RSI', 'SMA20', 'STD20', 'UpperBand', 'LowerBand', 'volume']
    if not hasattr(best_model, 'feature_names_in_'):
        best_model.feature_names_in_ = np.array(features_to_use)

    equity_curves = {}
    trade_dfs = {}
    combined_perf = {}
    initial_balance = 10000

    for label, path in INTERVAL_FILES.items():
        df = pd.read_csv(path, parse_dates=['date'], na_values=["Blank"])
        df = df[df['date'] >= pd.to_datetime("2024-04-01")]
        df.dropna(inplace=True)
        df.sort_values('date', inplace=True)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'oi', 'coinbase_premium_index']
        for col in numeric_cols:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

        # Indicators
        df['RSI'] = calculate_RSI(df['close'], period=14)
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['STD20'] = df['close'].rolling(window=20).std()
        df['UpperBand'] = df['SMA20'] + (2 * df['STD20'])
        df['LowerBand'] = df['SMA20'] - (2 * df['STD20'])
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['momentum'] = df['close'] - df['close'].shift(1)

        # Lags
        df['RSI_lag_1'] = df['RSI'].shift(1)
        df['SMA20_lag_1'] = df['SMA20'].shift(1)
        df['UpperBand_lag_1'] = df['UpperBand'].shift(1)
        df['LowerBand_lag_1'] = df['LowerBand'].shift(1)
        df['MACD_lag_1'] = df['MACD'].shift(1)
        df['MACD_signal_lag_1'] = df['MACD_signal'].shift(1)
        df['momentum_lag_1'] = df['momentum'].shift(1)
        df['volume_lag_1'] = df['volume'].shift(1)
        df['oi_lag_1'] = df['oi'].shift(1)

        # Targets
        n = 1
        df['future_return'] = (df['close'].shift(-n) - df['close']) / df['close']
        threshold = 0.005
        df['action'] = np.where(df['future_return'] > threshold, 0,
                                np.where(df['future_return'] < -threshold, 1, np.nan))
        df.dropna(subset=['action'], inplace=True)
        df = df[:-n]
        df.drop(columns=['future_return'], inplace=True)

        # Scaling
        df_raw = df.copy()
        df_scaled = df.copy()
        df_scaled.dropna(subset=features_to_use, inplace=True)
        scaler = StandardScaler()
        df_scaled[features_to_use] = scaler.fit_transform(df_scaled[features_to_use])

        X = df_scaled[features_to_use]
        expected_features = best_model.feature_names_in_
        X = X[expected_features].to_numpy().astype(np.float32)

        test_indices = df_scaled.index
        df_test_sim = df_raw.loc[test_indices].copy()

        preds = best_model.predict(X)
        df_test_sim['pred_signal'] = preds
        df_test_sim['trade_date'] = df_test_sim['date'].dt.date

        grouped = df_test_sim.groupby('trade_date')
        dates_list = sorted(grouped.groups.keys())

        balance = initial_balance
        fee_rate = 0.001
        cumulative_fee = 0
        trade_logs = []
        for i in range(len(dates_list) - 1):
            current_date = dates_list[i]
            next_date = dates_list[i + 1]
            day_data = grouped.get_group(current_date)
            entry_row = day_data.iloc[0]
            entry_price = entry_row['open']
            signal = entry_row['pred_signal']

            next_day_data = grouped.get_group(next_date)
            exit_price = next_day_data.iloc[0]['open']
            trade_type = 'Long' if signal == 0 else 'Short'

            if trade_type == 'Long':
                pnl = (exit_price - entry_price) / entry_price * balance
            else:
                pnl = (entry_price - exit_price) / entry_price * balance

            fee = fee_rate * balance
            cumulative_fee += fee
            net_result = pnl - fee
            balance += net_result

            trade_logs.append({
                'entry_date': current_date,
                'exit_date': next_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'fee': fee,
                'result': net_result,
                'balance': balance,
                'type': trade_type,
                'cumulative_fee': cumulative_fee
            })

        df_trades = pd.DataFrame(trade_logs)
        trade_dfs[label] = df_trades
        equity_curves[label] = (pd.to_datetime(df_trades['entry_date']), df_trades['balance'])
        pm = calculate_performance_metrics(df_trades.copy(), initial_balance)
        combined_perf[f"{model_name}-{label}"] = pm

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, (dates, balances) in equity_curves.items():
        perf = (balances.iloc[-1] - initial_balance) / initial_balance * 100
        ax.plot(dates, balances, label=f"{label} ({perf:.1f}%)")
    ax.set_title(f"{model_name} Equity Curves")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig, trade_dfs, combined_perf

def run_logistic_regression_ml1h():
    model_name = "LogReg1h"
    model_path = MODEL_FILES["LogisticRegressionML1h"]
    model = joblib.load(model_path)

    features_to_use = ['coinbase_premium_index', 'volume', 'RSI', 'STD20', 'UpperBand', 'LowerBand']
    equity_curves = {}
    trade_dfs = {}
    combined_perf = {}
    initial_balance = 10000

    for label, path in INTERVAL_FILES.items():
        df = pd.read_csv(path, parse_dates=['date'], na_values=["Blank"])
        df = df[df['date'] >= pd.to_datetime("2024-04-01")]
        df.dropna(inplace=True)
        df.sort_values('date', inplace=True)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'oi', 'coinbase_premium_index']
        for col in numeric_cols:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

        # Indicators
        df['RSI'] = calculate_RSI(df['close'], period=14)
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['STD20'] = df['close'].rolling(window=20).std()
        df['UpperBand'] = df['SMA20'] + 2*df['STD20']
        df['LowerBand'] = df['SMA20'] - 2*df['STD20']
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['momentum'] = df['close'] - df['close'].shift(1)

        # Lags
        df['RSI_lag_1'] = df['RSI'].shift(1)
        df['SMA20_lag_1'] = df['SMA20'].shift(1)
        df['UpperBand_lag_1'] = df['UpperBand'].shift(1)
        df['LowerBand_lag_1'] = df['LowerBand'].shift(1)
        df['MACD_lag_1'] = df['MACD'].shift(1)
        df['MACD_signal_lag_1'] = df['MACD_signal'].shift(1)
        df['momentum_lag_1'] = df['momentum'].shift(1)
        df['volume_lag_1'] = df['volume'].shift(1)
        df['oi_lag_1'] = df['oi'].shift(1)

        n = 1
        df['future_return'] = (df['close'].shift(-n) - df['close']) / df['close']
        threshold = 0.005
        df['action'] = np.where(df['future_return'] > threshold, 0,
                                np.where(df['future_return'] < -threshold, 1, np.nan))
        df.dropna(subset=['action'], inplace=True)
        df = df[:-n]
        df.drop(columns=['future_return'], inplace=True)

        df_raw = df.copy()
        df_scaled = df.copy()
        df_scaled.dropna(subset=features_to_use, inplace=True)
        scaler = StandardScaler()
        df_scaled[features_to_use] = scaler.fit_transform(df_scaled[features_to_use])
        X = df_scaled[features_to_use]
        expected_features = model.feature_names_in_
        X = X[expected_features]

        test_indices = df_scaled.index
        df_test_sim = df_raw.loc[test_indices].copy()

        preds = model.predict(X)
        df_test_sim['pred_signal'] = preds
        df_test_sim['trade_date'] = df_test_sim['date'].dt.date

        grouped = df_test_sim.groupby('trade_date')
        dates_list = sorted(grouped.groups.keys())

        balance = initial_balance
        fee_rate = 0.001
        cumulative_fee = 0
        trade_logs = []
        for i in range(len(dates_list) - 1):
            current_date = dates_list[i]
            next_date = dates_list[i+1]
            day_data = grouped.get_group(current_date)
            entry_row = day_data.iloc[0]
            entry_price = entry_row['open']
            signal = entry_row['pred_signal']

            next_day_data = grouped.get_group(next_date)
            exit_price = next_day_data.iloc[0]['open']

            trade_type = 'Long' if signal == 0 else 'Short'
            if trade_type == 'Long':
                pnl = (exit_price - entry_price) / entry_price * balance
            else:
                pnl = (entry_price - exit_price) / entry_price * balance

            fee = fee_rate * balance
            cumulative_fee += fee
            net_result = pnl - fee
            balance += net_result

            trade_logs.append({
                'entry_date': current_date,
                'exit_date': next_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'fee': fee,
                'result': net_result,
                'balance': balance,
                'type': trade_type
            })

        df_trades = pd.DataFrame(trade_logs)
        trade_dfs[label] = df_trades
        equity_curves[label] = (pd.to_datetime(df_trades['entry_date']), df_trades['balance'])
        pm = calculate_performance_metrics(df_trades.copy(), initial_balance)
        combined_perf[f"{model_name}-{label}"] = pm

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, (dates, balances) in equity_curves.items():
        perf = (balances.iloc[-1] - initial_balance) / initial_balance * 100
        ax.plot(dates, balances, label=f"{label} ({perf:.1f}%)")
    ax.set_title(f"{model_name} Equity Curves")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig, trade_dfs, combined_perf

def run_random_forest_1h():
    model_name = "RF1h"
    model_path = MODEL_FILES["RandomForest1h"]
    model = joblib.load(model_path)

    features_to_use = ['coinbase_premium_index', 'volume', 'RSI', 'STD20', 'UpperBand', 'LowerBand']
    equity_curves = {}
    trade_dfs = {}
    combined_perf = {}
    initial_balance = 10000

    for label, path in INTERVAL_FILES.items():
        df = pd.read_csv(path, parse_dates=['date'], na_values=["Blank"])
        df = df[df['date'] >= pd.to_datetime("2024-04-01")]
        df.dropna(inplace=True)
        df.sort_values('date', inplace=True)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'oi', 'coinbase_premium_index']
        for col in numeric_cols:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

        # Indicators
        df['RSI'] = calculate_RSI(df['close'], period=14)
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['STD20'] = df['close'].rolling(window=20).std()
        df['UpperBand'] = df['SMA20'] + (2 * df['STD20'])
        df['LowerBand'] = df['SMA20'] - (2 * df['STD20'])
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['momentum'] = df['close'] - df['close'].shift(1)

        # Lags
        df['RSI_lag_1'] = df['RSI'].shift(1)
        df['SMA20_lag_1'] = df['SMA20'].shift(1)
        df['UpperBand_lag_1'] = df['UpperBand'].shift(1)
        df['LowerBand_lag_1'] = df['LowerBand'].shift(1)
        df['MACD_lag_1'] = df['MACD'].shift(1)
        df['MACD_signal_lag_1'] = df['MACD_signal'].shift(1)
        df['momentum_lag_1'] = df['momentum'].shift(1)
        df['volume_lag_1'] = df['volume'].shift(1)
        df['oi_lag_1'] = df['oi'].shift(1)

        n = 1
        df['future_return'] = (df['close'].shift(-n) - df['close']) / df['close']
        threshold = 0.005
        df['action'] = np.where(df['future_return'] > threshold, 0,
                                np.where(df['future_return'] < -threshold, 1, np.nan))
        df.dropna(subset=['action'], inplace=True)
        df = df[:-n]
        df.drop(columns=['future_return'], inplace=True)

        df_raw = df.copy()
        df_scaled = df.copy()
        df_scaled.dropna(subset=features_to_use, inplace=True)
        scaler = StandardScaler()
        df_scaled[features_to_use] = scaler.fit_transform(df_scaled[features_to_use])
        X = df_scaled[features_to_use]
        expected_features = model.feature_names_in_
        X = X[expected_features]

        test_indices = df_scaled.index
        df_test_sim = df_raw.loc[test_indices].copy()

        preds = model.predict(X)
        df_test_sim['pred_signal'] = preds
        df_test_sim['trade_date'] = df_test_sim['date'].dt.date

        grouped = df_test_sim.groupby('trade_date')
        dates_list = sorted(grouped.groups.keys())

        balance = initial_balance
        fee_rate = 0.001
        cumulative_fee = 0
        trade_logs = []
        for i in range(len(dates_list) - 1):
            current_date = dates_list[i]
            next_date = dates_list[i+1]
            day_data = grouped.get_group(current_date)
            entry_row = day_data.iloc[0]
            entry_price = entry_row['open']
            signal = entry_row['pred_signal']

            next_day_data = grouped.get_group(next_date)
            exit_price = next_day_data.iloc[0]['open']

            trade_type = 'Long' if signal == 0 else 'Short'
            if trade_type == 'Long':
                pnl = (exit_price - entry_price) / entry_price * balance
            else:
                pnl = (entry_price - exit_price) / entry_price * balance

            fee = fee_rate * balance
            cumulative_fee += fee
            net_result = pnl - fee
            balance += net_result

            trade_logs.append({
                'entry_date': current_date,
                'exit_date': next_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'fee': fee,
                'result': net_result,
                'balance': balance,
                'type': trade_type
            })

        df_trades = pd.DataFrame(trade_logs)
        trade_dfs[label] = df_trades
        equity_curves[label] = (pd.to_datetime(df_trades['entry_date']), df_trades['balance'])
        pm = calculate_performance_metrics(df_trades.copy(), initial_balance)
        combined_perf[f"{model_name}-{label}"] = pm

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, (dates, balances) in equity_curves.items():
        perf = (balances.iloc[-1] - initial_balance) / initial_balance * 100
        ax.plot(dates, balances, label=f"{label} ({perf:.1f}%)")
    ax.set_title(f"{model_name} Equity Curves")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig, trade_dfs, combined_perf

def run_random_forest_5min():
    model_name = "RF5min"
    model_path = MODEL_FILES["RandomForest5min"]
    model = joblib.load(model_path)

    features_to_use = ['close', 'RSI', 'volume', 'coinbase_premium_index', 'LowerBand', 'STD20']
    equity_curves = {}
    trade_dfs = {}
    combined_perf = {}
    initial_balance = 10000

    for label, path in INTERVAL_FILES.items():
        df = pd.read_csv(path, parse_dates=['date'], na_values=["Blank"])
        df = df[df['date'] >= pd.to_datetime("2024-04-01")]
        df.dropna(inplace=True)
        df.sort_values('date', inplace=True)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'oi', 'coinbase_premium_index']
        for col in numeric_cols:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

        # Indicators
        df['RSI'] = calculate_RSI(df['close'], period=14)
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['STD20'] = df['close'].rolling(window=20).std()
        df['UpperBand'] = df['SMA20'] + (2 * df['STD20'])
        df['LowerBand'] = df['SMA20'] - (2 * df['STD20'])
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['momentum'] = df['close'] - df['close'].shift(1)

        # Lags
        df['RSI_lag_1'] = df['RSI'].shift(1)
        df['SMA20_lag_1'] = df['SMA20'].shift(1)
        df['UpperBand_lag_1'] = df['UpperBand'].shift(1)
        df['LowerBand_lag_1'] = df['LowerBand'].shift(1)
        df['MACD_lag_1'] = df['MACD'].shift(1)
        df['MACD_signal_lag_1'] = df['MACD_signal'].shift(1)
        df['momentum_lag_1'] = df['momentum'].shift(1)
        df['volume_lag_1'] = df['volume'].shift(1)
        df['oi_lag_1'] = df['oi'].shift(1)

        n = 1
        df['future_return'] = (df['close'].shift(-n) - df['close']) / df['close']
        threshold = 0.005
        df['action'] = np.where(df['future_return'] > threshold, 0,
                                np.where(df['future_return'] < -threshold, 1, np.nan))
        df.dropna(subset=['action'], inplace=True)
        df = df[:-n]
        df.drop(columns=['future_return'], inplace=True)

        df_raw = df.copy()
        df_scaled = df.copy()
        df_scaled.dropna(subset=features_to_use, inplace=True)
        scaler = StandardScaler()
        df_scaled[features_to_use] = scaler.fit_transform(df_scaled[features_to_use])
        X = df_scaled[features_to_use]
        expected_features = model.feature_names_in_
        X = X[expected_features]

        test_indices = df_scaled.index
        df_test_sim = df_raw.loc[test_indices].copy()

        preds = model.predict(X)
        df_test_sim['pred_signal'] = preds
        df_test_sim['trade_date'] = df_test_sim['date'].dt.date

        grouped = df_test_sim.groupby('trade_date')
        dates_list = sorted(grouped.groups.keys())

        balance = initial_balance
        fee_rate = 0.001
        cumulative_fee = 0
        trade_logs = []
        for i in range(len(dates_list) - 1):
            current_date = dates_list[i]
            next_date = dates_list[i + 1]
            day_data = grouped.get_group(current_date)
            entry_row = day_data.iloc[0]
            entry_price = entry_row['open']
            signal = entry_row['pred_signal']

            next_day_data = grouped.get_group(next_date)
            exit_price = next_day_data.iloc[0]['open']
            trade_type = 'Long' if signal == 0 else 'Short'

            if trade_type == 'Long':
                pnl = (exit_price - entry_price) / entry_price * balance
            else:
                pnl = (entry_price - exit_price) / entry_price * balance

            fee = fee_rate * balance
            cumulative_fee += fee
            net_result = pnl - fee
            balance += net_result

            trade_logs.append({
                'entry_date': current_date,
                'exit_date': next_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'fee': fee,
                'result': net_result,
                'balance': balance,
                'type': trade_type
            })

        df_trades = pd.DataFrame(trade_logs)
        trade_dfs[label] = df_trades
        equity_curves[label] = (pd.to_datetime(df_trades['entry_date']), df_trades['balance'])
        pm = calculate_performance_metrics(df_trades.copy(), initial_balance)
        combined_perf[f"{model_name}-{label}"] = pm

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, (dates, balances) in equity_curves.items():
        perf = (balances.iloc[-1] - initial_balance) / initial_balance * 100
        ax.plot(dates, balances, label=f"{label} ({perf:.1f}%)")
    ax.set_title(f"{model_name} Equity Curves")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig, trade_dfs, combined_perf

# 1️⃣ ────────────────────────────────────────────────────────────────────────────
# MODEL‑to‑FEATURES mapping, used by the new plot
# ──────────────────────────────────────────────────────────────────────────────
MODEL_FEATURES = {                                                             ### <-- NEW
    "Ensemble5min":       ['RSI', 'SMA20', 'STD20', 'UpperBand',               ### <-- NEW
                           'LowerBand', 'volume'],                             ### <-- NEW
    "LogReg1h":           ['coinbase_premium_index', 'volume', 'RSI',          ### <-- NEW
                           'STD20', 'UpperBand', 'LowerBand'],                 ### <-- NEW
    "RF1h":               ['coinbase_premium_index', 'volume', 'RSI',          ### <-- NEW
                           'STD20', 'UpperBand', 'LowerBand'],                 ### <-- NEW
    "RF5min":             ['close', 'RSI', 'volume', 'coinbase_premium_index', ### <-- NEW
                           'LowerBand', 'STD20']                               ### <-- NEW
}                                                                              ### <-- NEW


# 2️⃣ ────────────────────────────────────────────────────────────────────────────
# NEW helper – heat‑map of which model uses which feature
# ──────────────────────────────────────────────────────────────────────────────
def plot_features_used_heatmap(model_features):                                ### <-- NEW
    """Visual heat‑map (feature rows × model cols) showing feature inclusion."""### <-- NEW
    st.subheader("Features used by each model")                                ### <-- NEW
    # Flatten → unique sorted list of all features                              
    all_feats = sorted({f for feats in model_features.values() for f in feats})### <-- NEW
    models    = list(model_features.keys())                                    ### <-- NEW
    # Build boolean matrix (feat × model)
    mat = np.zeros((len(all_feats), len(models)))                              ### <-- NEW
    for j, m in enumerate(models):                                             ### <-- NEW
        for f in model_features[m]:                                            ### <-- NEW
            mat[all_feats.index(f), j] = 1                                     ### <-- NEW
    # Plot
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(all_feats)))                 ### <-- NEW
    im = ax.imshow(mat, aspect='auto', cmap='Greens')                          ### <-- NEW
    ax.set_xticks(np.arange(len(models)))                                      ### <-- NEW
    ax.set_xticklabels(models, rotation=45, ha="right")                        ### <-- NEW
    ax.set_yticks(np.arange(len(all_feats)))                                   ### <-- NEW
    ax.set_yticklabels(all_feats)                                              ### <-- NEW
    ax.set_title("Feature usage by model (dark = feature included)")           ### <-- NEW
    plt.colorbar(im, ax=ax, label="Used")                                      ### <-- NEW
    st.pyplot(fig)                                                             ### <-- NEW


# 3️⃣ ────────────────────────────────────────────────────────────────────────────
# MAIN APP (unchanged, except last lines – look for “### <‑‑ NEW”)
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.title("Trading Model Simulator - Side by Side + Combined Visualizations")

    # Run each model
    fig_ens, df_ens, perf_ens = run_ensemble5min()
    fig_lr,  df_lr,  perf_lr  = run_logistic_regression_ml1h()
    fig_rf1, df_rf1, perf_rf1 = run_random_forest_1h()
    fig_rf5, df_rf5, perf_rf5 = run_random_forest_5min()

    # … (the two rows of equity‑curve plots exactly as before) …

    # Merge metrics & trade logs (exactly as you had)
    combined_metrics = {**perf_ens, **perf_lr, **perf_rf1, **perf_rf5}
    combined_trade_dfs = {}
    for tag, d in [("Ens", df_ens), ("LogReg1h", df_lr),
                   ("RF1h", df_rf1), ("RF5min", df_rf5)]:
        for interval, trades in d.items():
            trades['balance_before'] = trades['balance'] - trades['result']
            trades['trade_return']   = trades['result'] / trades['balance_before']
            combined_trade_dfs[f"{tag}-{interval}"] = trades

    st.markdown("---")
    st.markdown("## Combined Visualizations for All Models + Intervals")

    # Existing combined plots
    plot_heatmap(combined_metrics, title="All Models/Intervals - Performance Metrics")
    plot_scatter_with_regression(combined_metrics, x_metric='MaxDD', y_metric='Sharpe')
    plot_boxplots_of_trade_returns(combined_trade_dfs, title="All Models/Intervals - Distribution of Trade Returns")

    # NEW: show which features each model consumes
    plot_features_used_heatmap(MODEL_FEATURES)                                 ### <-- NEW

    st.markdown("---")
    # The rest of your visualisations
    rolling_sharpe_and_return(combined_trade_dfs, window=10)
    pnl_by_trade_type_bar_chart(combined_trade_dfs)
    calendar_heatmap_of_daily_pnl(combined_trade_dfs)
    feature_importance_demo()
    drawdown_area_plot(combined_trade_dfs)


if __name__ == "__main__":
    main()