# Stock Market Analysis Dashboard
# Built with Streamlit
# Prepared by: Chandani Sah | Oracle Brain Internship

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title = 'Stock Market Analysis',
    page_icon  = '📈',
    layout     = 'wide'
)

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('stocks.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df         = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    return df

df = load_data()

# ── Sidebar Filters ───────────────────────────────────────────
st.sidebar.markdown('## 🎛 Filters')
st.sidebar.markdown('---')

selected_ticker = st.sidebar.selectbox(
    '🏢 Select Company',
    options=['All', 'AAPL', 'MSFT', 'NFLX', 'GOOG']
)

st.sidebar.markdown('---')
st.sidebar.markdown('**📌 About:**')
st.sidebar.markdown('Complete stock market analysis of AAPL, MSFT, NFLX and GOOG covering Feb–May 2023.')
st.sidebar.markdown('---')
st.sidebar.markdown('**👩‍💻 Prepared by:**')
st.sidebar.markdown('Chandani Sah')
st.sidebar.markdown('Data Analyst Intern')
st.sidebar.markdown('Oracle Brain, New Baneshwor')

# ── Separate stocks ───────────────────────────────────────────
def get_stock_data(ticker, data):
    d = data[data['Ticker'] == ticker].reset_index(drop=True)
    if len(d) > 0:
        d['MA20']         = d['Close'].rolling(window=20).mean()
        d['MA50']         = d['Close'].rolling(window=50).mean()
        d['Daily_Return'] = d['Close'].pct_change() * 100
    return d

aapl_full = get_stock_data('AAPL', df)
msft_full = get_stock_data('MSFT', df)
nflx_full = get_stock_data('NFLX', df)
goog_full = get_stock_data('GOOG', df)

stocks_full = [
    ('AAPL', aapl_full, 'steelblue'),
    ('MSFT', msft_full, 'seagreen'),
    ('NFLX', nflx_full, 'tomato'),
    ('GOOG', goog_full, 'orange')
]

# ── Decide which stocks to show ───────────────────────────────
if selected_ticker == 'All':
    active_stocks = stocks_full
else:
    color_map = {'AAPL': 'steelblue', 'MSFT': 'seagreen', 'NFLX': 'tomato', 'GOOG': 'orange'}
    full_map  = {'AAPL': aapl_full, 'MSFT': msft_full, 'NFLX': nflx_full, 'GOOG': goog_full}
    active_stocks = [(selected_ticker, full_map[selected_ticker], color_map[selected_ticker])]

# ── Header ────────────────────────────────────────────────────
st.title('📈 Stock Market Analysis Dashboard')
st.markdown('**Companies:** Apple (AAPL) | Microsoft (MSFT) | Netflix (NFLX) | Google (GOOG)')
st.markdown('**Period:** February 2023 — May 2023 | **Prepared by:** Chandani Sah | Oracle Brain Internship')
st.markdown('---')

# ── KPI Cards ─────────────────────────────────────────────────
st.markdown('### 📊 Performance Summary')
cols = st.columns(4)
for col, (ticker, data) in zip(cols, [('AAPL', aapl_full), ('MSFT', msft_full), ('NFLX', nflx_full), ('GOOG', goog_full)]):
    if len(data) > 1:
        start  = data['Close'].iloc[0]
        end    = data['Close'].iloc[-1]
        change = round(((end - start) / start) * 100, 2)
        col.metric(label=ticker, value=str(round(end, 2)) + ' USD', delta=str(change) + '%')

st.markdown('---')

# ── Section 1: Trend Analysis ─────────────────────────────────
st.markdown('### 1. 📈 Stock Price Trends')
st.info('Closing price movement over the selected period. Shows overall direction — upward, downward or flat.')

if len(active_stocks) == 1:
    ticker, data, color = active_stocks[0]
    if len(data) > 1:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(data['Date'], data['Close'], color=color, linewidth=2)
        ax.fill_between(data['Date'], data['Close'], alpha=0.15, color=color)
        ax.set_title(ticker + ' — Closing Price Trend', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.tick_params(axis='x', rotation=25)
        plt.tight_layout()
        st.pyplot(fig)
else:
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle('Closing Price Trends', fontsize=16, fontweight='bold')
    axes_flat = axes.flatten()
    for i, (ticker, data, color) in enumerate(active_stocks):
        if len(data) > 1:
            axes_flat[i].plot(data['Date'], data['Close'], color=color, linewidth=2)
            axes_flat[i].fill_between(data['Date'], data['Close'], alpha=0.15, color=color)
            axes_flat[i].set_title(ticker, fontweight='bold')
            axes_flat[i].set_xlabel('Date')
            axes_flat[i].set_ylabel('Price (USD)')
            axes_flat[i].tick_params(axis='x', rotation=25)
    for j in range(len(active_stocks), 4):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

if selected_ticker == 'All':
    st.markdown('**Normalized Price Comparison (Base = 100)**')
    fig, ax = plt.subplots(figsize=(14, 5))
    for ticker, data, color in stocks_full:
        if len(data) > 1:
            normalized = (data['Close'] / data['Close'].iloc[0]) * 100
            ax.plot(data['Date'], normalized, label=ticker, color=color, linewidth=2)
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_title('Normalized Price Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price')
    ax.legend()
    ax.tick_params(axis='x', rotation=25)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown('---')

# ── Section 2: Volume Analysis ────────────────────────────────
st.markdown('### 2. 📊 Volume Analysis')
st.info('Daily trading volume shows how actively each stock is being bought and sold. Spikes often signal important market events.')

n     = len(active_stocks)
ncols = min(2, n)
nrows = (n + 1) // 2
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
fig.suptitle('Daily Trading Volume', fontsize=16, fontweight='bold')
axes_flat = [axes] if n == 1 else axes.flatten()
for i, (ticker, data, color) in enumerate(active_stocks):
    if len(data) > 1:
        axes_flat[i].bar(data['Date'], data['Volume'] / 1e6, color=color, alpha=0.75, width=0.8)
        axes_flat[i].set_title(ticker + ' — Avg: ' + str(round(data['Volume'].mean() / 1e6, 1)) + 'M', fontweight='bold')
        axes_flat[i].set_xlabel('Date')
        axes_flat[i].set_ylabel('Volume (Millions)')
        axes_flat[i].tick_params(axis='x', rotation=25)
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.tight_layout()
st.pyplot(fig)

st.markdown('---')

# ── Section 3: Moving Averages ────────────────────────────────
st.markdown('### 3. 〰️ Moving Averages (20-Day & 50-Day)')
st.info('Moving averages smooth out daily price noise to reveal the underlying trend. When price stays above MA20 it confirms an uptrend.')

n     = len(active_stocks)
ncols = min(2, n)
nrows = (n + 1) // 2
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
fig.suptitle('Moving Averages — 20-Day & 50-Day', fontsize=16, fontweight='bold')
axes_flat = [axes] if n == 1 else axes.flatten()
for i, (ticker, data, color) in enumerate(active_stocks):
    if len(data) > 1:
        axes_flat[i].plot(data['Date'], data['Close'], color=color,  alpha=0.5, linewidth=1.5, label='Close')
        axes_flat[i].plot(data['Date'], data['MA20'],  color='navy', linewidth=2, linestyle='--', label='MA20')
        axes_flat[i].plot(data['Date'], data['MA50'],  color='purple', linewidth=2, linestyle='-.', label='MA50')
        axes_flat[i].set_title(ticker, fontweight='bold')
        axes_flat[i].set_xlabel('Date')
        axes_flat[i].set_ylabel('Price (USD)')
        axes_flat[i].legend(fontsize=8)
        axes_flat[i].tick_params(axis='x', rotation=25)
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.tight_layout()
st.pyplot(fig)

st.markdown('---')

# ── Section 4: Volatility ─────────────────────────────────────
st.markdown('### 4. 🎢 Volatility Analysis')
st.info('Volatility measures how wildly a stock price jumps around day to day. Higher volatility means higher risk.')

col_left, col_right = st.columns(2)

with col_left:
    n     = len(active_stocks)
    ncols = min(2, n)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, nrows * 4))
    fig.suptitle('Daily Returns', fontsize=14, fontweight='bold')
    axes_flat = [axes] if n == 1 else axes.flatten()
    for i, (ticker, data, color) in enumerate(active_stocks):
        if len(data) > 1:
            axes_flat[i].bar(data['Date'], data['Daily_Return'], color=color, alpha=0.7, width=0.8)
            axes_flat[i].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
            axes_flat[i].set_title(ticker, fontweight='bold')
            axes_flat[i].set_xlabel('Date')
            axes_flat[i].set_ylabel('Return (%)')
            axes_flat[i].tick_params(axis='x', rotation=25)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

with col_right:
    returns_dict = {}
    for ticker, data, color in active_stocks:
        if len(data) > 1 and 'Daily_Return' in data.columns:
            returns_dict[ticker] = data['Daily_Return'].dropna().values
    if returns_dict:
        returns_df     = pd.DataFrame({k: pd.Series(v) for k, v in returns_dict.items()})
        returns_melted = returns_df.melt(var_name='Stock', value_name='Daily Return (%)')
        palette        = ['steelblue', 'seagreen', 'tomato', 'orange']
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(data=returns_melted, x='Stock', y='Daily Return (%)', palette=palette[:len(returns_dict)], inner='box', ax=ax)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title('Return Distribution', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    vol_rows = []
    for ticker, data, color in active_stocks:
        if len(data) > 1 and 'Daily_Return' in data.columns:
            std  = data['Daily_Return'].std()
            risk = 'High' if std > 2 else ('Medium' if std > 1.5 else 'Low')
            vol_rows.append({
                'Stock':          ticker,
                'Avg Return (%)': round(data['Daily_Return'].mean(), 3),
                'Volatility (%)': round(std, 3),
                'Best Day (%)':   round(data['Daily_Return'].max(), 2),
                'Worst Day (%)':  round(data['Daily_Return'].min(), 2),
                'Risk':           risk
            })
    if vol_rows:
        st.markdown('**Volatility Summary**')
        st.dataframe(pd.DataFrame(vol_rows).sort_values('Volatility (%)', ascending=False), use_container_width=True)

st.markdown('---')

# ── Section 5: Correlation ────────────────────────────────────
st.markdown('### 5. 🤝 Correlation Analysis')
st.info('Correlation shows how much two stocks move together. Close to +1 means they move in sync. Close to 0 means they move independently.')

close_pivot = df.pivot(index='Date', columns='Ticker', values='Close')
corr_matrix = close_pivot.corr()

col_left, col_right = st.columns(2)
with col_left:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', linewidths=0.5, square=True, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlation Heatmap', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

with col_right:
    fig, axes = plt.subplots(4, 4, figsize=(7, 5))
    pd.plotting.scatter_matrix(close_pivot, figsize=(7, 5), diagonal='kde', alpha=0.5, color='steelblue', ax=axes)
    plt.suptitle('Scatter Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

st.markdown('**Correlation Matrix Table**')
st.dataframe(corr_matrix.round(3), use_container_width=True)

st.markdown('---')

# ── Section 6: ML Model ───────────────────────────────────────
st.markdown('### 6. 🤖 ML Model — Price Prediction')
st.info('Linear Regression and Random Forest models trained to predict next day closing price. Evaluated using R² Score and RMSE.')

results = {}
for ticker, data, color in stocks_full:
    d = data.copy()
    if len(d) < 15:
        continue
    d['Target']       = d['Close'].shift(-1)
    d['Price_Change'] = d['Close'] - d['Open']
    d['High_Low_Gap'] = d['High']  - d['Low']
    d['Prev_Close']   = d['Close'].shift(1)
    d = d.dropna()
    features = ['MA20', 'Daily_Return', 'Volume', 'Price_Change', 'High_Low_Gap', 'Prev_Close']
    X = d[features]
    y = d['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results[ticker] = {
        'lr_pred': lr_pred, 'rf_pred': rf_pred, 'y_test': y_test,
        'lr_r2':   round(r2_score(y_test, lr_pred), 3),
        'rf_r2':   round(r2_score(y_test, rf_pred), 3),
        'lr_rmse': round(np.sqrt(mean_squared_error(y_test, lr_pred)), 2),
        'rf_rmse': round(np.sqrt(mean_squared_error(y_test, rf_pred)), 2),
        'color':   color
    }

eval_rows = []
for ticker, res in results.items():
    eval_rows.append({
        'Stock':        ticker,
        'LR R2':        res['lr_r2'],
        'LR RMSE ($)':  res['lr_rmse'],
        'RF R2':        res['rf_r2'],
        'RF RMSE ($)':  res['rf_rmse'],
        'Better Model': 'Linear Regression' if res['lr_r2'] > res['rf_r2'] else 'Random Forest'
    })
st.markdown('**Model Evaluation Results**')
st.dataframe(pd.DataFrame(eval_rows), use_container_width=True)

active_results = {t: results[t] for t, d, c in active_stocks if t in results}
if active_results:
    n     = len(active_results)
    ncols = min(2, n)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
    fig.suptitle('Predicted vs Actual Closing Price', fontsize=16, fontweight='bold')
    axes_flat = [axes] if n == 1 else axes.flatten()
    for i, (ticker, res) in enumerate(active_results.items()):
        x_axis = range(len(res['y_test']))
        axes_flat[i].plot(x_axis, res['y_test'].values, color=res['color'], linewidth=2,   label='Actual')
        axes_flat[i].plot(x_axis, res['lr_pred'],       color='navy',       linewidth=1.5, linestyle='--', label='LR')
        axes_flat[i].plot(x_axis, res['rf_pred'],       color='purple',     linewidth=1.5, linestyle=':',  label='RF')
        axes_flat[i].set_title(ticker + ' | LR R2: ' + str(res['lr_r2']) + ' | RF R2: ' + str(res['rf_r2']), fontweight='bold')
        axes_flat[i].set_xlabel('Test Day')
        axes_flat[i].set_ylabel('Price (USD)')
        axes_flat[i].legend(fontsize=8)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown('---')

# ── Section 7: Key Insights ───────────────────────────────────
st.markdown('### 7. 💡 Key Insights & Summary')

col1, col2 = st.columns(2)
with col1:
    st.markdown('**📈 Performance Findings:**')
    st.markdown('- MSFT was the best performer with **+16.10%** return')
    st.markdown('- AAPL showed steady growth of **+12.23%**')
    st.markdown('- NFLX declined **-11.07%** — worst performer')
    st.markdown('- GOOG remained nearly flat at **-1.69%**')
    st.markdown('- AAPL had highest volume at **60.3M** shares/day')
    st.markdown('**🤖 ML Model Findings:**')
    st.markdown('- AAPL Linear Regression achieved excellent **R2 = 0.987**')
    st.markdown('- Limited dataset (34 training days) affected other models')
    st.markdown('- More data (1-2 years) would significantly improve results')

with col2:
    st.markdown('**🎢 Risk & Volatility Findings:**')
    st.markdown('- NFLX is the most volatile (**2.25%** std dev) — High Risk')
    st.markdown('- GOOG second most volatile (**2.07%** std dev) — High Risk')
    st.markdown('- MSFT moderate volatility (**1.79%** std dev) — Medium Risk')
    st.markdown('- AAPL most stable (**1.42%** std dev) — Low Risk')
    st.markdown('**🤝 Correlation Findings:**')
    st.markdown('- AAPL & MSFT: **0.953** — Very strongly correlated')
    st.markdown('- AAPL & GOOG: **0.902** — Strongly correlated')
    st.markdown('- NFLX & others: **~0.15–0.20** — Moves independently')
    st.markdown('- NFLX is the best diversification option in this group')

st.markdown('**📊 Final Performance Table**')
summary = []
for ticker, data, color in stocks_full:
    if len(data) > 1:
        start  = data['Close'].iloc[0]
        end    = data['Close'].iloc[-1]
        change = ((end - start) / start) * 100
        summary.append({
            'Stock':           ticker,
            'Start Price ($)': round(start, 2),
            'End Price ($)':   round(end, 2),
            '3M Return (%)':   round(change, 2),
            'Period High ($)': round(data['Close'].max(), 2),
            'Period Low ($)':  round(data['Close'].min(), 2),
            'Volatility (%)':  round(data['Daily_Return'].std(), 2),
            'Trend':           'Positive' if change > 0 else 'Negative'
        })
st.dataframe(pd.DataFrame(summary), use_container_width=True)

st.markdown('---')
st.markdown('📈 Stock Market Analysis Dashboard | Chandani Sah | Data Analyst Intern | Oracle Brain, New Baneshwor | April 2026')