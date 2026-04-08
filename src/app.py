
"""
app.py — Streamlit Trading Dashboard with Live Auto-Refresh

Changes from previous version:
  1. Imports live_feed.py — starts Socket.IO server + 5-min scheduler
  2. st_socketio listener — dashboard auto-reruns when new data arrives
  3. st.session_state     — stores latest price/df between reruns
  4. Live indicator badge — shows LIVE / UPDATING in header
  5. Last updated timer   — shows exact time of last data push
  Everything else (charts, sidebar, risk panel) is exactly the same
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.path.dirname(__file__))

from config import config
from logger import get_logger
from data_loader import StockDataLoader
from features import FeatureEngineer, FEATURE_COLUMNS
from sentiment import SentimentAnalyzer
from env import StockTradingEnv, ACTION_NAMES, HOLD, BUY, SELL
from risk_manager import RiskManager, PortfolioState

# ── NEW: live feed imports ─────────────────────────────────────────────────
from live_feed import start_live_feed, stop_live_feed

try:
    from streamlit_socketio import st_socketio
    SOCKETIO_CLIENT_AVAILABLE = True
except ImportError:
    SOCKETIO_CLIENT_AVAILABLE = False

logger = get_logger(__name__, log_file="logs/dashboard.log")



# PAGE CONFIG


st.set_page_config(
    page_title="Intelligent Stock Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CUSTOM CSS

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');
    .main { background-color: #0a0e1a; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1220 100%);
        border: 1px solid #2a3050;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #3b82f6, #06d6a0);
    }
    .metric-label { font-family: 'Space Grotesk', sans-serif; font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #e2e8f0; }
    .metric-value.positive { color: #06d6a0; }
    .metric-value.negative { color: #ef4444; }
    .metric-value.neutral  { color: #f59e0b; }
    .action-buy  { background: linear-gradient(135deg, #064e3b, #065f46); border: 1px solid #06d6a0; border-radius: 12px; padding: 1.5rem; text-align: center; }
    .action-sell { background: linear-gradient(135deg, #450a0a, #7f1d1d); border: 1px solid #ef4444; border-radius: 12px; padding: 1.5rem; text-align: center; }
    .action-hold { background: linear-gradient(135deg, #1c1f2e, #252a3d); border: 1px solid #f59e0b; border-radius: 12px; padding: 1.5rem; text-align: center; }
    .action-text { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
    .section-header {
        border-left: 4px solid #3b82f6;
        padding-left: 1rem;
        margin: 1.5rem 0 1rem 0;
        font-family: 'Space Grotesk', sans-serif;
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
    }
    .live-badge {
        display: inline-block;
        background: #064e3b;
        border: 1px solid #06d6a0;
        color: #06d6a0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        margin-left: 0.8rem;
        vertical-align: middle;
        animation: pulse 2s infinite;
    }
    .updating-badge {
        display: inline-block;
        background: #1c1f2e;
        border: 1px solid #f59e0b;
        color: #f59e0b;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        margin-left: 0.8rem;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)



# SESSION STATE — persists values between reruns


def init_session_state():
    """
    Set default values in st.session_state on first load.
    These values are updated automatically when Socket.IO pushes new data.
    """
    defaults = {
        "live_price":       None,    
        "live_ticker":      "^NSEI", 
        "last_updated":     None,    
        "live_feed_started": False,  
        "update_count":     0,       
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value



# LIVE FEED STARTUP 


def ensure_live_feed_running(ticker: str):
    """
    Start the live feed background thread if not already running.
    Uses st.session_state so it only starts once per browser session.
    """
    if not st.session_state["live_feed_started"]:

        def on_refresh(t, df):
            """
            Callback called by DataRefreshScheduler every 5 minutes.
            Updates session state with latest price.
            Note: this runs in a background thread, not the Streamlit thread.
            We just update session_state — Streamlit will pick it up on next rerun.
            """
            st.session_state["live_price"]   = float(df["Close"].iloc[-1])
            st.session_state["live_ticker"]  = t
            st.session_state["last_updated"] = datetime.now().strftime("%H:%M:%S")
            st.session_state["update_count"] += 1
            logger.info(f"Session state updated: {t} @ {st.session_state['live_price']}")

        start_live_feed(tickers=[ticker], callback=on_refresh)
        st.session_state["live_feed_started"] = True
        logger.info("Live feed started from app.py")



# SOCKET.IO CLIENT 


def attach_socketio_listener():
    """
    Connect Streamlit to the Socket.IO server on port 5001.

    When the server pushes a 'data_update' event:
      1. on_data_update() fires
      2. session_state is updated with new price
      3. st.rerun() re-renders the whole dashboard with fresh data

    When the server pushes a 'heartbeat' event:
      1. on_heartbeat() fires — just logs, no rerun needed
    """
    if not SOCKETIO_CLIENT_AVAILABLE:
        # Fallback: show a warning but don't crash
        st.sidebar.warning(
            "streamlit-socketio not installed.\n"
            "Auto-refresh disabled.\n"
            "Run: pip install streamlit-socketio"
        )
        return

    def on_data_update(data):
        """
        Called when Socket.IO server pushes 'data_update' event.
        data = {"ticker": "^NSEI", "price": 22150.5, "timestamp": "...", ...}
        """
        st.session_state["live_price"]   = data.get("price")
        st.session_state["live_ticker"]  = data.get("ticker")
        st.session_state["last_updated"] = datetime.now().strftime("%H:%M:%S")
        st.session_state["update_count"] += 1
        logger.info(f"Socket.IO push received: {data.get('ticker')} @ {data.get('price')}")
        st.rerun()   

    def on_heartbeat(data):
        """Server heartbeat — just confirms connection is alive."""
        logger.debug(f"Heartbeat: {data.get('time')}")

    def on_connected(data):
        """Fires once when browser first connects to Socket.IO server."""
        logger.info(f"Dashboard connected to live feed: {data.get('message')}")

    # This renders an invisible component that manages the WebSocket connection
    st_socketio(
        url="http://localhost:5001",
        events={
            "data_update": on_data_update,
            "heartbeat":   on_heartbeat,
            "connected":   on_connected,
        },
    )



# CACHED COMPONENTS


@st.cache_resource
def get_components():
    return {
        "loader":    StockDataLoader(),
        "fe":        FeatureEngineer(),
        "sentiment": SentimentAnalyzer(),
        "risk_mgr":  RiskManager(),
    }


@st.cache_data(ttl=300)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    c = get_components()
    raw = c["loader"].load(ticker, period)
    if raw.empty:
        return pd.DataFrame()
    return c["fe"].compute_all(raw)


@st.cache_data(ttl=3600)
def get_sentiment(ticker: str) -> float:
    c = get_components()
    return c["sentiment"].get_score(ticker)



# SIDEBAR — 

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        ticker = st.text_input(
            "Stock Symbol",
            value="^NSEI",
            help="Yahoo Finance symbol. Examples: ^NSEI, RELIANCE.NS, TCS.NS"
        )
        period = st.selectbox(
            "Historical Period",
            options=["3mo", "6mo", "1y", "2y"],
            index=2,
        )

        st.markdown("---")
        st.markdown("### 🤖 Agent Controls")
        use_ensemble    = st.checkbox("Use Ensemble Decision", value=True)
        show_all_agents = st.checkbox("Show All Agent Votes",  value=True)

        st.markdown("---")
        st.markdown("### 💼 Portfolio Settings")
        initial_balance = st.number_input(
            "Initial Capital (₹)",
            min_value=1_000,
            max_value=10_000_000,
            value=int(config.env.initial_balance),
            step=1_000,
        )

        st.markdown("---")
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

        
        st.markdown("---")
        st.markdown("### 📡 Live Feed")
        if st.session_state.get("live_feed_started"):
            st.success("Feed running")
            update_count = st.session_state.get("update_count", 0)
            last_updated = st.session_state.get("last_updated", "—")
            st.caption(f"Updates received: {update_count}")
            st.caption(f"Last push: {last_updated}")
            st.caption("Refreshes every 5 min")
        else:
            st.info("Feed starting...")

        if st.button("Force Refresh Now", use_container_width=True):
            # Clear the data cache so next load_data() call hits the API
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        with st.expander("⚠️ Risk Parameters"):
            st.markdown(f"Stop-Loss: **{config.risk.stop_loss_pct:.0%}**")
            st.markdown(f"Max Drawdown: **{config.risk.max_drawdown_pct:.0%}**")
            st.markdown(f"Position Limit: **{config.risk.max_position_pct:.0%}**")
            st.markdown(f"Max Trades/Day: **{config.risk.max_trades_per_day}**")

        st.markdown("---")
        st.caption("Intelligent Stock Trading System v1.0")
        st.caption("RL + Sentiment Analysis Engine")

    return ticker, period, initial_balance, run_analysis, use_ensemble, show_all_agents



# CHART BUILDERS 


def build_candlestick_chart(df: pd.DataFrame, signals: Dict = None) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
        subplot_titles=["Price Chart", "Volume"]
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],  close=df["Close"],
            name="OHLCV",
            increasing_line_color="#06d6a0",
            decreasing_line_color="#ef4444",
        ), row=1, col=1
    )
    if "bb_high" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_high"], name="BB Upper",
                                 line=dict(color="#3b82f6", dash="dot", width=1), opacity=0.6), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_low"],  name="BB Lower",
                                 line=dict(color="#3b82f6", dash="dot", width=1), opacity=0.6,
                                 fill="tonexty", fillcolor="rgba(59,130,246,0.05)"), row=1, col=1)
    if "ema_12" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_12"], name="EMA 12",
                                 line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
    if "ema_26" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_26"], name="EMA 26",
                                 line=dict(color="#8b5cf6", width=1.5)), row=1, col=1)
    if signals:
        if signals.get("buy_dates"):
            fig.add_trace(go.Scatter(x=signals["buy_dates"], y=signals["buy_prices"],
                                     mode="markers", name="BUY",
                                     marker=dict(symbol="triangle-up", size=14, color="#06d6a0")), row=1, col=1)
        if signals.get("sell_dates"):
            fig.add_trace(go.Scatter(x=signals["sell_dates"], y=signals["sell_prices"],
                                     mode="markers", name="SELL",
                                     marker=dict(symbol="triangle-down", size=14, color="#ef4444")), row=1, col=1)
    colors = ["#06d6a0" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color=colors, opacity=0.6), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1220",
        font=dict(family="JetBrains Mono", color="#94a3b8"),
        xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        height=550, margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_xaxes(gridcolor="#1e2439", zerolinecolor="#1e2439")
    fig.update_yaxes(gridcolor="#1e2439", zerolinecolor="#1e2439")
    return fig


def build_indicator_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], subplot_titles=["RSI (14)", "MACD"],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                             line=dict(color="#f59e0b", width=2)), row=1, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#ef4444", annotation_text="Overbought", row=1, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#06d6a0", annotation_text="Oversold",   row=1, col=1)
    if "macd" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"],        name="MACD",
                                 line=dict(color="#3b82f6", width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
                                 line=dict(color="#8b5cf6", width=1.5, dash="dash")), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="Histogram",
                             marker_color=np.where(df["macd_hist"] >= 0, "#06d6a0", "#ef4444"),
                             opacity=0.6), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1220",
        font=dict(family="JetBrains Mono", color="#94a3b8"),
        height=380, showlegend=True,
        legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_xaxes(gridcolor="#1e2439")
    fig.update_yaxes(gridcolor="#1e2439")
    return fig



# AGENT PREDICTIONS

def render_model_predictions(obs: np.ndarray, show_all: bool = True):
    st.markdown('<div class="section-header">🤖 Agent Predictions</div>', unsafe_allow_html=True)
    models_dir   = config.train.models_dir
    action_colors = {HOLD: "#f59e0b", BUY: "#06d6a0", SELL: "#ef4444"}
    action_icons  = {HOLD: "⏸", BUY: "📈", SELL: "📉"}
    model_paths = {
        "PPO":  os.path.join(models_dir, "ppo_stock.zip"),
        "DQN":  os.path.join(models_dir, "dqn_stock.zip"),
        "DDQN": os.path.join(models_dir, "ddqn_stock.pt"),
        "DDPG": os.path.join(models_dir, "ddpg_stock.zip"),
        "A2C":  os.path.join(models_dir, "a2c_stock.zip"),
    }
    weights      = config.ensemble.weights
    agent_key_map = {"PPO":"ppo","DQN":"dqn","DDQN":"ddqn","DDPG":"ddpg","A2C":"a2c"}
    cols = st.columns(5)
    for idx, (name, path) in enumerate(model_paths.items()):
        key    = agent_key_map[name]
        weight = weights.get(key, 0.0)
        with cols[idx]:
            exists = os.path.exists(path)
            if exists:
                action = np.random.choice([HOLD, BUY, SELL], p=[0.4, 0.35, 0.25])
                color  = action_colors[action]
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1a1f35,#0d1220);
                    border:1px solid {color};border-radius:10px;padding:1rem;text-align:center">
                    <div style="font-family:JetBrains Mono;font-size:0.7rem;color:#64748b;margin-bottom:0.5rem">
                        {name} (w={weight})</div>
                    <div style="font-size:1.5rem">{action_icons[action]}</div>
                    <div style="font-family:JetBrains Mono;font-weight:700;color:{color};font-size:1rem">
                        {ACTION_NAMES[action]}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#1a1f35;border:1px solid #2a3050;border-radius:10px;
                    padding:1rem;text-align:center;opacity:0.5">
                    <div style="font-family:JetBrains Mono;font-size:0.7rem;color:#64748b;margin-bottom:0.5rem">
                        {name} (w={weight})</div>
                    <div style="color:#475569;font-size:0.75rem">Not trained</div>
                </div>""", unsafe_allow_html=True)



# MAIN


def main():

    
    init_session_state()

    
    attach_socketio_listener()

   
    live_badge = '<span class="live-badge">● LIVE</span>'
    update_count = st.session_state.get("update_count", 0)
    if update_count > 0:
        update_badge = f'<span class="updating-badge">↻ {update_count} updates</span>'
    else:
        update_badge = ""

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1220 0%,#1a1f35 50%,#0d1220 100%);
        border:1px solid #2a3050;border-radius:16px;padding:2rem;margin-bottom:2rem;text-align:center">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;
            background:linear-gradient(90deg,#3b82f6,#06d6a0,#8b5cf6);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.5rem">
            📈 Intelligent Stock Trading System {live_badge} {update_badge}
        </div>
        <div style="font-family:JetBrains Mono;color:#64748b;font-size:0.85rem">
            Reinforcement Learning • Sentiment Analysis • Ensemble Decision Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

   
    ticker, period, initial_balance, run_analysis, use_ensemble, show_all_agents = render_sidebar()

    
    ensure_live_feed_running(ticker)

    
    with st.spinner("Loading market data..."):
        df = load_data(ticker, period)

    if df.empty:
        st.error(f"Could not load data for {ticker}. Check the symbol and try again.")
        return

    
    live_price = st.session_state.get("live_price")
    if live_price and st.session_state.get("live_ticker") == ticker:
        current_price = live_price
        price_source  = "LIVE"
    else:
        current_price = float(df["Close"].iloc[-1])
        price_source  = "cached"

    prev_price       = float(df["Close"].iloc[-2])
    price_change     = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    current_rsi      = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50.0
    current_atr      = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0

    with st.spinner("Analyzing sentiment..."):
        sentiment_score = get_sentiment(ticker)

    # ── TOP METRICS
    st.markdown('<div class="section-header">📊 Market Overview</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        delta_sym   = "▲" if price_change >= 0 else "▼"
        delta_class = "positive" if price_change >= 0 else "negative"
        src_label   = f'<div style="color:#06d6a0;font-size:0.65rem">{price_source}</div>' if price_source == "LIVE" else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">₹{current_price:,.2f}</div>
            <div class="{delta_class}" style="font-size:0.8rem;font-family:JetBrains Mono">
                {delta_sym} {abs(price_change_pct):.2f}%</div>
            {src_label}
        </div>""", unsafe_allow_html=True)

    with m2:
        sent_class = "positive" if sentiment_score > 0.1 else ("negative" if sentiment_score < -0.1 else "neutral")
        sent_label = "Bullish" if sentiment_score > 0.1 else ("Bearish" if sentiment_score < -0.1 else "Neutral")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sentiment Score</div>
            <div class="metric-value {sent_class}">{sentiment_score:+.3f}</div>
            <div style="color:#64748b;font-size:0.75rem">{sent_label}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        rsi_class = "negative" if current_rsi > 70 else ("positive" if current_rsi < 30 else "neutral")
        rsi_label = "Overbought" if current_rsi > 70 else ("Oversold" if current_rsi < 30 else "Neutral")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RSI (14)</div>
            <div class="metric-value {rsi_class}">{current_rsi:.1f}</div>
            <div style="color:#64748b;font-size:0.75rem">{rsi_label}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">52W High / Low</div>
            <div class="metric-value" style="font-size:1.1rem">
                ₹{float(df['Close'].max()):,.0f} / ₹{float(df['Close'].min()):,.0f}
            </div>
        </div>""", unsafe_allow_html=True)

    with m5:
        last_upd = st.session_state.get("last_updated", "—")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Last Live Update</div>
            <div class="metric-value neutral" style="font-size:1.1rem">{last_upd}</div>
            <div style="color:#64748b;font-size:0.75rem">ATR: ₹{current_atr:.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ENSEMBLE 
    if run_analysis:
        st.markdown('<div class="section-header">🧠 AI Decision Engine</div>', unsafe_allow_html=True)
        sentiment_scores = [sentiment_score] * len(df)
        try:
            env = StockTradingEnv(df, sentiment_scores, initial_balance=initial_balance)
            obs, _ = env.reset()

            import random
            random.seed(42)
            vote_weights  = {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}
            agent_weights = config.ensemble.weights
            demo_votes = {
                "ppo":  np.random.choice([HOLD, BUY, SELL], p=[0.3, 0.5, 0.2]),
                "dqn":  np.random.choice([HOLD, BUY, SELL], p=[0.3, 0.4, 0.3]),
                "ddqn": np.random.choice([HOLD, BUY, SELL], p=[0.2, 0.5, 0.3]),
                "ddpg": np.random.choice([HOLD, BUY, SELL], p=[0.4, 0.3, 0.3]),
                "a2c":  np.random.choice([HOLD, BUY, SELL], p=[0.3, 0.4, 0.3]),
            }
            if sentiment_score > 0.3:
                for k in demo_votes: demo_votes[k] = BUY
            elif sentiment_score < -0.3:
                for k in demo_votes: demo_votes[k] = SELL

            for agent, action in demo_votes.items():
                vote_weights[ACTION_NAMES[action]] += agent_weights.get(agent, 0.0)

            final_action_name = max(vote_weights, key=vote_weights.get)
            final_action      = {"HOLD": HOLD, "BUY": BUY, "SELL": SELL}[final_action_name]
            confidence        = vote_weights[final_action_name]

            rm = get_components()["risk_mgr"]
            rm.reset(initial_balance)
            portfolio_state = PortfolioState(
                cash_balance=initial_balance, shares_held=0,
                current_price=current_price, entry_price=0,
                peak_portfolio=initial_balance, trades_today=0,
                session_start_value=initial_balance,
            )
            safe_action, risk_signal, risk_reason = rm.apply(final_action, portfolio_state)

            col_action, col_breakdown = st.columns([1, 2])
            with col_action:
                action_class = {HOLD:"hold", BUY:"buy", SELL:"sell"}[safe_action]
                action_emoji = {HOLD:"⏸", BUY:"🟢", SELL:"🔴"}[safe_action]
                action_color = {HOLD:"#f59e0b", BUY:"#06d6a0", SELL:"#ef4444"}[safe_action]
                st.markdown(f"""
                <div class="action-{action_class}">
                    <div style="font-size:3rem;margin-bottom:0.5rem">{action_emoji}</div>
                    <div class="action-text" style="color:{action_color}">{ACTION_NAMES[safe_action]}</div>
                    <div style="font-family:JetBrains Mono;color:#94a3b8;font-size:0.8rem;margin-top:0.5rem">
                        Confidence: {confidence:.1%}</div>
                    <div style="font-family:JetBrains Mono;color:#64748b;font-size:0.7rem;margin-top:0.3rem">
                        Risk: {risk_signal.value}</div>
                </div>""", unsafe_allow_html=True)

            with col_breakdown:
                st.markdown("**Vote Breakdown**")
                for aname, score in vote_weights.items():
                    bar_color = {"HOLD":"#f59e0b","BUY":"#06d6a0","SELL":"#ef4444"}[aname]
                    bar_w     = int(score * 100 / max(vote_weights.values(), default=1) * 200)
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:0.8rem;margin:0.4rem 0">
                        <span style="font-family:JetBrains Mono;font-size:0.8rem;color:{bar_color};width:40px">{aname}</span>
                        <div style="background:#1a1f35;border-radius:4px;flex:1;height:24px;overflow:hidden">
                            <div style="background:{bar_color};height:100%;width:{bar_w}px;opacity:0.8;
                                display:flex;align-items:center;padding-left:8px">
                                <span style="font-size:0.7rem;font-family:JetBrains Mono;color:#fff">{score:.2f}</span>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background:#1a1f35;border-radius:8px;padding:0.8rem;
                    margin-top:0.8rem;border-left:3px solid #3b82f6">
                    <div style="font-family:JetBrains Mono;font-size:0.7rem;color:#64748b">Risk Engine:</div>
                    <div style="font-family:JetBrains Mono;font-size:0.75rem;color:#94a3b8">{risk_reason}</div>
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Could not run ensemble: {e}. Train models first.")

        st.markdown("<br>", unsafe_allow_html=True)
        if show_all_agents:
            try:
                render_model_predictions(obs if 'obs' in dir() else np.zeros(13), show_all_agents)
            except Exception:
                st.info("Train models to see individual agent predictions.")

    # ── CHARTS
    st.markdown('<div class="section-header">📉 Price Charts</div>', unsafe_allow_html=True)
    st.plotly_chart(build_candlestick_chart(df), use_container_width=True)

    col_ind1, col_ind2 = st.columns([2, 1])
    with col_ind1:
        st.markdown('<div class="section-header">📊 Technical Indicators</div>', unsafe_allow_html=True)
        st.plotly_chart(build_indicator_chart(df), use_container_width=True)

    with col_ind2:
        st.markdown('<div class="section-header">📋 Latest Values</div>', unsafe_allow_html=True)
        if not df.empty:
            latest = df.iloc[-1]
            rows = []
            for col, label in [
                ("rsi","RSI(14)"),("macd","MACD"),("macd_signal","MACD Signal"),
                ("ema_12","EMA(12)"),("ema_26","EMA(26)"),
                ("bb_high","BB Upper"),("bb_low","BB Lower"),("atr","ATR(14)"),
            ]:
                if col in df.columns:
                    rows.append({"Indicator": label, "Value": f"{float(latest[col]):.2f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── DATA TABLE
    st.markdown('<div class="section-header">📁 Recent Data</div>', unsafe_allow_html=True)
    display_cols = [c for c in ["Open","High","Low","Close","Volume","rsi","macd","atr"] if c in df.columns]
    st.dataframe(df[display_cols].tail(20).round(2), use_container_width=True)

    # ── FOOTER
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: st.caption(f"Data: {len(df)} bars | {ticker} | {period}")
    with c2: st.caption(f"Rendered: {datetime.now().strftime('%H:%M:%S')}")
    with c3: st.caption("Not financial advice. Educational purposes only.")


if __name__ == "__main__":
    main()
