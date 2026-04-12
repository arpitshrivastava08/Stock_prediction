import os
import sys
import time
import json
import threading
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
from live_runner import start_live_feed, stop_live_feed
from predictor import LivePredictor, get_predictor

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

logger = get_logger(__name__, log_file="logs/dashboard.log")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Intelligent Stock Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    .pred-card {
        background: linear-gradient(135deg, #0d1525 0%, #111827 100%);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 1.5rem;
    }
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
    .retrain-badge {
        display: inline-block;
        background: #1c1028;
        border: 1px solid #8b5cf6;
        color: #8b5cf6;
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


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "live_price":           None,
        "live_ticker":          "^NSEI",
        "last_updated":         None,
        "live_feed_started":    False,
        "update_count":         0,
        "latest_prediction":    None,   # PredictionResult dataclass instance
        "retrain_running":      False,
        "retrain_result":       None,
        "predictor_ready":      False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# CACHED COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_components():
    return {
        "loader":    StockDataLoader(),
        "fe":        FeatureEngineer(),
        "sentiment": SentimentAnalyzer(),
        "risk_mgr":  RiskManager(),
    }


@st.cache_resource
def get_live_predictor():
    """Create and initialize the live predictor (loads models from disk)."""
    predictor = LivePredictor()
    predictor.load_models()
    return predictor


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


# ─────────────────────────────────────────────────────────────────────────────
# LIVE FEED STARTUP
# ─────────────────────────────────────────────────────────────────────────────

def ensure_live_feed_running(ticker: str):
    if not st.session_state["live_feed_started"]:
        predictor = get_live_predictor()

        def on_refresh(t, raw_df):
            """Called by DataRefreshScheduler every 5 minutes."""
            # Run the live prediction in the background OS thread.
            # (Do NOT touch st.session_state here, as it raises context errors!)
            result = predictor.on_new_data(t, raw_df)
            if result:
                logger.info(f"Live prediction stored offline: {result.action_name}")

        start_live_feed(tickers=[ticker], callback=on_refresh)
        st.session_state["live_feed_started"] = True


# ─────────────────────────────────────────────────────────────────────────────
# SOCKET.IO CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def setup_autorefresh():
    if not AUTOREFRESH_AVAILABLE:
        st.sidebar.warning(
            "streamlit-autorefresh not installed.\n"
            "Auto-refresh disabled.\n"
            "Run: pip install streamlit-autorefresh"
        )
        return
    # Refresh every 60 seconds to pull the fresh data
    st_autorefresh(interval=60000, limit=None, key="data_refresh")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

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
        show_all_agents = st.checkbox("Show All Agent Votes", value=True)

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

        # ── Live Feed Status ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📡 Live Feed")
        if st.session_state.get("live_feed_started"):
            st.success("Feed running ● Auto refresh")
            
            # Show live stats from the predictor instance properly
            latest_pred = get_live_predictor().get_latest()
            time_str = latest_pred.timestamp[11:19] if latest_pred else "—"
            st.caption(f"Last background process: {time_str}")
        else:
            st.info("Feed starting...")

        if st.button("Force Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        # ── End-of-Day Retrain ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔄 End-of-Day Retrain")
        _render_retrain_sidebar(ticker)

        # ── Risk Parameters ────────────────────────────────────────────
        st.markdown("---")
        with st.expander("⚠️ Risk Parameters"):
            st.markdown(f"Stop-Loss: **{config.risk.stop_loss_pct:.0%}**")
            st.markdown(f"Max Drawdown: **{config.risk.max_drawdown_pct:.0%}**")
            st.markdown(f"Position Limit: **{config.risk.max_position_pct:.0%}**")
            st.markdown(f"Max Trades/Day: **{config.risk.max_trades_per_day}**")

        st.markdown("---")
        st.caption("Intelligent Stock Trading System v2.0")
        st.caption("RL + Sentiment + Sliding Window Retrain")

    return ticker, period, initial_balance, run_analysis, show_all_agents


def _render_retrain_sidebar(ticker: str):
    """
    Sidebar section for end-of-day retraining.
    Shows: last retrain time, new bars count, trigger button, results.
    """
    from retrain import RetrainState, SlidingWindowRetrainer

    retrain_state = RetrainState()
    last_time = retrain_state.get_last_trained_time(ticker)
    last_bar  = retrain_state.get_last_trained_bar(ticker)

    if last_time:
        st.caption(f"Last retrain: {last_time[:16].replace('T', ' ')}")
        st.caption(f"Trained on: {last_bar} bars")
    else:
        st.caption("No retrain history yet.")
        st.caption("Run initial training first.")

    # Load last backtest results if they exist
    results_path = os.path.join(config.train.models_dir, "backtest_results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                bt = json.load(f)
            updated_at = bt.get("updated_at", "")[:16].replace("T", " ")
            strategies = bt.get("strategies", {})
            with st.expander("📊 Last Backtest Results"):
                st.caption(f"As of: {updated_at}")
                for name, m in strategies.items():
                    if name == "buy_hold":
                        continue
                    st.caption(
                        f"{name}: Return={m.get('total_return',0):.1%} | "
                        f"Sharpe={m.get('sharpe_ratio',0):.2f}"
                    )
        except Exception:
            pass

    # Retrain controls
    agents_options = ["ppo", "dqn", "ddqn", "a2c"]
    selected_agents = st.multiselect(
        "Agents to retrain",
        options=agents_options,
        default=agents_options,
        key="retrain_agents_select",
    )

    retrain_timesteps = st.number_input(
        "Retrain timesteps",
        min_value=1_000,
        max_value=100_000,
        value=config.retrain.retrain_timesteps,
        step=1_000,
        key="retrain_timesteps_input",
    )

    force_retrain = st.checkbox("Force retrain (ignore bar threshold)", key="force_retrain")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        trigger_retrain = st.button(
            "🔄 Retrain Now",
            use_container_width=True,
            disabled=st.session_state.get("retrain_running", False),
            key="retrain_btn",
        )
    with col_r2:
        reload_models = st.button(
            "↺ Reload Models",
            use_container_width=True,
            key="reload_models_btn",
        )

    if reload_models:
        try:
            predictor = get_live_predictor()
            predictor.reload_models()
            st.success("Models reloaded!")
        except Exception as e:
            st.error(f"Reload failed: {e}")

    if trigger_retrain and not st.session_state.get("retrain_running"):
        st.session_state["retrain_running"] = True
        st.session_state["retrain_result"] = None

        def _run_retrain():
            try:
                retrainer = SlidingWindowRetrainer()
                result = retrainer.run(
                    ticker=ticker,
                    period="2y",
                    force=force_retrain,
                    timesteps=retrain_timesteps,
                    agents=selected_agents,
                )
                st.session_state["retrain_result"] = result

                # Auto-reload models after successful retrain
                if result.get("success"):
                    try:
                        predictor = get_live_predictor()
                        predictor.reload_models()
                    except Exception:
                        pass
            except Exception as e:
                st.session_state["retrain_result"] = {
                    "success": False,
                    "error": str(e)
                }
            finally:
                st.session_state["retrain_running"] = False

        thread = threading.Thread(target=_run_retrain, daemon=True)
        thread.start()
        st.rerun()

    # Show retrain progress / result
    if st.session_state.get("retrain_running"):
        st.info("⏳ Retraining in progress... (check logs for details)")

    result = st.session_state.get("retrain_result")
    if result:
        if result.get("success"):
            elapsed = result.get("elapsed_sec", 0)
            agents_trained = result.get("agents_trained", [])
            st.success(f"✓ Retrain complete in {elapsed:.0f}s\nAgents: {', '.join(agents_trained)}")
        else:
            err = result.get("error", "Unknown error")
            st.warning(f"⚠ {err}")


# ─────────────────────────────────────────────────────────────────────────────
# LIVE PREDICTION PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_live_prediction_panel(ticker: str, df: pd.DataFrame, sentiment_score: float):
    """
    Main live prediction display showing:
      - Latest 5-min ensemble decision (BUY / SELL / HOLD)
      - Confidence score and per-agent vote breakdown
      - Risk signal and reason
      - Today's prediction history table
    """
    st.markdown('<div class="section-header">🤖 Live 5-Min Prediction Engine</div>',
                unsafe_allow_html=True)

    # Grab highest confidence prediction stored by the background thread.
    predictor = get_live_predictor()
    latest = predictor.get_latest()

    # If no cached prediction yet, run one now directly in the UI thread
    if latest is None and not df.empty:
        with st.spinner("Running initial prediction..."):
            raw = get_components()["loader"].load(ticker, "2mo")
            if raw is not None and not raw.empty:
                latest = predictor.predict(ticker, raw)

    if latest is None:
        st.info(
            "No prediction yet. The system will predict automatically every 5 minutes "
            "once the live feed is running. Click 'Run Analysis' or wait for the next refresh."
        )
        return

    # ── Main prediction display ────────────────────────────────────────
    col_action, col_details, col_votes = st.columns([1.2, 1.5, 2])

    with col_action:
        action       = latest.final_action
        action_name  = latest.action_name
        action_class = {HOLD: "hold", BUY: "buy", SELL: "sell"}[action]
        action_emoji = {HOLD: "⏸", BUY: "🟢", SELL: "🔴"}[action]
        action_color = {HOLD: "#f59e0b", BUY: "#06d6a0", SELL: "#ef4444"}[action]

        st.markdown(f"""
        <div class="action-{action_class}">
            <div style="font-size:3rem;margin-bottom:0.5rem">{action_emoji}</div>
            <div class="action-text" style="color:{action_color}">{action_name}</div>
            <div style="font-family:JetBrains Mono;color:#94a3b8;font-size:0.85rem;margin-top:0.5rem">
                Confidence: {latest.confidence:.1%}
            </div>
            <div style="font-family:JetBrains Mono;color:#64748b;font-size:0.7rem;margin-top:0.3rem">
                {latest.timestamp[11:19]}
            </div>
        </div>""", unsafe_allow_html=True)

    with col_details:
        risk_color = {
            "OK": "#06d6a0",
            "STOP_LOSS": "#ef4444",
            "MAX_DRAWDOWN": "#ef4444",
            "POSITION_CAP": "#f59e0b",
            "OVERTRADING": "#f59e0b",
        }.get(latest.risk_signal, "#94a3b8")

        sent_color = "#06d6a0" if latest.sentiment_score > 0.1 else (
            "#ef4444" if latest.sentiment_score < -0.1 else "#f59e0b"
        )

        st.markdown(f"""
        <div class="pred-card">
            <div style="margin-bottom:0.8rem">
                <div class="metric-label">Price at Prediction</div>
                <div style="font-family:JetBrains Mono;font-size:1.3rem;color:#e2e8f0">
                    ₹{latest.current_price:,.2f}
                </div>
            </div>
            <div style="margin-bottom:0.8rem">
                <div class="metric-label">Sentiment</div>
                <div style="font-family:JetBrains Mono;font-size:1.1rem;color:{sent_color}">
                    {latest.sentiment_score:+.3f}
                </div>
            </div>
            <div style="margin-bottom:0.8rem">
                <div class="metric-label">Risk Signal</div>
                <div style="font-family:JetBrains Mono;font-size:0.9rem;color:{risk_color}">
                    {latest.risk_signal}
                </div>
            </div>
            <div style="background:#0a0e1a;border-radius:6px;padding:0.5rem;margin-top:0.5rem">
                <div style="font-family:JetBrains Mono;font-size:0.65rem;color:#475569">
                    {latest.risk_reason[:80]}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_votes:
        st.markdown("**Agent Vote Breakdown**")

        # Vote score bars (HOLD / BUY / SELL)
        vote_scores = latest.vote_scores or {}
        max_score = max(vote_scores.values(), default=1) or 1
        bar_colors = {"HOLD": "#f59e0b", "BUY": "#06d6a0", "SELL": "#ef4444"}

        for action_label, score in sorted(vote_scores.items()):
            bar_w = int(score / max_score * 180)
            color = bar_colors.get(action_label, "#94a3b8")
            is_winner = action_label == latest.action_name
            border = f"border: 1px solid {color};" if is_winner else ""
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.6rem;margin:0.3rem 0;
                background:#1a1f35;border-radius:6px;padding:0.4rem 0.6rem;{border}">
                <span style="font-family:JetBrains Mono;font-size:0.75rem;
                    color:{color};width:36px;font-weight:{'700' if is_winner else '400'}">{action_label}</span>
                <div style="background:#0d1220;border-radius:4px;flex:1;height:20px;overflow:hidden">
                    <div style="background:{color};height:100%;width:{bar_w}px;opacity:0.85"></div>
                </div>
                <span style="font-family:JetBrains Mono;font-size:0.7rem;color:#94a3b8;width:36px">
                    {score:.2f}</span>
            </div>""", unsafe_allow_html=True)

        # Per-agent votes
        if latest.agent_votes:
            st.markdown("**Per-Agent Votes**")
            agent_colors = {
                "HOLD": "#f59e0b", "BUY": "#06d6a0", "SELL": "#ef4444"
            }
            cols_per_row = 5
            agent_items = list(latest.agent_votes.items())
            for i in range(0, len(agent_items), cols_per_row):
                cols = st.columns(min(cols_per_row, len(agent_items) - i))
                for j, (aname, avote) in enumerate(agent_items[i:i + cols_per_row]):
                    an  = avote.get("action_name", "HOLD")
                    w   = avote.get("weight", 0.0)
                    col = agent_colors.get(an, "#94a3b8")
                    with cols[j]:
                        st.markdown(f"""
                        <div style="background:#0d1220;border:1px solid {col};
                            border-radius:8px;padding:0.5rem;text-align:center">
                            <div style="font-family:JetBrains Mono;font-size:0.65rem;
                                color:#64748b">{aname.upper()}</div>
                            <div style="font-family:JetBrains Mono;font-size:0.75rem;
                                color:{col};font-weight:700">{an}</div>
                            <div style="font-family:JetBrains Mono;font-size:0.6rem;
                                color:#475569">w={w:.2f}</div>
                        </div>""", unsafe_allow_html=True)

    # ── Prediction History Table ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Today's Prediction History", expanded=False):
        history_df = predictor.get_history_df()
        if not history_df.empty:
            st.dataframe(
                history_df.iloc[::-1],   # newest first
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No predictions recorded yet today.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_candlestick_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
        subplot_titles=["Price Chart", "Volume"]
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Price", increasing_line_color="#06d6a0",
            decreasing_line_color="#ef4444",
        ), row=1, col=1
    )
    if "ema_12" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_12"], name="EMA 12",
                                 line=dict(color="#3b82f6", width=1.5)), row=1, col=1)
    if "ema_26" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_26"], name="EMA 26",
                                 line=dict(color="#8b5cf6", width=1.5)), row=1, col=1)

    # Bollinger Bands
    if "bb_high" in df.columns and "bb_low" in df.columns and "bb_mid" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_high"], name="BB Upper",
            line=dict(color="#f59e0b", width=1, dash="dot"),
            opacity=0.7), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_mid"], name="BB Mid",
            line=dict(color="#f59e0b", width=1, dash="dash"),
            opacity=0.5), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_low"], name="BB Lower",
            line=dict(color="#f59e0b", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(245,158,11,0.05)",
            opacity=0.7), row=1, col=1)

    # Overlay buy/sell signals from prediction history
    predictor = get_live_predictor()
    history = predictor.get_history()
    if history:
        buy_times  = [r.timestamp for r in history if r.final_action == BUY]
        buy_prices = [r.current_price for r in history if r.final_action == BUY]
        sell_times  = [r.timestamp for r in history if r.final_action == SELL]
        sell_prices = [r.current_price for r in history if r.final_action == SELL]

        if buy_times:
            fig.add_trace(go.Scatter(
                x=buy_times, y=buy_prices, mode="markers", name="Live BUY",
                marker=dict(symbol="triangle-up", size=14, color="#06d6a0")
            ), row=1, col=1)
        if sell_times:
            fig.add_trace(go.Scatter(
                x=sell_times, y=sell_prices, mode="markers", name="Live SELL",
                marker=dict(symbol="triangle-down", size=14, color="#ef4444")
            ), row=1, col=1)

    colors = ["#06d6a0" if c >= o else "#ef4444"
              for c, o in zip(df["Close"], df["Open"])]
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
    fig.add_hline(y=70, line_dash="dot", line_color="#ef4444",
                  annotation_text="Overbought", row=1, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#06d6a0",
                  annotation_text="Oversold",   row=1, col=1)
    if "macd" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    init_session_state()
    setup_autorefresh()

    # Header
    live_badge = '<span class="live-badge">● LIVE</span>'
    update_count = st.session_state.get("update_count", 0)
    retrain_badge = ""
    if st.session_state.get("retrain_running"):
        retrain_badge = '<span class="retrain-badge">⟳ RETRAINING</span>'

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1220 0%,#1a1f35 50%,#0d1220 100%);
        border:1px solid #2a3050;border-radius:16px;padding:2rem;margin-bottom:2rem;text-align:center">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;
            background:linear-gradient(90deg,#3b82f6,#06d6a0,#8b5cf6);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.5rem">
            📈 Intelligent Stock Trading System {live_badge} {retrain_badge}
        </div>
        <div style="font-family:JetBrains Mono;color:#64748b;font-size:0.85rem">
            Reinforcement Learning • Sentiment Analysis • Ensemble • Sliding Window Retrain
        </div>
    </div>
    """, unsafe_allow_html=True)

    ticker, period, initial_balance, run_analysis, show_all_agents = render_sidebar()

    ensure_live_feed_running(ticker)

    with st.spinner("Loading market data..."):
        df = load_data(ticker, period)

    if df.empty:
        st.error(f"Could not load data for {ticker}. Check the symbol and try again.")
        return

    # Current snapshot mapping
    current_price    = float(df["Close"].iloc[-1])
    prev_price       = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
    price_change     = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100 if prev_price else 0
    current_rsi      = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50.0
    current_atr      = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
    price_source     = "LIVE 5m Auto-sync"

    with st.spinner("Analyzing sentiment..."):
        sentiment_score = get_sentiment(ticker)

    # ── TOP METRICS ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Market Overview</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        delta_sym   = "▲" if price_change >= 0 else "▼"
        delta_class = "positive" if price_change >= 0 else "negative"
        src_label   = (f'<div style="color:#06d6a0;font-size:0.65rem">{price_source}</div>'
                       if price_source == "LIVE" else "")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">₹{current_price:,.2f}</div>
            <div class="{delta_class}" style="font-size:0.8rem;font-family:JetBrains Mono">
                {delta_sym} {abs(price_change_pct):.2f}%</div>
            {src_label}
        </div>""", unsafe_allow_html=True)

    with m2:
        sent_class = "positive" if sentiment_score > 0.1 else (
            "negative" if sentiment_score < -0.1 else "neutral")
        sent_label = ("Bullish" if sentiment_score > 0.1 else
                      ("Bearish" if sentiment_score < -0.1 else "Neutral"))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sentiment Score</div>
            <div class="metric-value {sent_class}">{sentiment_score:+.3f}</div>
            <div style="color:#64748b;font-size:0.75rem">{sent_label}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        rsi_class = "negative" if current_rsi > 70 else (
            "positive" if current_rsi < 30 else "neutral")
        rsi_label = ("Overbought" if current_rsi > 70 else
                     ("Oversold" if current_rsi < 30 else "Neutral"))
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

    # ── LIVE PREDICTION PANEL ─────────────────────────────────────────
    render_live_prediction_panel(ticker, df, sentiment_score)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CHARTS ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📉 Price Charts (with Live BUY/SELL signals)</div>',
                unsafe_allow_html=True)
    st.plotly_chart(build_candlestick_chart(df), use_container_width=True)

    col_ind1, col_ind2 = st.columns([2, 1])
    with col_ind1:
        st.markdown('<div class="section-header">📊 Technical Indicators</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(build_indicator_chart(df), use_container_width=True)

    with col_ind2:
        st.markdown('<div class="section-header">📋 Latest Values</div>',
                    unsafe_allow_html=True)
        if not df.empty:
            latest_row = df.iloc[-1]
            rows = []
            for col, label in [
                ("rsi", "RSI(14)"), ("macd", "MACD"),
                ("macd_signal", "MACD Signal"),
                ("ema_12", "EMA(12)"), ("ema_26", "EMA(26)"),
                ("bb_high", "BB Upper"), ("bb_low", "BB Lower"),
                ("atr", "ATR(14)"),
            ]:
                if col in df.columns:
                    rows.append({"Indicator": label, "Value": f"{float(latest_row[col]):.2f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── DATA TABLE ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📁 Recent Data</div>',
                unsafe_allow_html=True)
    display_cols = [c for c in ["Open","High","Low","Close","Volume",
                                 "rsi","macd","atr"] if c in df.columns]
    st.dataframe(df[display_cols].tail(20).round(2), use_container_width=True)

    # ── FOOTER ────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: st.caption(f"Data: {len(df)} bars | {ticker} | {period}")
    with c2: st.caption(f"Rendered: {datetime.now().strftime('%H:%M:%S')}")
    with c3: st.caption("Not financial advice. Educational purposes only.")


if __name__ == "__main__":
    main()