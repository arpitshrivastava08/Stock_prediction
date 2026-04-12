"""
config.py — Centralized configuration for the Intelligent Stock Trading System
All tunable parameters in one place. No magic numbers elsewhere.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────
# DATA CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class DataConfig:
    default_ticker: str = "^NSEI"
    default_symbols: List[str] = field(default_factory=lambda: [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"
    ])
    default_period: str = "2y"
    default_interval: str = "5min"
    csv_dir: str = "data/"
    api_timeout: int = 30


# ─────────────────────────────────────────────
# FEATURE ENGINEERING CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class FeatureConfig:
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ema_short: int = 12
    ema_long: int = 26
    bb_window: int = 20
    bb_std: int = 2
    atr_window: int = 14


# ─────────────────────────────────────────────
# SENTIMENT ANALYSIS CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class SentimentConfig:
    model_name: str = "ProsusAI/finbert"
    newsapi_key: str = os.getenv("NEWSAPI_KEY", "")
    max_articles: int = 20
    refresh_minutes: int = 60
    cache_file: str = "data/sentiment_cache.json"


# ─────────────────────────────────────────────
# RL ENVIRONMENT CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class EnvConfig:
    initial_balance: float = 10_000.0
    transaction_cost: float = 0.001
    overtrading_penalty: float = 0.002
    max_trades_per_episode: int = 10   # ✅ updated
    reward_scaling: float = 1e-4


# ─────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class TrainConfig:
    timesteps: int = 600_000   # ✅ updated
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    models_dir: str = "models/"

    # DQN specific
    dqn_buffer_size: int = 50_000
    dqn_exploration_fraction: float = 0.2
    dqn_target_update_interval: int = 1_000

    # Double DQN specific
    ddqn_buffer_size: int = 50_000
    ddqn_tau: float = 0.005

    # PPO specific
    ppo_n_steps: int = 2_048
    ppo_n_epochs: int = 10
    ppo_clip_range: float = 0.2

    # A2C specific
    a2c_n_steps: int = 5


# ─────────────────────────────────────────────
# ENSEMBLE CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class EnsembleConfig:
    weights: dict = field(default_factory=lambda: {
        "ppo":  0.30,
        "dqn":  0.20,
        "ddqn": 0.20,
        "a2c":  0.30,
    })


# ─────────────────────────────────────────────
# RISK MANAGEMENT CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class RiskConfig:
    stop_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    max_position_pct: float = 0.10
    max_trades_per_day: int = 5


# ─────────────────────────────────────────────
# BACKTESTING CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class BacktestConfig:
    train_split: float = 0.80
    min_sharpe: float = 0.5
    target_sharpe: float = 1.0
    risk_free_rate: float = 0.06


# ─────────────────────────────────────────────
# LIVE PREDICTION CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class LiveConfig:
    prediction_interval_sec: int = 300
    prediction_log_file: str = "data/predictions_log.json"
    obs_lookback: int = 1


# ─────────────────────────────────────────────
# SLIDING WINDOW RETRAINING CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class RetrainConfig:
    min_new_bars: int = 20
    window_size: int = 2_000
    retrain_timesteps: int = 20_000
    retrain_agents: List[str] = field(default_factory=lambda: [
        "ppo", "dqn", "ddqn", "a2c"
    ])
    state_file: str = "data/retrain_state.json"
    update_weights_after_retrain: bool = True


# ─────────────────────────────────────────────
# MASTER CONFIG — import this everywhere
# ─────────────────────────────────────────────
class Config:
    data = DataConfig()
    features = FeatureConfig()
    sentiment = SentimentConfig()
    env = EnvConfig()
    train = TrainConfig()
    ensemble = EnsembleConfig()
    risk = RiskConfig()
    backtest = BacktestConfig()
    live = LiveConfig()
    retrain = RetrainConfig()

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/trading_system.log"


# Singleton
config = Config()