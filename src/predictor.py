import os
import json
import time
import threading
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List

from logger import get_logger
from config import config
from features import FeatureEngineer, FEATURE_COLUMNS
from sentiment import SentimentAnalyzer
from env import StockTradingEnv, ACTION_NAMES, HOLD, BUY, SELL
from risk_manager import RiskManager, PortfolioState

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """
    Complete output of one 5-minute prediction cycle.
    Stored to disk and displayed in the dashboard.
    """
    timestamp: str
    ticker: str
    current_price: float
    final_action: int
    action_name: str
    confidence: float
    sentiment_score: float
    risk_signal: str
    risk_reason: str
    agent_votes: Dict
    vote_scores: Dict
    obs: List[float]
    n_bars: int


class LivePredictor:
    """
    Stateful 5-minute prediction engine.
    """

    def __init__(self, initial_balance: float = None):
        self.initial_balance = initial_balance or config.env.initial_balance
        self.fe = FeatureEngineer()
        self.sentiment = SentimentAnalyzer()
        self.risk_mgr = RiskManager()
        self.risk_mgr.reset(self.initial_balance)

        self._models: Dict[str, Any] = {}
        self._ensemble = None
        self._models_loaded = False

        self._cash_balance = self.initial_balance
        self._shares_held = 0
        self._entry_price = 0.0
        self._peak_portfolio = self.initial_balance

        self._history: List[PredictionResult] = []
        self._log_file = config.live.prediction_log_file
        os.makedirs(os.path.dirname(self._log_file), exist_ok=True)
        self._load_history()

        logger.info(
            f"LivePredictor initialized | "
            f"initial_balance={self.initial_balance:.0f} | "
            f"log_file={self._log_file}"
        )

    def load_models(self) -> bool:
        """
        Load all trained RL models from disk.
        """
        try:
            from ensemble import load_all_models, EnsembleEngine

            dummy_df = self._make_dummy_df()
            dummy_env = StockTradingEnv(dummy_df)

            self._models = load_all_models(dummy_env)

            if self._models:
                self._ensemble = EnsembleEngine(self._models)
                self._models_loaded = True
                logger.info(f"Loaded {len(self._models)} models for live prediction")
                return True
            else:
                logger.warning("No models loaded — predictions will use HOLD fallback")
                return False

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def reload_models(self):
        logger.info("Reloading models after retraining...")
        self._models_loaded = False
        self._models = {}
        self._ensemble = None
        return self.load_models()

    def on_new_data(self, ticker: str, raw_df: pd.DataFrame) -> Optional[PredictionResult]:
        return self.predict(ticker, raw_df)

    def predict(self, ticker: str, raw_df: pd.DataFrame) -> Optional[PredictionResult]:
        try:
            df = self.fe.compute_all(raw_df)
            if df.empty or len(df) < 2:
                logger.warning(f"[{ticker}] Not enough data for prediction ({len(df)} bars)")
                return None

            sentiment_score = self.sentiment.get_score(ticker)

            obs = self._build_obs(df, sentiment_score)
            current_price = float(df["Close"].iloc[-1])

            if self._models_loaded and self._ensemble is not None:
                final_action, confidence, breakdown = self._ensemble.predict(obs)
                agent_votes = breakdown.get("agent_votes", {})
                vote_scores = breakdown.get("vote_scores", {})
            else:
                final_action = HOLD
                confidence = 1.0
                agent_votes = {}
                vote_scores = {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0}
                logger.warning(f"[{ticker}] No models loaded, defaulting to HOLD")

            portfolio_value = self._cash_balance + self._shares_held * current_price
            if portfolio_value > self._peak_portfolio:
                self._peak_portfolio = portfolio_value

            portfolio_state = PortfolioState(
                cash_balance=self._cash_balance,
                shares_held=self._shares_held,
                current_price=current_price,
                entry_price=self._entry_price,
                peak_portfolio=self._peak_portfolio,
                trades_today=self.risk_mgr.trades_today,
                session_start_value=self.initial_balance,
            )

            safe_action, risk_signal, risk_reason = self.risk_mgr.apply(
                final_action, portfolio_state
            )

            self._simulate_action(safe_action, current_price)

            result = PredictionResult(
                timestamp=datetime.now().isoformat(),
                ticker=ticker,
                current_price=current_price,
                final_action=safe_action,
                action_name=ACTION_NAMES[safe_action],
                confidence=confidence,
                sentiment_score=sentiment_score,
                risk_signal=risk_signal.value,
                risk_reason=risk_reason,
                agent_votes={
                    k: {
                        "action_name": v.get("action_name", "HOLD"),
                        "weight": v.get("weight", 0.0),
                    }
                    for k, v in agent_votes.items()
                },
                vote_scores=vote_scores,
                obs=obs.tolist(),
                n_bars=len(df),
            )

            self._history.append(result)
            self._save_history()

            logger.info(
                f"[{ticker}] PREDICTION → {result.action_name} | "
                f"Price: {current_price:.2f} | "
                f"Confidence: {confidence:.2%} | "
                f"Sentiment: {sentiment_score:+.3f} | "
                f"Risk: {result.risk_signal}"
            )

            return result

        except Exception as e:
            logger.error(f"[{ticker}] Prediction failed: {e}", exc_info=True)
            return None

    def _build_obs(self, df: pd.DataFrame, sentiment_score: float) -> np.ndarray:
        sentiment_col = [sentiment_score] * len(df)
        temp_env = StockTradingEnv(df, sentiment_col)
        temp_env.current_step = len(df) - 1
        return temp_env._get_obs()

    def _simulate_action(self, action: int, price: float):
        if action == BUY and self._cash_balance >= price:
            cost = price * (1 + config.env.transaction_cost)
            if self._cash_balance >= cost:
                self._shares_held += 1
                self._cash_balance -= cost
                self._entry_price = price

        elif action == SELL and self._shares_held > 0:
            proceeds = price * (1 - config.env.transaction_cost)
            self._shares_held -= 1
            self._cash_balance += proceeds
            if self._shares_held == 0:
                self._entry_price = 0.0

    def _load_history(self):
        if not os.path.exists(self._log_file):
            self._history = []
            return

        try:
            with open(self._log_file, "r") as f:
                raw = json.load(f)

            if not isinstance(raw, list):
                logger.warning("Prediction log is not a list. Resetting in-memory history.")
                self._history = []
                return

            loaded: List[PredictionResult] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                try:
                    loaded.append(PredictionResult(**item))
                except TypeError:
                    # Skip malformed historical records to keep live service running.
                    continue

            self._history = loaded
            logger.info(f"Loaded {len(self._history)} historical predictions")

        except Exception as e:
            logger.warning(f"Failed to load prediction history: {e}")
            self._history = []

    def _save_history(self):
        try:
            with open(self._log_file, "w") as f:
                json.dump([asdict(r) for r in self._history], f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save prediction history: {e}")

    def _make_dummy_df(self, n_rows: int = 300) -> pd.DataFrame:
        prices = np.linspace(100.0, 120.0, n_rows)
        idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n_rows, freq="5min")

        raw = pd.DataFrame(
            {
                "Open": prices,
                "High": prices * 1.002,
                "Low": prices * 0.998,
                "Close": prices,
                "Volume": np.full(n_rows, 100000, dtype=np.int64),
            },
            index=idx,
        )

        engineered = self.fe.compute_all(raw)
        if engineered.empty:
            fallback = pd.DataFrame(
                {col: np.zeros(n_rows, dtype=np.float32) for col in FEATURE_COLUMNS}
            )
            fallback["Close"] = prices
            return fallback
        return engineered

    def get_latest(self) -> Optional[PredictionResult]:
        if not self._history:
            return None
        return self._history[-1]

    def get_history(self) -> List[PredictionResult]:
        return list(self._history)

    def get_history_df(self) -> pd.DataFrame:
        if not self._history:
            return pd.DataFrame()

        rows = [asdict(r) for r in self._history]
        df = pd.DataFrame(rows)
        preferred = [
            "timestamp",
            "ticker",
            "current_price",
            "action_name",
            "confidence",
            "sentiment_score",
            "risk_signal",
            "risk_reason",
        ]
        cols = [c for c in preferred if c in df.columns]
        return df[cols] if cols else df


_predictor_instance: Optional[LivePredictor] = None
_predictor_lock = threading.Lock()


def get_predictor(load_models: bool = True) -> LivePredictor:
    """Return a process-level singleton predictor for compatibility with app/retrain imports."""
    global _predictor_instance

    with _predictor_lock:
        if _predictor_instance is None:
            _predictor_instance = LivePredictor()
            if load_models:
                _predictor_instance.load_models()
        elif load_models and not _predictor_instance._models_loaded:
            _predictor_instance.load_models()

    return _predictor_instance