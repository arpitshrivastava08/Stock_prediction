import os
import json
import time
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