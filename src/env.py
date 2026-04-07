"""
env.py — Custom Gymnasium Trading Environment (Markov Decision Process)

MDP Specification:
  State  (S): 13-dim vector = [price, rsi, macd, macd_signal, macd_hist,
                                ema_12, ema_26, bb_high, bb_mid, bb_low,
                                bb_width, atr, sentiment_score]
  Action (A): Discrete(3) — 0=Hold, 1=Buy, 2=Sell
  Reward (R): Portfolio value change - overtrading penalty + trend reward
  Done   (D): All timesteps exhausted

Design decisions:
  - Single share at a time for simplicity / interpretability
  - Transaction costs applied on every trade (0.1%)
  - Overtrading penalty deducted when > N trades in episode
  - Trend reward: bonus for holding a position in direction of EMA trend
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List

from logger import get_logger
from config import config
from features import FEATURE_COLUMNS

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ACTION CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
HOLD = 0
BUY  = 1
SELL = 2
ACTION_NAMES = {HOLD: "HOLD", BUY: "BUY", SELL: "SELL"}


class StockTradingEnv(gym.Env):
    """
    Gymnasium environment for single-stock RL trading.

    Observation (state vector):
      Normalized values of all technical indicators + sentiment score.
      Shape: (STATE_DIM,)  — 13 dimensions as per SRS Section 3.3.2

    Actions:
      0 = HOLD  — Do nothing
      1 = BUY   — Purchase 1 share (if sufficient balance)
      2 = SELL  — Sell 1 share (if holding any)

    Reward function:
      r = Δportfolio_value
        - transaction_cost  (if buy or sell)
        - overtrading_penalty (if > max_trades in episode)
        + trend_bonus (if holding position in direction of EMA trend)

    Episode terminates when all timesteps in the DataFrame are exhausted.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        df: pd.DataFrame,
        sentiment_scores: Optional[List[float]] = None,
        initial_balance: float = None,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            df: DataFrame with OHLCV + computed indicators
                Must contain all FEATURE_COLUMNS
            sentiment_scores: List of sentiment values aligned with df rows.
                              If None, all set to 0.0 (neutral).
            initial_balance: Starting capital in currency units
            render_mode: 'human' for console output
        """
        super().__init__()

        self.cfg = config.env
        self.render_mode = render_mode

        # ── Data ──────────────────────────────────────────────────────
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        if sentiment_scores is None:
            self.sentiment_scores = [0.0] * self.n_steps
        else:
            assert len(sentiment_scores) == self.n_steps, (
                f"sentiment_scores length {len(sentiment_scores)} "
                f"must match df length {self.n_steps}"
            )
            self.sentiment_scores = sentiment_scores

        # ── State / Action Spaces ─────────────────────────────────────
        self.state_dim = len(FEATURE_COLUMNS) + 1  # features + sentiment
        self.action_space = spaces.Discrete(3)     # HOLD, BUY, SELL
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        # ── Portfolio State ───────────────────────────────────────────
        self.initial_balance = initial_balance or self.cfg.initial_balance
        self.balance: float = self.initial_balance
        self.shares_held: int = 0
        self.entry_price: float = 0.0
        self.current_step: int = 0
        self.trade_count: int = 0
        self.portfolio_history: List[float] = []
        self.trade_log: List[dict] = []

        # Normalization stats (computed once from training data)
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None
        self._compute_normalization()

        logger.info(
            f"StockTradingEnv initialized | "
            f"Steps: {self.n_steps} | "
            f"Initial balance: {self.initial_balance:.0f} | "
            f"State dim: {self.state_dim}"
        )

    # ------------------------------------------------------------------
    # GYMNASIUM API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.shares_held = 0
        self.entry_price = 0.0
        self.current_step = 0
        self.trade_count = 0
        self.portfolio_history = [self.initial_balance]
        self.trade_log = []

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one trading step.

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        price = float(self.df["Close"].iloc[self.current_step])
        prev_portfolio = self._portfolio_value(price)

        reward = 0.0
        trade_made = False

        # ── Execute Action ────────────────────────────────────────────
        if action == BUY:
            cost = price * (1 + self.cfg.transaction_cost)
            if self.balance >= cost:
                self.shares_held += 1
                self.balance -= cost
                self.entry_price = price
                self.trade_count += 1
                trade_made = True
                self.trade_log.append({
                    "step": self.current_step,
                    "action": "BUY",
                    "price": price,
                    "shares": self.shares_held,
                })

        elif action == SELL:
            if self.shares_held > 0:
                proceeds = price * (1 - self.cfg.transaction_cost)
                self.shares_held -= 1
                self.balance += proceeds
                self.trade_count += 1
                trade_made = True
                self.trade_log.append({
                    "step": self.current_step,
                    "action": "SELL",
                    "price": price,
                    "shares": self.shares_held,
                })

        # ── Advance Step ──────────────────────────────────────────────
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        # ── Compute Reward ────────────────────────────────────────────
        next_price = float(self.df["Close"].iloc[self.current_step])
        new_portfolio = self._portfolio_value(next_price)

        # Base reward: change in portfolio value (scaled)
        reward = (new_portfolio - prev_portfolio) * self.cfg.reward_scaling

        # Overtrading penalty
        if self.trade_count > self.cfg.max_trades_per_episode:
            reward -= self.cfg.overtrading_penalty

        # Trend alignment bonus: reward holding in direction of EMA trend
        # Trend alignment bonus: reward holding in direction of EMA trend
        if self.shares_held > 0:
            ema_short = float(self.df[f"ema_{config.features.ema_short}"].iloc[self.current_step])
            ema_long  = float(self.df[f"ema_{config.features.ema_long}"].iloc[self.current_step])
            if ema_short > ema_long:  # Uptrend — holding is good
                atr = float(self.df["atr"].iloc[self.current_step])
                trend_bonus = atr * self.cfg.reward_scaling * 0.1
                reward += trend_bonus

        self.portfolio_history.append(new_portfolio)

        obs = self._get_obs()
        info = self._get_info()
        info["trade_made"] = trade_made
        info["action_name"] = ACTION_NAMES[action]

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Console render of current state."""
        price = float(self.df["Close"].iloc[self.current_step])
        pv = self._portfolio_value(price)
        pnl = pv - self.initial_balance
        pnl_pct = (pnl / self.initial_balance) * 100
        sentiment = self.sentiment_scores[self.current_step]

        msg = (
            f"Step {self.current_step:4d}/{self.n_steps} | "
            f"Price: {price:8.2f} | "
            f"Balance: {self.balance:9.2f} | "
            f"Shares: {self.shares_held} | "
            f"Portfolio: {pv:9.2f} | "
            f"P&L: {pnl_pct:+.1f}% | "
            f"Sentiment: {sentiment:+.2f}"
        )

        if self.render_mode == "human":
            print(msg)
        return msg

    def close(self):
        pass

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Build the 13-dim state vector for the current timestep.

        Vector = [Close, RSI, MACD, MACD_Signal, MACD_Hist,
                  EMA12, EMA26, BB_High, BB_Mid, BB_Low, BB_Width,
                  ATR, Sentiment]
        """
        step = min(self.current_step, self.n_steps - 1)
        row = self.df.iloc[step]

        raw = np.array(
            [float(row[col]) for col in FEATURE_COLUMNS] +
            [float(self.sentiment_scores[step])],
            dtype=np.float32
        )

        # Normalize using precomputed stats
        if self._norm_mean is not None and self._norm_std is not None:
            raw = (raw - self._norm_mean) / (self._norm_std + 1e-8)

        return raw.astype(np.float32)

    def _portfolio_value(self, price: float) -> float:
        """Current total portfolio value (cash + stock holdings)."""
        return self.balance + self.shares_held * price

    def _get_info(self) -> dict:
        """Return auxiliary info dict."""
        price = float(self.df["Close"].iloc[min(self.current_step, self.n_steps - 1)])
        return {
            "step": self.current_step,
            "price": price,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "portfolio_value": self._portfolio_value(price),
            "trade_count": self.trade_count,
        }

    def _compute_normalization(self):
        """
        Pre-compute mean/std for state normalization across the full dataset.
        This ensures consistent scaling regardless of market price level.
        """
        try:
            feature_data = self.df[FEATURE_COLUMNS].values.astype(np.float32)
            sentiment_col = np.array(self.sentiment_scores, dtype=np.float32).reshape(-1, 1)
            all_data = np.hstack([feature_data, sentiment_col])

            self._norm_mean = all_data.mean(axis=0)
            self._norm_std = all_data.std(axis=0)
        except Exception as e:
            logger.warning(f"Could not compute normalization stats: {e}")
            self._norm_mean = None
            self._norm_std = None

    # ------------------------------------------------------------------
    # PORTFOLIO METRICS (for backtesting)
    # ------------------------------------------------------------------

    def compute_metrics(self) -> dict:
        """
        Compute portfolio performance metrics after an episode.

        Returns dict with:
          total_return, sharpe_ratio, max_drawdown, win_rate, trade_count
        """
        if len(self.portfolio_history) < 2:
            return {}

        pv = np.array(self.portfolio_history)
        returns = np.diff(pv) / pv[:-1]

        total_return = (pv[-1] - pv[0]) / pv[0]

        # Sharpe Ratio (annualized, daily returns)
        rf_daily = (1 + config.backtest.risk_free_rate) ** (1 / 252) - 1
        excess = returns - rf_daily
        sharpe = (
            (excess.mean() / (excess.std() + 1e-9)) * np.sqrt(252)
        )

        # Max Drawdown
        peak = np.maximum.accumulate(pv)
        drawdowns = (pv - peak) / peak
        max_dd = float(drawdowns.min())

        # Win rate from trade log
        sells = [t for t in self.trade_log if t["action"] == "SELL"]
        buys  = [t for t in self.trade_log if t["action"] == "BUY"]
        wins  = sum(
            1 for s, b in zip(sells, buys) if s["price"] > b["price"]
        )
        win_rate = wins / len(sells) if sells else 0.0

        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "trade_count": len(self.trade_log),
            "final_portfolio": float(pv[-1]),
        }