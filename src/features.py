
"""
features.py — Technical Indicator computation module.

Indicators Computed:
  RSI(14)           — Momentum: overbought (>70) / oversold (<30)
  MACD              — Trend: crossover signals for entry/exit timing
  EMA(12), EMA(26)  — Trend: price direction and dynamic support
  Bollinger Bands   — Volatility: band breakout detection
  ATR(14)           — Volatility: used for stop-loss / position sizing

All indicators are computed using the `ta` library or manual numpy/pandas
implementations for transparency.
"""

import pandas as pd
import numpy as np
from typing import Optional

from logger import get_logger
from config import config

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Adds technical indicators to a raw OHLCV DataFrame.

    Usage:
        fe = FeatureEngineer()
        df_with_features = fe.compute_all(df)
    """

    def __init__(self):
        self.cfg = config.features

    
    # PUBLIC API
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators and append them as new columns.

        Args:
            df: OHLCV DataFrame with columns [Open, High, Low, Close, Volume]

        Returns:
            DataFrame with added indicator columns + NaN rows dropped
        """
        df = df.copy()
        self._validate(df)

        logger.info("Computing technical indicators...")

        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_ema(df)
        df = self._add_bollinger_bands(df)
        df = self._add_atr(df)

        # Drop rows that have NaN due to indicator lookback periods
        before = len(df)
        df.dropna(inplace=True)
        after = len(df)
        logger.info(f"Indicators computed. Dropped {before - after} NaN rows. Final shape: {df.shape}")

        return df

    
    # RSI — Relative Strength Index
    

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        RSI = 100 - (100 / (1 + RS))
        RS  = Average Gain / Average Loss over N periods

        Signals:
          RSI > 70 → Overbought (potential sell)
          RSI < 30 → Oversold  (potential buy)
        """
        w = self.cfg.rsi_window
        close = df["Close"]

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        
        avg_gain = gain.ewm(com=w - 1, min_periods=w).mean()
        avg_loss = loss.ewm(com=w - 1, min_periods=w).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = (100 - (100 / (1 + rs))).clip(0, 100)

        logger.debug(f"RSI({w}): range [{df['rsi'].min():.1f}, {df['rsi'].max():.1f}]")
        return df

   
    # MACD — Moving Average Convergence Divergence
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACD Line    = EMA(fast) - EMA(slow)
        Signal Line  = EMA(macd_line, signal_period)
        Histogram    = MACD Line - Signal Line

        Signals:
          MACD crosses above Signal → Bullish
          MACD crosses below Signal → Bearish
        """
        fast = self.cfg.macd_fast
        slow = self.cfg.macd_slow
        signal = self.cfg.macd_signal

        ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        logger.debug(f"MACD({fast},{slow},{signal}) computed.")
        return df

    
    # EMA — Exponential Moving Average
    
    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        EMA gives more weight to recent prices than SMA.
        alpha = 2 / (span + 1)

        EMA(12) is more reactive; EMA(26) acts as baseline trend.
        When EMA(12) > EMA(26): bullish trend
        """
        short = self.cfg.ema_short
        long_ = self.cfg.ema_long

        df[f"ema_{short}"] = df["Close"].ewm(span=short, adjust=False).mean()
        df[f"ema_{long_}"] = df["Close"].ewm(span=long_, adjust=False).mean()

        logger.debug(f"EMA({short}) and EMA({long_}) computed.")
        return df

   
    # BOLLINGER- BANDS
   

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Middle Band  = SMA(N)
        Upper Band   = SMA(N) + k * StdDev(N)
        Lower Band   = SMA(N) - k * StdDev(N)

        Default N=20, k=2

        Signals:
          Price > Upper Band → Overbought / breakout up
          Price < Lower Band → Oversold / breakout down
          Bandwidth         → Measures volatility squeeze
        """
        w = self.cfg.bb_window
        k = self.cfg.bb_std

        sma = df["Close"].rolling(window=w).mean()
        std = df["Close"].rolling(window=w).std()

        df["bb_mid"] = sma
        df["bb_high"] = sma + k * std
        df["bb_low"] = sma - k * std
        df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]  # Normalized width

        logger.debug(f"Bollinger Bands({w}, {k}) computed.")
        return df

  
    # ATR — Average True Range
   
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        True Range = max(
            High - Low,
            |High - Prev Close|,
            |Low  - Prev Close|
        )
        ATR = Wilder EMA of True Range over N periods

        Used for:
          - Dynamic stop-loss: entry - ATR * multiplier
          - Position sizing:   risk / ATR
        """
        w = self.cfg.atr_window

        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        df["atr"] = tr.ewm(com=w - 1, min_periods=w).mean()

        logger.debug(f"ATR({w}): mean={df['atr'].mean():.2f}")
        return df

 
    # VALIDATION
   

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        if len(df) < 50:
            raise ValueError(f"Insufficient data: {len(df)} rows (minimum 50 required)")



# FEATURE NAMES — for RL state vector


FEATURE_COLUMNS = [
    "Close",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    f"ema_{config.features.ema_short}",
    f"ema_{config.features.ema_long}",
    "bb_high",
    "bb_mid",
    "bb_low",
    "bb_width",
    "atr",
]

STATE_DIM = len(FEATURE_COLUMNS) + 1  


_fe = FeatureEngineer()

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Module-level shortcut."""
    return _fe.compute_all(df)
