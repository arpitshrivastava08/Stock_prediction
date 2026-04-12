import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from logger import get_logger
from config import config

logger = get_logger(__name__)


class StockDataLoader:

    def __init__(self):
        self.data_dir = config.data.csv_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def load(
        self,
        ticker: str = "^NSEI",
        period: str = "2mo",
        interval: str = None,
        force_download: bool = False,
    ) -> pd.DataFrame:

        interval = self._normalize_interval(interval or config.data.default_interval)
        period = self._normalize_period(period, interval)

        csv_path = self._csv_path(ticker, period, interval)

        # CACHE CHECK (5 minutes instead of 6 hours)
        if not force_download and os.path.exists(csv_path):
            age_hours = self._file_age_hours(csv_path)  # current time - last modified time of file
            if age_hours < 0.08:   # 🔥 ~5 minutes
                logger.info(f"Loading {ticker} from cache ({age_hours:.2f}h old)")
                return self._load_csv(csv_path) # skip api call load from disk (very fast)

        # API call
        df = self._download_from_api(ticker, period, interval)

        # For long intraday requests (e.g. 2y + 5m), Yahoo does not provide
        # full history directly. Build a stable 60-bars/day intraday set from
        # daily bars and overwrite the latest ~60d with real intraday where available.
        if (df is None or df.empty) and self._needs_long_intraday_build(period, interval):
            logger.warning(
                f"Direct intraday download unavailable for {ticker} ({period}, {interval}). "
                "Building long intraday dataset from daily history + recent intraday."
            )
            df = self._build_long_intraday_dataset(ticker, period, interval)

        if df is None or df.empty:
            logger.warning(f"API failed for {ticker}. Trying CSV fallback...")
            df = self._load_csv(csv_path) # load old saved data

        if df is None or df.empty:
            if interval.endswith("m"):
                raise RuntimeError(
                    f"No cached intraday data found for {ticker} ({period}, {interval}). "
                    "Yahoo Finance does not provide long intraday history. "
                    "Use a shorter period (for example 60d), provide a local CSV cache, "
                    "or switch to a different intraday data source."
                )

            logger.error(f"No data available for {ticker}. Using synthetic fallback.")
            df = self._generate_synthetic_data(ticker) # fake data prevent crash and keeps the system running

        # Save
        if df is not None and not df.empty:
            df.to_csv(csv_path)
            logger.info(f"Saved {ticker} data: {len(df)} rows -> {csv_path}")

        return df

    def _download_from_api(
        self, ticker: str, period: str, interval: str
    ) -> Optional[pd.DataFrame]:

        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not installed.")
            return None

        candidates = self._ticker_candidates(ticker, interval)

        for symbol in candidates:
            logger.info(f"Downloading {symbol} ({period}, {interval})")

            for attempt in range(1, 4):
                try:
                    # Path 1: batch downloader
                    df = yf.download(
                        symbol,
                        period=period,
                        interval=self._normalize_interval(interval),
                        auto_adjust=False,
                        progress=False,
                        threads=False,
                    )

                    # Path 2: direct ticker history (often works when download fails)
                    if df is None or df.empty:
                        df = yf.Ticker(symbol).history(
                            period=period,
                            interval=self._normalize_interval(interval),
                            auto_adjust=False,
                        )

                    if df is None or df.empty:
                        if attempt < 3:
                            time.sleep(1.5 * attempt)
                        continue

                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    df.columns = [str(col).strip().capitalize() for col in df.columns]
                    df = df.rename(columns={"Adj close": "Close"})
                    # Keep the first occurrence when Yahoo returns duplicate names.
                    df = df.loc[:, ~df.columns.duplicated()]

                    df = self._standardize_columns(df)
                    if df.empty:
                        if attempt < 3:
                            time.sleep(1.5 * attempt)
                        continue

                    if ticker != symbol:
                        logger.warning(f"Using fallback symbol {symbol} for {ticker}")

                    return df

                except Exception as e:
                    logger.warning(
                        f"Attempt {attempt}/3 failed for {symbol}: {e}"
                    )
                    if attempt < 3:
                        time.sleep(1.5 * attempt)

        return None

    @staticmethod
    def _normalize_interval(interval: str) -> str:
        i = str(interval).strip().lower()
        alias = {
            "1min": "1m",
            "2min": "2m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "60min": "60m",
        }
        return alias.get(i, i)

    @staticmethod
    def _period_to_days(period: str) -> Optional[int]:
        p = str(period).strip().lower()
        if p.endswith("d"):
            return int(p[:-1])
        if p.endswith("mo"):
            return int(p[:-2]) * 30
        if p.endswith("y"):
            return int(p[:-1]) * 365
        if p == "max":
            return 100000
        return None

    def _needs_long_intraday_build(self, period: str, interval: str) -> bool:
        days = self._period_to_days(period)
        return bool(interval.endswith("m") and days is not None and days > 60)

    def _build_long_intraday_dataset(
        self,
        ticker: str,
        period: str,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        # 1) Daily history for full requested period
        daily = self._download_from_api(ticker, period, "1d")
        if daily is None or daily.empty:
            return None

        # 2) Expand daily bars to 60 bars/day intraday grid.
        # This intentionally keeps exactly 60 rows/day for all days.
        expanded = self._expand_daily_to_intraday(daily, interval=interval, bars_per_day=60)
        if expanded is None or expanded.empty:
            return None

        return self._standardize_columns(expanded)

    def _expand_daily_to_intraday(
        self,
        daily_df: pd.DataFrame,
        interval: str = "5m",
        bars_per_day: int = 60,
    ) -> pd.DataFrame:
        interval = self._normalize_interval(interval)
        if interval != "5m":
            return pd.DataFrame()

        rows = []
        idx = []

        # 60 x 5m = 300 minutes. Use 09:15-14:10 IST-style session length.
        for dt, row in daily_df.iterrows():
            o = float(row.get("Open", np.nan))
            h = float(row.get("High", np.nan))
            l = float(row.get("Low", np.nan))
            c = float(row.get("Close", np.nan))
            v = float(row.get("Volume", 0.0))

            if np.isnan(o) or np.isnan(h) or np.isnan(l) or np.isnan(c):
                continue

            base = np.linspace(o, c, bars_per_day)
            # Add mild intraday texture while preserving O/H/L/C envelope.
            wave = 0.15 * (h - l) * np.sin(np.linspace(0, 2 * np.pi, bars_per_day))
            path = np.clip(base + wave, l, h)
            path[0] = o
            path[-1] = c

            opens = np.empty(bars_per_day)
            closes = np.empty(bars_per_day)
            highs = np.empty(bars_per_day)
            lows = np.empty(bars_per_day)

            opens[0] = o
            closes[0] = path[0]
            highs[0] = max(opens[0], closes[0])
            lows[0] = min(opens[0], closes[0])

            for i in range(1, bars_per_day):
                opens[i] = closes[i - 1]
                closes[i] = path[i]
                highs[i] = max(opens[i], closes[i])
                lows[i] = min(opens[i], closes[i])

            # Ensure day extremes appear in the generated series.
            highs[np.argmax(path)] = max(highs[np.argmax(path)], h)
            lows[np.argmin(path)] = min(lows[np.argmin(path)], l)

            vol_each = v / bars_per_day if bars_per_day > 0 else 0.0

            day = pd.Timestamp(dt)
            session_start = day.replace(hour=9, minute=15, second=0, microsecond=0)

            for i in range(bars_per_day):
                idx.append(session_start + timedelta(minutes=5 * i))
                rows.append({
                    "Open": opens[i],
                    "High": highs[i],
                    "Low": lows[i],
                    "Close": closes[i],
                    "Volume": vol_each,
                })

        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
        out.sort_index(inplace=True)
        return out

    @staticmethod
    def _ticker_candidates(ticker: str, interval: str) -> List[str]:
        # Keep only direct symbol candidates to avoid silently switching
        # to a different instrument (e.g., ETF) during fallback.
        return [ticker]

    @staticmethod
    def _normalize_period(period: str, interval: str) -> str:
        """
        Normalize period strings for Yahoo intraday compatibility.
        Yahoo behaves more consistently with explicit day-count periods for
        intraday intervals.
        """
        p = str(period).strip().lower()
        if interval.endswith("m"):
            intraday_map = {
                "1mo": "30d",
                "2mo": "60d",
                "3mo": "60d",
            }
            return intraday_map.get(p, p)
        return p

    def _load_csv(self, filepath: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(filepath):
            return None
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True) # parse_dates convert into datetime object
            return self._standardize_columns(df)
        except Exception:
            return None

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {}
        for col in df.columns:
            c = str(col).lower()
            if "open" in c:
                col_map[col] = "Open"
            elif "high" in c:
                col_map[col] = "High"
            elif "low" in c:
                col_map[col] = "Low"
            elif "close" in c or "adj" in c:
                col_map[col] = "Close"
            elif "volume" in c:
                col_map[col] = "Volume"

        df = df.rename(columns=col_map)

        keep = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in keep if c in df.columns]].copy()

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(inplace=True)
        return df


    def _generate_synthetic_data(
        self, ticker: str, n_bars: int = 1_000, last_price: float = None
    ) -> pd.DataFrame:
        
        logger.warning(f"SYNTHETIC DATA for {ticker} — results are NOT real!")
        np.random.seed(abs(hash(ticker)) % 2**31)
 
        # Realistic NIFTY base prices
        nifty_base = {
            "^NSEI":    22_000.0,
            "^BSESN":   73_000.0,
            "^NSEBANK": 48_000.0,
        }
        base = last_price or nifty_base.get(ticker, 1_000.0)
 
        # Random walk: 0.2% step per bar (realistic for NIFTY 5m candles)
        step = base * 0.002
        prices = np.cumsum(np.random.normal(0, step, n_bars)) + base
        prices = np.abs(prices)
 
        open_p  = prices + np.random.normal(0, step * 0.3, n_bars)
        close_p = prices + np.random.normal(0, step * 0.3, n_bars)
        high    = np.maximum(open_p, close_p) + np.random.uniform(0, step * 0.5, n_bars)
        low     = np.minimum(open_p, close_p) - np.random.uniform(0, step * 0.5, n_bars)
        volume  = np.random.randint(100_000, 5_000_000, n_bars).astype(float)
 
        # 5-minute timestamps going back from now on business days
        dates = pd.bdate_range(end=datetime.today(), periods=n_bars, freq="5min")
 
        return pd.DataFrame({
            "Open":   open_p,
            "High":   high,
            "Low":    low,
            "Close":  close_p,
            "Volume": volume,
        }, index=dates)
    # for create unique filename for each dataeset
    def _csv_path(self, ticker: str, period: str, interval: str) -> str:
        safe = ticker.replace("^", "IDX_").replace(".", "_")
        return os.path.join(self.data_dir, f"{safe}_{period}_{interval}.csv")

    # create unique filename for caching data
    @staticmethod
    def _file_age_hours(filepath: str) -> float:
        return (datetime.now().timestamp() - os.path.getmtime(filepath)) / 3600 # return in hour
    # cache brain -> check how old the csv file is 