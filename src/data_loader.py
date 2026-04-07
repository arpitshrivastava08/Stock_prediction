import os
import pandas as pd
import yfinance as yf
from datetime import datetime

class StockDataLoader:

    def __init__(self):
        self.cache_dir = "data"
        os.makedirs(self.cache_dir, exist_ok=True)

    def load(self, ticker="^NSEI", period="1mo", interval="1d", force_download=False):
        file_path = self._get_cache_path(ticker, period, interval)

        # Load from cache if recent
        if not force_download and os.path.exists(file_path):
            age = self._file_age_minutes(file_path)
            if age < 10:  # cache valid for 10 minutes
                print(f"Loading cached data ({age:.1f} min old)")
                return self._load_csv(file_path)

        # Download from API
        df = self._download(ticker, period, interval)

        if df is None or df.empty:
            print("Download failed, trying cache...")
            return self._load_csv(file_path)

        # 🔹 Clean + standardize
        df = self._standardize_columns(df)

        # 🔹 Save to cache
        df.to_csv(file_path)

        return df

    # Helper Functions

    def _download(self, ticker, period, interval):
        try:
            df = yf.download(ticker, period=period, interval=interval)
            df.dropna(inplace=True)
            return df
        except Exception as e:
            print("Download error:", e)
            return None

    def _standardize_columns(self, df):
        df.columns = [str(col).strip().capitalize() for col in df.columns]

        # Rename adjusted close if present
        if "Adj close" in df.columns:
            df.rename(columns={"Adj close": "Close"}, inplace=True)

        # Keep only required columns
        required = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[col for col in required if col in df.columns]]

        return df

    def _get_cache_path(self, ticker, period, interval):
        safe = ticker.replace("^", "IDX_")
        return os.path.join(self.cache_dir, f"{safe}_{period}_{interval}.csv")

    def _file_age_minutes(self, filepath):
        return (datetime.now().timestamp() - os.path.getmtime(filepath)) / 60

    def _load_csv(self, filepath):
        if not os.path.exists(filepath):
            return None
        try:
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        except Exception:
            return None