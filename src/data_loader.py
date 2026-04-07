import pandas as pd
import yfinance as yf

class StockDataLoader:

    def load(self, ticker="^NSEI", period="1mo", interval="1d"):
        try:
            df = yf.download(ticker, period=period, interval=interval)
            df = df.dropna()
            return df
        except Exception as e:
            print("Error loading data:", e)
            return None