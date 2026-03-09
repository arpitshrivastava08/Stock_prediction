import yfinance as yf
import pandas as pd

ticker = "^NSEI"

data = yf.download(ticker, start="2010-01-01", end="2024-01-01")

data.to_csv("nifty50_data.csv")

print("Dataset downloaded successfully")
print(data.head())