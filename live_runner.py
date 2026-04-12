import threading
import time
import logging
from datetime import datetime
from typing import List, Optional, Callable, Dict

import pandas as pd

try:
    import socketio
    import eventlet
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    logging.warning("pip install python-socketio eventlet")

from data_loader import StockDataLoader
from logger import get_logger

logger = get_logger(__name__)

REFRESH_INTERVAL_SEC = 300
SOCKETIO_PORT = 5001


class SocketIOServer:
    # simple socket server to push live updates

    def __init__(self):
        if not SOCKETIO_AVAILABLE:
            self.sio = None
            return

        self.sio = socketio.Server(
            cors_allowed_origins="*",
            async_mode="eventlet",
            logger=False,
            engineio_logger=False,
        )
        self.app = socketio.WSGIApp(self.sio)
        self.connected_clients: set = set()
        self._register_events()

    def _register_events(self):

        @self.sio.event
        def connect(sid, environ):
            self.connected_clients.add(sid)
            logger.info(f"Client connected: {sid}")

        @self.sio.event
        def disconnect(sid):
            self.connected_clients.discard(sid)
            logger.info(f"Client disconnected: {sid}")

    def start(self):
        if not SOCKETIO_AVAILABLE:
            logger.warning("Socket.IO not available")
            return

        def _run():
            logger.info(f"Starting socket server on {SOCKETIO_PORT}")
            try:
                eventlet.wsgi.server(
                    eventlet.listen(("0.0.0.0", SOCKETIO_PORT)),
                    self.app,
                    log_output=False,
                )
            except Exception as e:
                logger.error(f"Server crashed: {e}")

        threading.Thread(target=_run, daemon=True).start()

    def push(self, ticker: str, df: pd.DataFrame):
        # send latest row to all clients
        if not SOCKETIO_AVAILABLE or self.sio is None:
            return
        if df is None or df.empty:
            return
        if not self.connected_clients:
            return

        try:
            row = df.iloc[-1]
            payload = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "price": round(float(row["Close"]), 2),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
                "rows_total": len(df),
            }
            self.sio.emit("data_update", payload)
            logger.info(f"Pushed {ticker} to {len(self.connected_clients)} clients")
        except Exception as e:
            logger.error(f"Push error: {e}")

    def heartbeat(self):
        # simple ping to keep connection alive
        if self.sio and self.connected_clients:
            self.sio.emit("heartbeat", {"time": datetime.now().isoformat()})


class DataRefreshScheduler:
    # background loop for fetching data

    def __init__(self, sio_server: SocketIOServer):
        self.sio = sio_server
        self.loader = StockDataLoader()
        self.tickers: List[str] = []
        self.callbacks: List[Callable] = []
        self._stop = threading.Event()
        self._thread = None
        self._cache: Dict[str, pd.DataFrame] = {}

    def subscribe(self, ticker: str):
        if ticker not in self.tickers:
            self.tickers.append(ticker)
            logger.info(f"Subscribed: {ticker}")

    def add_callback(self, fn: Callable):
        self.callbacks.append(fn)

    def get_latest(self, ticker: str) -> Optional[pd.DataFrame]:
        return self._cache.get(ticker)

    def start(self):
        if not self.tickers:
            logger.warning("No tickers subscribed")
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Scheduler started for {self.tickers}")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Scheduler stopped")

    def _loop(self):
        logger.info("Refresh loop started")

        # fetch once immediately
        self._fetch_all()

        while not self._stop.wait(timeout=REFRESH_INTERVAL_SEC):
            self._fetch_all()
            self.sio.heartbeat()

        logger.info("Refresh loop ended")

    def _fetch_all(self):
        for ticker in self.tickers:
            try:
                logger.info(f"Refreshing: {ticker}")

                df = self.loader.load(
                    ticker=ticker,
                    period="2mo",
                    interval="5m",
                    force_download=True,
                )

                if df is not None and not df.empty:
                    self._cache[ticker] = df
                    self.sio.push(ticker, df)

                    for cb in self.callbacks:
                        try:
                            cb(ticker, df)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

            except Exception as e:
                logger.error(f"Refresh failed for {ticker}: {e}")


_sio_server = SocketIOServer()
_scheduler = DataRefreshScheduler(_sio_server)


def start_live_feed(
    tickers: List[str] = None,
    callback: Callable = None,
):
    tickers = tickers or ["^NSEI"]

    _sio_server.start()

    for t in tickers:
        _scheduler.subscribe(t)

    if callback:
        _scheduler.add_callback(callback)

    _scheduler.start()

    logger.info("Live feed started")


def stop_live_feed():
    _scheduler.stop()
    logger.info("Live feed stopped")