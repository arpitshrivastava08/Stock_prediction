"""
sentiment.py — NLP-based Financial News Sentiment Analysis Module

Architecture:

  NEWS SOURCES (NewsAPI / RSS / Fallback Headlines)
       ↓
  PREPROCESSING (tokenize, clean, lowercase)
       ↓
  FinBERT NLP Model (ProsusAI/finbert)
       ↓
  SENTIMENT SCORING: POSITIVE→+1, NEGATIVE→-1, NEUTRAL→0
       ↓
  AGGREGATED SCORE: weighted avg → float in [-1.0, +1.0]
       ↓
  STATE VECTOR INJECTION: state[..., -1] = sentiment_score
"""

import re
import json
import os
import time
from datetime import datetime
from typing import List, Optional, Dict

from logger import get_logger
from config import config
from universe import NIFTY50

logger = get_logger(__name__)


# SENTIMENT MODULE

class SentimentAnalyzer:
    """
    Fetches financial news headlines and converts them to a
    scalar sentiment score in [-1.0, +1.0] using FinBERT.

    The score is cached to disk to avoid hammering the API and
    re-running inference on unchanged news.
    """

    def __init__(self):
        self.cfg = config.sentiment
        self._pipe = None          # Lazy-load: only initialize when needed
        self._cache: Dict = {}
        self._load_cache()

    # PUBLIC API

    def get_score(self, ticker: str, force_refresh: bool = False) -> float:
        """
        Return sentiment score for a ticker symbol.

        Args:
            ticker: Stock symbol used as search query
            force_refresh: Skip cache, re-fetch and re-score

        Returns:
            float in [-1.0, +1.0]
        """
#         DO THIS IN RL / PORTFOLIO CODE:
# held_stocks = [t for t in portfolio if t in NIFTY50]
# scores = get_scores_batch(held_stocks)

        cache_key = self._cache_key(ticker)

        # Check cache freshness
        if not force_refresh and self._is_cache_fresh(cache_key):
            score = self._cache[cache_key]["score"]
            logger.info(f"[{ticker}] Sentiment from cache: {score:.3f}")
            return score

        # Fetch headlines
        headlines = self._fetch_headlines(ticker)

        if not headlines:
            logger.warning(f"[{ticker}] No news found. Returning neutral score 0.0")
            return 0.0

        # Clean text
        cleaned = [self._clean_text(h) for h in headlines if h.strip()]
        cleaned = [c for c in cleaned if len(c) > 10]  # filter very short

        if not cleaned:
            return 0.0

        # Run FinBERT inference
        score = self._infer_score(cleaned)

        # Cache result
        self._cache[cache_key] = {
            "score": score,
            "timestamp": time.time(),
            "headlines_count": len(cleaned),
        }
        self._save_cache()

        logger.info(f"[{ticker}] Sentiment score: {score:.3f} (from {len(cleaned)} headlines)")
        return score

    def get_scores_batch(self, tickers: List[str]) -> Dict[str, float]:
        """Get sentiment scores for multiple tickers."""
        return {ticker: self.get_score(ticker) for ticker in tickers}

    # NEWS FETCHING

    def _fetch_headlines(self, ticker: str) -> List[str]:
        """
        Attempts multiple news sources in order of priority:
        1. NewsAPI (if key is configured)
        2. Fallback sample headlines (for demo / offline use)
        """
        headlines = []

        # 1. NewsAPI
        if self.cfg.newsapi_key:
            headlines = self._fetch_from_newsapi(ticker)

        # 2. No fallback — return empty if no news
        if not headlines:
            logger.warning(f"[{ticker}] No real news found.")
            return []

        return headlines[:self.cfg.max_articles]

    def _fetch_from_newsapi(self, ticker: str) -> List[str]:
        """Fetch from NewsAPI.org"""
        try:
            from newsapi import NewsApiClient
            api = NewsApiClient(api_key=self.cfg.newsapi_key)

            # Build human-readable query from ticker symbol
            query = self._ticker_to_query(ticker)

            resp = api.get_everything(
                q=query,
                language="en",
                sort_by="publishedAt",
                page_size=self.cfg.max_articles,
            )
            articles = resp.get("articles", [])
            headlines = [a["title"] for a in articles if a.get("title")]
            logger.info(f"[{ticker}] Fetched {len(headlines)} headlines from NewsAPI")
            return headlines

        except ImportError:
            logger.warning("newsapi-python not installed. Skipping NewsAPI.")
            return []
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return []

    @staticmethod
    def _ticker_to_query(ticker: str) -> str:
        """Convert stock symbol to human-readable search query."""
        mapping = {
            "^NSEI": "NIFTY 50 India stock market",
            "^BSESN": "Sensex BSE India stock market",
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS": "Tata Consultancy Services",
            "INFY.NS": "Infosys",
            "HDFCBANK.NS": "HDFC Bank India",
            "WIPRO.NS": "Wipro IT services",
        }
        return mapping.get(
            ticker,
            ticker.replace(".NS", "").replace("^", "") + " stock India news"
        )

    # TEXT PREPROCESSING

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Preprocessing pipeline:
          1. Remove URLs
          2. Remove special characters, keep alphanumeric + spaces
          3. Normalize whitespace
          4. Lowercase
        """
        text = re.sub(r"http\S+|www\S+", "", text)            # URLs
        text = re.sub(r"[^A-Za-z0-9 .,!?%$&\-']", " ", text)  # Special chars
        text = re.sub(r"\s+", " ", text)                        # Multi-space
        return text.strip().lower()

    # FINBERT INFERENCE

    def _get_pipeline(self):
        """
        Lazy-load FinBERT pipeline.
        Only downloaded once; subsequent calls use cached model.
        """
        if self._pipe is not None:
            return self._pipe

        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"Loading FinBERT model: {self.cfg.model_name}")
            self._pipe = hf_pipeline(
                "sentiment-analysis",
                model=self.cfg.model_name,
                tokenizer=self.cfg.model_name,
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}. Will use fallback scorer.")
            self._pipe = None

        return self._pipe

    def _infer_score(self, headlines: List[str]) -> float:
        """
        Run FinBERT (or fallback) on list of cleaned headlines.
        Returns normalized score in [-1.0, +1.0].

        Scoring formula:
          positive → +confidence
          negative → -confidence
          neutral  →  0.0
          final    = sum / count  (normalized average)
        """
        pipe = self._get_pipeline()

        if pipe is not None:
            return self._finbert_score(pipe, headlines)
        else:
            logger.warning("FinBERT unavailable. Using lexicon-based fallback.")
            return self._lexicon_fallback_score(headlines)

    def _finbert_score(self, pipe, headlines: List[str]) -> float:
        """Score using the FinBERT transformer model."""
        try:
            results = pipe(headlines)
            score = self._aggregate_results(results)
            return score
        except Exception as e:
            logger.error(f"FinBERT inference error: {e}")
            return 0.0

    @staticmethod
    def _aggregate_results(results: List[dict]) -> float:
        """
        Convert FinBERT label+score dicts to single scalar.

        Input: [{'label': 'positive', 'score': 0.94}, ...]
        Output: float in [-1.0, +1.0]
        """
        if not results:
            return 0.0

        total = 0.0
        for r in results:
            label = r["label"].lower()
            confidence = r["score"]
            if label == "positive":
                total += confidence
            elif label == "negative":
                total -= confidence
            # neutral contributes 0

        return max(-1.0, min(1.0, total / len(results)))

    @staticmethod
    def _lexicon_fallback_score(headlines: List[str]) -> float:
        """
        Simple keyword-based sentiment fallback when FinBERT is unavailable.
        Uses financial domain positive/negative keyword lists.
        """
        positive_words = {
            "profit", "gain", "rise", "surge", "record", "beat",
            "growth", "dividend", "upgrade", "bullish", "rally",
            "outperform", "strong", "inflow", "increase", "win"
        }
        negative_words = {
            "loss", "fall", "decline", "slump", "miss", "weak",
            "downgrade", "bearish", "probe", "investigation", "selloff",
            "outflow", "decrease", "resign", "fraud", "concern"
        }

        scores = []
        for h in headlines:
            words = set(h.lower().split())
            pos = len(words & positive_words)
            neg = len(words & negative_words)
            total = pos + neg
            if total == 0:
                scores.append(0.0)
            else:
                scores.append((pos - neg) / total)

        return sum(scores) / len(scores) if scores else 0.0

    # CACHE MANAGEMENT

    def _load_cache(self):
        if os.path.exists(self.cfg.cache_file):
            try:
                with open(self.cfg.cache_file, "r") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cfg.cache_file), exist_ok=True)
        try:
            with open(self.cfg.cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save sentiment cache: {e}")

    def _is_cache_fresh(self, key: str) -> bool:
        if key not in self._cache:
            return False
        age = time.time() - self._cache[key].get("timestamp", 0)
        return age < self.cfg.refresh_minutes * 60

    @staticmethod
    def _cache_key(ticker: str) -> str:
        return f"sentiment_{ticker}_{datetime.now().strftime('%Y%m%d')}"


# MODULE-LEVEL CONVENIENCE

_analyzer = SentimentAnalyzer()

def get_sentiment_score(ticker: str, force_refresh: bool = False) -> float:
    """Module-level shortcut to SentimentAnalyzer.get_score()"""
    return _analyzer.get_score(ticker, force_refresh)