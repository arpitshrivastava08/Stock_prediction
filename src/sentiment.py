import random
from typing import List

class SentimentAnalyzer:
    """
    Basic sentiment analyzer using simple keyword matching.
    """

    def get_score(self, ticker: str) -> float:
        headlines = self._get_demo_headlines(ticker)

        if not headlines:
            return 0.0

        return self._simple_score(headlines)

    def _get_demo_headlines(self, ticker: str) -> List[str]:
        """Generate dummy headlines"""
        positive = [
            f"{ticker} reports strong profits",
            f"{ticker} stock surges after good results",
            f"{ticker} sees strong growth",
        ]

        negative = [
            f"{ticker} reports losses",
            f"{ticker} stock falls sharply",
            f"{ticker} faces investigation",
        ]

        neutral = [
            f"{ticker} trades flat today",
            f"{ticker} announces upcoming results",
        ]

        pool = positive + negative + neutral
        return random.sample(pool, min(5, len(pool)))

    def _simple_score(self, headlines: List[str]) -> float:
        positive_words = {"profit", "growth", "surge", "strong"}
        negative_words = {"loss", "fall", "investigation"}

        score = 0

        for h in headlines:
            words = set(h.lower().split())

            if words & positive_words:
                score += 1
            elif words & negative_words:
                score -= 1

        return score / len(headlines)


# module-level function
_analyzer = SentimentAnalyzer()

def get_sentiment_score(ticker: str) -> float:
    return _analyzer.get_score(ticker)