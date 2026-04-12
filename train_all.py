"""
train_all.py — Master Training Script

Trains all 4 RL agents in sequence:
  1. DQN    (stable-baselines3)
  2. DDQN   (custom implementation)
  3. PPO    (stable-baselines3)
  4. A2C    (stable-baselines3)

Then evaluates each agent and updates ensemble weights.

Supports two modes:
  1. INITIAL TRAINING (default):
     Trains all agents from scratch on historical data.
     Use this once at the start.

  2. RETRAIN (--retrain flag):
     Runs the sliding window retrainer.
     Use this at end of each trading day to update models with new data.
     Much faster than initial training (~20k steps vs 100k).

Usage:
  # Initial training:
  python src/train_all.py --ticker RELIANCE.NS --timesteps 100000

  # End-of-day retrain (sliding window):
  python src/train_all.py --ticker ^NSEI --retrain

  # Force retrain even without enough new bars:
  python src/train_all.py --ticker ^NSEI --retrain --force

  # Quick test run:
  python src/train_all.py --ticker ^NSEI --timesteps 50000 --quick
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import config
from logger import get_logger
from data_loader import StockDataLoader
from features import FeatureEngineer
from sentiment import SentimentAnalyzer
from env import StockTradingEnv
from train_dqn import train_dqn, train_ddqn
from train_ppo import train_ppo, train_a2c
from backtest import Backtester

logger = get_logger(__name__)


def prepare_data(ticker: str, period: str = "2y", interval: str = None) -> tuple:
    """Load data, compute features, get sentiment scores."""
    logger.info(f"=== DATA PREPARATION: {ticker} ===")

    loader = StockDataLoader()
    raw = loader.load(ticker, period, interval=interval)
    if raw.empty:
        raise RuntimeError(f"No data loaded for {ticker}")

    fe = FeatureEngineer()
    df = fe.compute_all(raw)
    logger.info(f"Features computed: {df.shape}")

    analyzer = SentimentAnalyzer()
    base_score = analyzer.get_score(ticker)

    np.random.seed(42)
    noise = np.random.normal(0, 0.05, len(df))
    sentiment_scores = np.clip(base_score + noise, -1, 1).tolist()
    logger.info(f"Sentiment scores: mean={np.mean(sentiment_scores):.3f}")

    return df, sentiment_scores


def train_test_split(df: pd.DataFrame, sentiment_scores: list, split: float = 0.8):
    """Split data into train/test sets."""
    n = len(df)
    split_idx = int(n * split)

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    train_sent = sentiment_scores[:split_idx]
    test_sent = sentiment_scores[split_idx:]

    logger.info(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    return train_df, test_df, train_sent, test_sent


def run_backtest(test_df: pd.DataFrame, test_sent: list, models: dict):
    """Run backtest comparison on test data."""
    logger.info("=== BACKTESTING ===")

    test_env = StockTradingEnv(test_df, test_sent)
    backtester = Backtester(test_env)

    bh = backtester.run_buy_and_hold()

    results = {"buy_hold": bh}
    for name, model in models.items():
        r = backtester.run(model, label=name)
        results[name] = r

    comparison = backtester.compare(models)
    logger.info("\n" + comparison.to_string(index=False))

    os.makedirs(config.train.models_dir, exist_ok=True)
    summary = {
        k: {
            "total_return": v.get("total_return", 0),
            "sharpe_ratio": v.get("sharpe_ratio", 0),
            "max_drawdown": v.get("max_drawdown", 0),
        }
        for k, v in results.items()
    }
    with open(os.path.join(config.train.models_dir, "backtest_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Backtest results saved.")

    return results


def run_initial_training(args):
    """Full initial training from scratch."""
    logger.info("=" * 60)
    logger.info("INITIAL TRAINING MODE")
    logger.info("=" * 60)
    logger.info(
        f"Ticker: {args.ticker} | Period: {args.period} | Interval: {args.interval} | Timesteps: {args.timesteps}"
    )

    start_time = time.time()

    df, sentiment_scores = prepare_data(args.ticker, args.period, args.interval)
    train_df, test_df, train_sent, test_sent = train_test_split(df, sentiment_scores)

    train_env = StockTradingEnv(train_df, train_sent)

    trained_models = {}
    os.makedirs(config.train.models_dir, exist_ok=True)

    training_tasks = [
        ("DQN",  "dqn",  lambda: train_dqn(train_env,  args.timesteps)),
        ("DDQN", "ddqn", lambda: train_ddqn(train_env, args.timesteps)),
        ("PPO",  "ppo",  lambda: train_ppo(train_env,  args.timesteps)),
        ("A2C",  "a2c",  lambda: train_a2c(train_env,  args.timesteps)),
    ]

    for display_name, key, train_fn in training_tasks:
        if key in args.skip:
            logger.info(f"Skipping {display_name} (--skip {key})")
            continue

        logger.info(f"\n{'-'*40}")
        logger.info(f"Training {display_name}...")
        logger.info(f"{'-'*40}")

        try:
            t0 = time.time()
            model = train_fn()
            elapsed = time.time() - t0
            trained_models[key] = model
            logger.info(f"✓ {display_name} trained in {elapsed/60:.1f} minutes")
        except Exception as e:
            logger.error(f"✗ {display_name} training failed: {e}")
            import traceback
            traceback.print_exc()

    if trained_models:
        logger.info(f"\n{'-'*40}")
        logger.info("Running backtest on held-out test data...")

        try:
            backtest_results = run_backtest(test_df, test_sent, trained_models)

            # After initial training, record the bar count so the retrainer
            # knows how much data was in scope during this training run
            from retrain import RetrainState
            raw_len = len(StockDataLoader().load(args.ticker, args.period, interval=args.interval))
            RetrainState().update(args.ticker, raw_len, {})
            logger.info(f"Retrain state initialized: {raw_len} bars recorded")

        except Exception as e:
            logger.error(f"Backtest failed: {e}")

    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"INITIAL TRAINING COMPLETE")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Models trained: {list(trained_models.keys())}")
    logger.info(f"Models saved in: {config.train.models_dir}")
    logger.info(f"{'='*60}")
    logger.info("\nNext steps:")
    logger.info("  1. Run the dashboard:  streamlit run src/app.py")
    logger.info("  2. Retrain at day end: python src/train_all.py --retrain")


def run_retrain(args):
    """
    Sliding window end-of-day retraining.
    Delegates to retrain.SlidingWindowRetrainer.
    """
    logger.info("=" * 60)
    logger.info("SLIDING WINDOW RETRAIN MODE")
    logger.info("=" * 60)

    from retrain import SlidingWindowRetrainer

    timesteps = args.timesteps if hasattr(args, "timesteps") and args.timesteps else None
    agents = args.skip  # --skip flag repurposed as exclusion list
    # Build list of agents to train (exclude skipped ones)
    all_agents = config.retrain.retrain_agents
    agents_to_train = [a for a in all_agents if a not in (args.skip or [])]

    retrainer = SlidingWindowRetrainer()
    result = retrainer.run(
        ticker=args.ticker,
        period=args.period,
        force=args.force if hasattr(args, "force") else False,
        timesteps=timesteps,
        agents=agents_to_train,
    )

    if result.get("success"):
        logger.info(f"\n{'='*60}")
        logger.info("RETRAIN COMPLETE")
        logger.info(f"  Agents trained: {result['agents_trained']}")
        logger.info(f"  Window used   : {result['window_used']} bars")
        logger.info(f"  Elapsed       : {result['elapsed_sec']:.1f}s")
        logger.info(f"\nBacktest Results:")
        for agent, metrics in result.get("backtest_metrics", {}).items():
            logger.info(
                f"  {agent:6s}: Return={metrics.get('total_return', 0):.2%} | "
                f"Sharpe={metrics.get('sharpe_ratio', 0):.3f}"
            )
        logger.info(f"{'='*60}")
        logger.info("\nModels updated. Restart the dashboard to use new weights.")
        logger.info("(Or click 'Reload Models' in the dashboard sidebar.)")
    else:
        logger.error(f"Retrain failed: {result.get('error')}")


def main():
    parser = argparse.ArgumentParser(description="Train all RL trading agents")
    parser.add_argument("--ticker", default="^NSEI", help="Stock symbol to train on")
    parser.add_argument("--period", default="2y", help="Historical data period")
    parser.add_argument("--interval", default=config.data.default_interval,
                        help="Data interval, for example 5m or 1d")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Training timesteps (initial) or retrain timesteps")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with 10k timesteps")
    parser.add_argument("--skip", nargs="+", default=[],
                        help="Agents to skip: dqn ddqn ppo a2c")
    parser.add_argument("--retrain", action="store_true",
                        help="Run sliding window end-of-day retraining instead of full training")
    parser.add_argument("--force", action="store_true",
                        help="Force retrain even without enough new bars (retrain mode only)")

    args = parser.parse_args()

    if args.quick:
        args.timesteps = 10_000
        logger.info("Quick mode: 10k timesteps")

    if args.retrain:
        # If retrain mode and no explicit timesteps, use retrain config
        if not any("--timesteps" in s for s in sys.argv):
            args.timesteps = config.retrain.retrain_timesteps
        run_retrain(args)
    else:
        run_initial_training(args)


if __name__ == "__main__":
    main()