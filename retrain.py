import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(__file__))

from logger import get_logger
from config import config
from data_loader import StockDataLoader
from features import FeatureEngineer
from sentiment import SentimentAnalyzer
from env import StockTradingEnv
from backtest import Backtester

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RETRAIN STATE — tracks last trained position per ticker
# ─────────────────────────────────────────────────────────────────────────────

class RetrainState:
    """
    Persists the "last trained bar" index for each ticker.
    Ensures we only retrain when enough new data has arrived.
    """

    def __init__(self, state_file: str = None):
        self.state_file = state_file or config.retrain.state_file
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        self._state: Dict = self._load()

    def get_last_trained_bar(self, ticker: str) -> int:
        """Return the number of bars that were present during last training."""
        return self._state.get(ticker, {}).get("last_bar_count", 0)

    def get_last_trained_time(self, ticker: str) -> Optional[str]:
        """Return ISO timestamp of last retraining."""
        return self._state.get(ticker, {}).get("last_trained_at")

    def update(self, ticker: str, bar_count: int, metrics: Dict = None):
        """Record that retraining completed at this bar count."""
        self._state[ticker] = {
            "last_bar_count": bar_count,
            "last_trained_at": datetime.now().isoformat(),
            "metrics": metrics or {},
        }
        self._save()

    def needs_retrain(self, ticker: str, current_bar_count: int) -> bool:
        """
        Return True if enough new bars have arrived since last training.
        Threshold: config.retrain.min_new_bars
        """
        last = self.get_last_trained_bar(ticker)
        new_bars = current_bar_count - last
        if new_bars >= config.retrain.min_new_bars:
            logger.info(
                f"[{ticker}] {new_bars} new bars since last training "
                f"(threshold={config.retrain.min_new_bars}) → retrain needed"
            )
            return True
        logger.info(
            f"[{ticker}] Only {new_bars} new bars since last training "
            f"(threshold={config.retrain.min_new_bars}) → skip"
        )
        return False

    def _load(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save(self):
        try:
            with open(self.state_file, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save retrain state: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDING WINDOW RETRAINER
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowRetrainer:

    def __init__(self):
        self.loader = StockDataLoader()
        self.fe = FeatureEngineer()
        self.sentiment = SentimentAnalyzer()
        self.retrain_state = RetrainState()
        self.cfg = config.retrain

    def run(
        self,
        ticker: str = "^NSEI",
        period: str = "2y",
        force: bool = False,
        timesteps: int = None,
        agents: list = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        
        start_time = time.time()
        timesteps = timesteps or self.cfg.retrain_timesteps
        agents = agents or self.cfg.retrain_agents

        def progress(step, total, msg):
            pct = int(step / total * 100)
            logger.info(f"[Retrain] {pct}% — {msg}")
            if progress_callback:
                progress_callback(step, total, msg)

        logger.info("=" * 60)
        logger.info(f"SLIDING WINDOW RETRAIN: {ticker}")
        logger.info(f"Timesteps: {timesteps} | Agents: {agents}")
        logger.info("=" * 60)

        total_steps = 6

        # ── Step 1: Load Data ─────────────────────────────────────────
        progress(1, total_steps, "Loading market data...")
        raw = self.loader.load(ticker, period=period, force_download=True)
        if raw is None or raw.empty:
            return {"success": False, "error": "No data loaded"}

        # ── Step 2: Check if retrain needed ───────────────────────────
        current_bar_count = len(raw)
        if not force and not self.retrain_state.needs_retrain(ticker, current_bar_count):
            return {
                "success": False,
                "error": f"Not enough new data (need {self.cfg.min_new_bars} new bars)",
                "new_bars": current_bar_count - self.retrain_state.get_last_trained_bar(ticker),
            }

        # ── Step 3: Compute Features ──────────────────────────────────
        progress(2, total_steps, "Computing technical features...")
        df = self.fe.compute_all(raw)
        if df.empty:
            return {"success": False, "error": "Feature computation failed"}

        # ── Step 4: Sliding Window ────────────────────────────────────
        progress(3, total_steps, "Applying sliding window...")
        window = self.cfg.window_size
        if window > 0 and len(df) > window:
            df = df.iloc[-window:].reset_index(drop=True)
            logger.info(f"Sliding window applied: using last {window} bars")
        else:
            logger.info(f"Using all {len(df)} bars (window_size={window})")

        # ── Step 5: Sentiment Scores ──────────────────────────────────
        base_score = self.sentiment.get_score(ticker)
        np.random.seed(int(time.time()) % 1000)
        noise = np.random.normal(0, 0.05, len(df))
        sentiment_scores = np.clip(base_score + noise, -1, 1).tolist()

        # ── Step 6: Train/Test split ──────────────────────────────────
        split_idx = int(len(df) * config.backtest.train_split)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df  = df.iloc[split_idx:].reset_index(drop=True)
        train_sent = sentiment_scores[:split_idx]
        test_sent  = sentiment_scores[split_idx:]

        logger.info(f"Train: {len(train_df)} bars | Test: {len(test_df)} bars")

        train_env = StockTradingEnv(train_df, train_sent)

        # ── Step 7: Retrain Each Agent ────────────────────────────────
        progress(4, total_steps, f"Retraining {len(agents)} agents...")
        trained_models = {}
        training_results = {}

        os.makedirs(config.train.models_dir, exist_ok=True)

        for agent_name in agents:
            try:
                logger.info(f"  Retraining {agent_name}...")
                t0 = time.time()
                model = self._retrain_agent(agent_name, train_env, timesteps)
                elapsed = time.time() - t0

                if model is not None:
                    trained_models[agent_name] = model
                    training_results[agent_name] = {
                        "status": "success",
                        "elapsed_sec": round(elapsed, 1),
                    }
                    logger.info(f"  ✓ {agent_name} retrained in {elapsed:.1f}s")
                else:
                    training_results[agent_name] = {"status": "skipped"}

            except Exception as e:
                logger.error(f"  ✗ {agent_name} retraining failed: {e}")
                training_results[agent_name] = {
                    "status": "failed",
                    "error": str(e),
                }

        if not trained_models:
            return {"success": False, "error": "All agent retraining failed", "details": training_results}

        # ── Step 8: Quick Backtest ────────────────────────────────────
        progress(5, total_steps, "Running quick backtest...")
        backtest_metrics = {}
        sharpe_scores = {}

        try:
            test_env = StockTradingEnv(test_df, test_sent)
            backtester = Backtester(test_env)

            for name, model in trained_models.items():
                r = backtester.run(model, label=name)
                backtest_metrics[name] = {
                    "total_return": r.get("total_return", 0),
                    "sharpe_ratio": r.get("sharpe_ratio", 0),
                    "max_drawdown": r.get("max_drawdown", 0),
                }
                sharpe_scores[name] = r.get("sharpe_ratio", 0)
                logger.info(
                    f"  [{name}] Return={r.get('total_return',0):.2%} | "
                    f"Sharpe={r.get('sharpe_ratio',0):.3f}"
                )

        except Exception as e:
            logger.warning(f"Backtest failed (non-fatal): {e}")

        # ── Step 9: Update Ensemble Weights ───────────────────────────
        progress(6, total_steps, "Updating ensemble weights...")
        if self.cfg.update_weights_after_retrain and sharpe_scores:
            self._update_ensemble_weights(sharpe_scores)

        # ── Step 10: Save Summary ─────────────────────────────────────
        elapsed_total = time.time() - start_time
        summary = {
            "success": True,
            "ticker": ticker,
            "retrained_at": datetime.now().isoformat(),
            "agents_trained": list(trained_models.keys()),
            "training_results": training_results,
            "backtest_metrics": backtest_metrics,
            "elapsed_sec": round(elapsed_total, 1),
            "new_bar_count": current_bar_count,
            "window_used": len(train_df),
        }

        # Persist summary
        summary_path = os.path.join(config.train.models_dir, "retrain_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Update retrain state
        self.retrain_state.update(ticker, current_bar_count, backtest_metrics)

        logger.info(f"RETRAIN COMPLETE | elapsed={elapsed_total:.1f}s | "
                    f"agents={list(trained_models.keys())}")
        return summary

    # ------------------------------------------------------------------
    # AGENT RETRAINING
    # ------------------------------------------------------------------

    def _retrain_agent(
        self,
        agent_name: str,
        env: StockTradingEnv,
        timesteps: int,
    ):
        """
        Load an existing model and continue training (fine-tuning).

        For SB3 models: load existing .zip, call model.learn() again.
        For custom DDQN: load .pt checkpoint, run training loop.

        This is the key advantage of the sliding window approach:
        we DON'T start from random weights — we continue from where
        the last training left off, so convergence is much faster.
        """
        models_dir = config.train.models_dir

        if agent_name == "ppo":
            return self._retrain_sb3(
                "ppo", env, timesteps,
                os.path.join(models_dir, "ppo_stock")
            )

        elif agent_name == "a2c":
            return self._retrain_sb3(
                "a2c", env, timesteps,
                os.path.join(models_dir, "a2c_stock")
            )

        elif agent_name == "dqn":
            return self._retrain_sb3(
                "dqn", env, timesteps,
                os.path.join(models_dir, "dqn_stock")
            )

        elif agent_name == "ddqn":
            return self._retrain_ddqn(env, timesteps)

        else:
            logger.warning(f"Unknown agent: {agent_name}")
            return None

    def _retrain_sb3(self, algo: str, env: StockTradingEnv, timesteps: int, model_path: str):
        """
        Fine-tune an SB3 model (PPO, A2C, DQN) by:
          1. Loading existing .zip if it exists
          2. Setting the new environment
          3. Calling model.learn() for `timesteps` more steps
          4. Saving back to the same path
        """
        from stable_baselines3 import PPO, A2C, DQN
        from stable_baselines3.common.vec_env import DummyVecEnv

        algo_map = {"ppo": PPO, "a2c": A2C, "dqn": DQN}
        AlgoClass = algo_map[algo]

        vec_env = DummyVecEnv([lambda: env])
        zip_path = model_path + ".zip"

        if os.path.exists(zip_path) or os.path.exists(model_path):
            logger.info(f"  Fine-tuning existing {algo.upper()} model...")
            model = AlgoClass.load(model_path, env=vec_env, device="auto")
            model.set_env(vec_env)
        else:
            logger.info(f"  No existing {algo.upper()} model found — training from scratch...")
            if algo == "ppo":
                from train_ppo import train_ppo
                return train_ppo(env, timesteps, model_path)
            elif algo == "a2c":
                from train_ppo import train_a2c
                return train_a2c(env, timesteps, model_path)
            elif algo == "dqn":
                from train_dqn import train_dqn
                return train_dqn(env, timesteps, model_path)

        model.learn(
            total_timesteps=timesteps,
            log_interval=max(1, timesteps // 10),
            reset_num_timesteps=False,  # Continue from where we left off
        )
        model.save(model_path)
        return model

    def _retrain_ddqn(self, env: StockTradingEnv, timesteps: int):
        """Fine-tune or train custom Double DQN."""
        from train_dqn import DoubleDQNAgent, train_ddqn

        model_path = os.path.join(config.train.models_dir, "ddqn_stock.pt")

        if os.path.exists(model_path):
            logger.info("  Fine-tuning existing DDQN model...")
            agent = DoubleDQNAgent.load(model_path)

            # Continue training loop
            obs, _ = env.reset()
            episode_reward = 0.0
            log_interval = max(1, timesteps // 10)

            for step in range(timesteps):
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store_transition(obs, action, reward, next_obs, done)
                agent.update()

                episode_reward += reward
                obs = next_obs

                if done:
                    obs, _ = env.reset()
                    episode_reward = 0.0

                if (step + 1) % log_interval == 0:
                    stats = agent.get_training_stats()
                    logger.info(
                        f"  DDQN fine-tune | Step {step+1}/{timesteps} | "
                        f"Loss: {stats['mean_loss_100']:.5f}"
                    )

            agent.save(model_path)
            return agent
        else:
            return train_ddqn(env, timesteps, model_path)

    # ------------------------------------------------------------------
    # ENSEMBLE WEIGHT UPDATE
    # ------------------------------------------------------------------

    def _update_ensemble_weights(self, sharpe_scores: Dict[str, float]):
        """
        Update ensemble weights based on new Sharpe ratios.
        Persists updated weights to a JSON file so they survive restarts.
        """
        try:
            from ensemble import EnsembleEngine

            # Create a minimal ensemble just to use the update logic
            # We need at least one model — use a dummy
            class DummyModel:
                def predict(self, obs, deterministic=True):
                    return 0, None

            dummy_models = {k: DummyModel() for k in sharpe_scores}
            engine = EnsembleEngine(dummy_models)
            engine.update_weights_from_sharpe(sharpe_scores)

            # Save updated weights
            weights_path = os.path.join(config.train.models_dir, "ensemble_weights.json")
            with open(weights_path, "w") as f:
                json.dump(engine.active_weights, f, indent=2)

            # Update the live config so the next prediction uses new weights
            config.ensemble.weights.update(engine.active_weights)

            logger.info(f"Ensemble weights updated: {engine.active_weights}")

        except Exception as e:
            logger.warning(f"Could not update ensemble weights: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sliding Window End-of-Day Retraining"
    )
    parser.add_argument("--ticker", default="^NSEI", help="Stock symbol")
    parser.add_argument("--period", default="2y", help="Data period")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Retrain timesteps (default from config)")
    parser.add_argument("--agents", nargs="+", default=None,
                        help="Agents to retrain: ppo dqn ddqn a2c")
    parser.add_argument("--force", action="store_true",
                        help="Force retrain even without enough new bars")
    parser.add_argument("--window", type=int, default=None,
                        help="Override window_size from config")

    args = parser.parse_args()

    if args.window:
        config.retrain.window_size = args.window

    retrainer = SlidingWindowRetrainer()
    result = retrainer.run(
        ticker=args.ticker,
        period=args.period,
        force=args.force,
        timesteps=args.timesteps,
        agents=args.agents,
    )

    print("\n" + "=" * 60)
    if result.get("success"):
        print(f"✓ RETRAIN COMPLETE")
        print(f"  Ticker      : {result['ticker']}")
        print(f"  Agents      : {result['agents_trained']}")
        print(f"  Window used : {result['window_used']} bars")
        print(f"  Elapsed     : {result['elapsed_sec']:.1f}s")
        print("\nBacktest Results:")
        for agent, metrics in result.get("backtest_metrics", {}).items():
            print(
                f"  {agent:6s}: Return={metrics.get('total_return', 0):.2%} | "
                f"Sharpe={metrics.get('sharpe_ratio', 0):.3f}"
            )
    else:
        print(f"✗ RETRAIN FAILED: {result.get('error')}")
    print("=" * 60)


if __name__ == "__main__":
    main()