
import os
import json
import numpy as np
from typing import Dict, Optional, Tuple, Any

from logger import get_logger
from config import config
from env import StockTradingEnv, ACTION_NAMES, HOLD, BUY, SELL

logger = get_logger(__name__)

# Path where retrain.py saves updated weights
WEIGHTS_FILE = os.path.join(config.train.models_dir, "ensemble_weights.json")


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleEngine:
   

    def __init__(
        self,
        models: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ):
      
        self.models = models

        # Priority: explicit weights → persisted weights → config defaults
        if weights is not None:
            self.weights = weights
        else:
            self.weights = self._load_persisted_weights() or config.ensemble.weights

        # Validate weights sum to ~1.0
        total = sum(self.weights.get(k, 0.0) for k in self.models)
        if total < 0.01:
            raise ValueError(
                "Ensemble weights sum to ~0. Check model names match weight keys."
            )

        # Normalize weights (in case not all models are loaded)
        active = {k: self.weights.get(k, 0.0) for k in self.models if k in self.weights}
        total_active = sum(active.values())
        self.active_weights = {k: v / total_active for k, v in active.items()}

        logger.info(
            f"Ensemble initialized with {len(self.models)} agents: "
            f"{list(self.models.keys())}"
        )
        logger.info(f"Normalized weights: {self.active_weights}")

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, float, Dict]:
      
        vote_scores = {HOLD: 0.0, BUY: 0.0, SELL: 0.0}
        agent_actions = {}

        for name, model in self.models.items():
            weight = self.active_weights.get(name, 0.0)
            if weight == 0.0:
                continue

            try:
                action, _ = model.predict(obs, deterministic=deterministic)
                action = int(action)
                action = max(HOLD, min(SELL, action))

                agent_actions[name] = {
                    "action": action,
                    "action_name": ACTION_NAMES[action],
                    "weight": weight,
                }
                vote_scores[action] += weight

            except Exception as e:
                logger.warning(f"Agent '{name}' prediction failed: {e}. Defaulting to HOLD.")
                agent_actions[name] = {
                    "action": HOLD,
                    "action_name": "HOLD",
                    "weight": weight,
                    "error": str(e),
                }
                vote_scores[HOLD] += weight

        # Final decision: highest weighted vote wins
        final_action = max(vote_scores, key=vote_scores.get)
        confidence = vote_scores[final_action]

        breakdown = {
            "vote_scores": {ACTION_NAMES[k]: round(v, 4) for k, v in vote_scores.items()},
            "agent_votes": agent_actions,
            "final_action": ACTION_NAMES[final_action],
            "confidence": round(confidence, 4),
        }

        logger.debug(
            f"Ensemble vote: HOLD={vote_scores[0]:.3f} "
            f"BUY={vote_scores[1]:.3f} "
            f"SELL={vote_scores[2]:.3f} "
            f"→ {ACTION_NAMES[final_action]} (conf={confidence:.3f})"
        )

        return final_action, confidence, breakdown

    def update_weights_from_sharpe(
        self,
        sharpe_scores: Dict[str, float],
    ):
      
        min_weight = 0.05  # Floor so no agent gets completely ignored

        min_sharpe = min(sharpe_scores.values())
        shifted = {k: max(0.01, v - min_sharpe) for k, v in sharpe_scores.items()}
        total = sum(shifted.values())

        new_weights = {}
        for name in self.models:
            if name in shifted:
                raw = shifted[name] / total
                new_weights[name] = max(min_weight, raw)

        total_new = sum(new_weights.values())
        self.active_weights = {k: v / total_new for k, v in new_weights.items()}

        logger.info(f"Ensemble weights updated from Sharpe ratios: {self.active_weights}")

    def evaluate_agents(
        self,
        env: StockTradingEnv,
        n_episodes: int = 3,
    ) -> Dict[str, Dict]:
        """
        Run each agent independently and compute their performance metrics.
        Used for dynamic weight adjustment.
        """
        results = {}

        for name, model in self.models.items():
            logger.info(f"Evaluating agent: {name}")
            all_metrics = []

            for ep in range(n_episodes):
                obs, _ = env.reset()
                done = False

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(int(action))
                    done = terminated or truncated

                metrics = env.compute_metrics()
                all_metrics.append(metrics)

            avg = {
                k: np.mean([m[k] for m in all_metrics if k in m])
                for k in ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]
            }
            results[name] = avg
            logger.info(
                f"  {name}: Sharpe={avg['sharpe_ratio']:.2f}, "
                f"Return={avg['total_return']:.2%}"
            )

        return results

    @staticmethod
    def _load_persisted_weights() -> Optional[Dict[str, float]]:
        """
        Load weights saved by the retrainer after last end-of-day update.
        Returns None if no persisted weights exist yet.
        """
        if os.path.exists(WEIGHTS_FILE):
            try:
                with open(WEIGHTS_FILE, "r") as f:
                    weights = json.load(f)
                logger.info(f"Loaded persisted ensemble weights from {WEIGHTS_FILE}")
                return weights
            except Exception as e:
                logger.warning(f"Could not load persisted weights: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER — loads all saved models from disk
# ─────────────────────────────────────────────────────────────────────────────

def load_all_models(env: StockTradingEnv) -> Dict[str, Any]:
    """
    Load all trained RL models from disk.

    Returns dict with only models that exist (graceful partial loading).
    This allows running the ensemble even if not all models are trained.
    """
    from train_dqn import DoubleDQNAgent, load_dqn
    from train_ppo import load_ppo, load_a2c

    models_dir = config.train.models_dir
    models = {}

    def try_load(name, loader_fn, path):
        if os.path.exists(path) or os.path.exists(path + ".zip"):
            try:
                models[name] = loader_fn(path, env)
                logger.info(f"✓ Loaded {name}")
            except Exception as e:
                logger.warning(f"✗ Failed to load {name}: {e}")
        else:
            logger.warning(f"✗ {name} model not found at {path}")

    try_load("ppo",  load_ppo,  os.path.join(models_dir, "ppo_stock"))
    try_load("a2c",  load_a2c,  os.path.join(models_dir, "a2c_stock"))
    try_load("dqn",  load_dqn,  os.path.join(models_dir, "dqn_stock"))

    ddqn_path = os.path.join(models_dir, "ddqn_stock.pt")
    if os.path.exists(ddqn_path):
        try:
            models["ddqn"] = DoubleDQNAgent.load(ddqn_path)
            logger.info("✓ Loaded ddqn")
        except Exception as e:
            logger.warning(f"✗ Failed to load ddqn: {e}")

    if not models:
        logger.error("No models loaded! Please train models first.")
    else:
        logger.info(f"Loaded {len(models)} models: {list(models.keys())}")

    return models


def build_ensemble(env: StockTradingEnv) -> Optional[EnsembleEngine]:
    """
    Convenience function: load all models and return ensemble.
    Returns None if no models are available.
    """
    models = load_all_models(env)
    if not models:
        return None
    return EnsembleEngine(models)