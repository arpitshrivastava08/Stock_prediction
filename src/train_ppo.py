import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from logger import get_logger
from config import config
from env import StockTradingEnv

logger = get_logger(__name__)

os.makedirs(config.train.models_dir, exist_ok=True)


def train_ppo(
    env: StockTradingEnv,
    timesteps: int = None,
    save_path: str = None,
) -> PPO:
    """
    Train PPO model on trading environment.
    """

    timesteps = timesteps or config.train.timesteps
    save_path = save_path or os.path.join(config.train.models_dir, "ppo_stock")

    vec_env = DummyVecEnv([lambda: env])

    logger.info(f"Training PPO for {timesteps} steps")

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=config.train.learning_rate,
        n_steps=config.train.ppo_n_steps,
        batch_size=config.train.batch_size,
        n_epochs=config.train.ppo_n_epochs,
        gamma=config.train.gamma,
        verbose=1,
        device="auto",
    )

    model.learn(
        total_timesteps=timesteps,
        log_interval=10,
    )

    model.save(save_path)
    logger.info(f"PPO model saved at {save_path}")

    return model