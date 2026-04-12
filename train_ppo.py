import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from logger import get_logger
from config import config
from env import StockTradingEnv

logger = get_logger(__name__)

# ensure model directory exists
os.makedirs(config.train.models_dir, exist_ok=True)


def train_ppo(
    env: StockTradingEnv,
    timesteps: int = None,
    save_path: str = None,
    existing_model_path: str = None,
) -> PPO:

    timesteps = timesteps or config.train.timesteps
    save_path = save_path or os.path.join(config.train.models_dir, "ppo_stock")

    # wrap env for stable-baselines
    vec_env = DummyVecEnv([lambda: env])

    # check if we are continuing from an existing model
    load_path = existing_model_path or save_path
    is_finetune = os.path.exists(load_path + ".zip") or os.path.exists(load_path)

    if is_finetune:
        logger.info(f"Loading PPO model from {load_path}")
        model = PPO.load(load_path, env=vec_env, device="auto")
        model.set_env(vec_env)
    else:
        logger.info(f"Training PPO for {timesteps} steps")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=config.train.learning_rate,
            n_steps=config.train.ppo_n_steps,
            batch_size=config.train.batch_size,
            n_epochs=config.train.ppo_n_epochs,
            gamma=config.train.gamma,
            gae_lambda=0.95,
            clip_range=config.train.ppo_clip_range,
            verbose=1,
            device="auto",
        )

    model.learn(
        total_timesteps=timesteps,
        log_interval=10,
        reset_num_timesteps=not is_finetune,  # continue step count if fine-tuning
    )

    model.save(save_path)
    logger.info(f"PPO model saved at {save_path}")

    return model


def load_ppo(path: str, env: StockTradingEnv) -> PPO:
    # load PPO with env attached
    vec_env = DummyVecEnv([lambda: env])
    return PPO.load(path, env=vec_env, device="auto")


def train_a2c(
    env: StockTradingEnv,
    timesteps: int = None,
    save_path: str = None,
    existing_model_path: str = None,
) -> A2C:

    timesteps = timesteps or config.train.timesteps
    save_path = save_path or os.path.join(config.train.models_dir, "a2c_stock")

    vec_env = DummyVecEnv([lambda: env])

    # same logic as PPO for re-training
    load_path = existing_model_path or save_path
    is_finetune = os.path.exists(load_path + ".zip") or os.path.exists(load_path)

    if is_finetune:
        logger.info(f"Loading A2C model from {load_path}")
        model = A2C.load(load_path, env=vec_env, device="auto")
        model.set_env(vec_env)
    else:
        logger.info(f"Training A2C for {timesteps} steps")
        model = A2C(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=config.train.learning_rate,
            n_steps=config.train.a2c_n_steps,
            gamma=config.train.gamma,
            verbose=1,
            device="auto",
        )

    model.learn(
        total_timesteps=timesteps,
        log_interval=100,
        reset_num_timesteps=not is_finetune,
    )

    model.save(save_path)
    logger.info(f"A2C model saved at {save_path}")

    return model


def load_a2c(path: str, env: StockTradingEnv) -> A2C:
    # load A2C model
    vec_env = DummyVecEnv([lambda: env])
    return A2C.load(path, env=vec_env, device="auto")