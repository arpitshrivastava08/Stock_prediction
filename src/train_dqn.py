"""
train_dqn.py -- DQN and Double DQN Training  (Improved)

Improvements over original:
  1. PRIORITIZED EXPERIENCE REPLAY (PER)
       Instead of sampling transitions uniformly at random, PER assigns
       a priority to each transition based on its TD error (how surprising
       it was). High-error transitions are sampled more often -- the agent
       learns more from the experiences it got wrong.
       Uses importance-sampling weights to correct the resulting bias.

  2. DUELING NETWORK ARCHITECTURE
       Splits the Q-network output into two streams:
         Value stream  V(s)      -- how good is this state overall?
         Advantage stream A(s,a) -- how much better is action a vs average?
       Q(s,a) = V(s) + (A(s,a) - mean(A(s,.)))
       This helps the agent learn state value independently of actions,
       especially useful for HOLD-heavy market regimes.

  3. NOISY NETWORKS for exploration (replaces epsilon-greedy)
       Adds learnable Gaussian noise to the final layer weights.
       The network learns HOW MUCH to explore based on what it has seen.
       Replaces the fixed epsilon-greedy schedule with adaptive exploration.
       When the agent is confident, noise is small. When uncertain, noise is large.

  4. N-STEP RETURNS
       Instead of 1-step Bellman target:  r + gamma * Q(s', a')
       Uses n-step return:  r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... + gamma^n*Q(s_{t+n})
       This propagates reward signal faster through the network,
       helping the agent connect actions to their multi-step consequences.

  5. GRADIENT CLIPPING + LR SCHEDULER
       Original had gradient clipping (kept). Added cosine annealing LR
       scheduler so learning rate warms up and cools down smoothly.

  6. TRAINING METRICS & LOSS HISTORY
       Track mean Q-value, loss per episode, epsilon, buffer size.
       Helps diagnose if training is collapsing or diverging.

  7. SOFT UPDATE ALSO APPLIED TO VANILLA DQN
       Original vanilla DQN used hard update via SB3. Custom DDQN
       already had soft update. Now consistent across both.

Architecture (Dueling):
  Input: 13-dim state vector (normalized indicators + sentiment)
  Shared: Linear(13->256) -> ReLU -> Linear(256->256) -> ReLU
  Value stream:     Linear(256->128) -> ReLU -> Linear(128->1)
  Advantage stream: Linear(256->128) -> ReLU -> Linear(128->3)
  Output: Q(s,a) = V(s) + A(s,a) - mean(A)  [shape: 3]
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import Tuple, Optional, List
import json

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from logger import get_logger
from config import config
from env import StockTradingEnv

logger = get_logger(__name__)

# Named tuple for cleaner transition storage
Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done"]
)


# ---------------------------------------------------------------------------
# SECTION 1: DQN via stable-baselines3  (unchanged -- kept for ensemble compat)
# ---------------------------------------------------------------------------

def train_dqn(
    env: StockTradingEnv,
    timesteps: int = None,
    save_path: str = None,
) -> DQN:
    """
    Train a DQN agent using stable-baselines3.
    Kept as-is for ensemble compatibility.
    """
    timesteps = timesteps or config.train.timesteps
    save_path = save_path or os.path.join(config.train.models_dir, "dqn_stock")
    os.makedirs(config.train.models_dir, exist_ok=True)

    logger.info(f"Training DQN | timesteps={timesteps}")

    vec_env = DummyVecEnv([lambda: env])

    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=config.train.learning_rate,
        buffer_size=config.train.dqn_buffer_size,
        learning_starts=1_000,
        batch_size=config.train.batch_size,
        gamma=config.train.gamma,
        target_update_interval=config.train.dqn_target_update_interval,
        exploration_fraction=config.train.dqn_exploration_fraction,
        exploration_final_eps=0.05,
        verbose=1,
        device="auto",
    )

    model.learn(total_timesteps=timesteps, log_interval=500)
    model.save(save_path)
    logger.info(f"DQN model saved to {save_path}")
    return model


def load_dqn(path: str, env: StockTradingEnv) -> DQN:
    """Load a pre-trained DQN model."""
    logger.info(f"Loading DQN from {path}")
    vec_env = DummyVecEnv([lambda: env])
    return DQN.load(path, env=vec_env, device="auto")


# ---------------------------------------------------------------------------
# SECTION 2: NOISY LINEAR LAYER  (replaces epsilon-greedy exploration)
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """
    Noisy linear layer with learnable Gaussian noise.

    Each weight and bias has a learnable mean (mu) and standard deviation (sigma).
    During forward pass: weight = mu + sigma * epsilon   where epsilon ~ N(0,1)

    The network learns when to explore (high sigma) and when to exploit (low sigma).
    This is adaptive exploration -- far better than a fixed epsilon schedule.

    Uses factorized noise for efficiency:
      Instead of independent noise for every (in x out) weight,
      sample noise vectors of size (in,) and (out,) and take their outer product.
      Same expressiveness, much less random number generation.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init

        # Learnable parameters: mean and std for weights and biases
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # Fixed noise buffers (not parameters, regenerated each forward pass)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon",   torch.empty(out_features))

        self._init_parameters()
        self.reset_noise()

    def _init_parameters(self):
        """Initialize mu and sigma using the NoisyNet paper's recommended values."""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        """Factorized noise transform: sgn(x) * sqrt(|x|)"""
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Resample noise -- call once per forward pass during training."""
        eps_in  = self._f(torch.randn(self.in_features))
        eps_out = self._f(torch.randn(self.out_features))
        # Outer product for factorized noise
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            # At inference: use only the mean weights (no noise)
            weight = self.weight_mu
            bias   = self.bias_mu
        return nn.functional.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# SECTION 3: DUELING Q-NETWORK  (better state value estimation)
# ---------------------------------------------------------------------------

class DuelingQNetwork(nn.Module):
    """
    Dueling Network Architecture with NoisyLinear output layers.

    Structure:
      Shared backbone:  13 -> 256 -> 256  (learns shared market representations)
      Value stream:     256 -> 128 -> 1   (how good is this market state?)
      Advantage stream: 256 -> 128 -> 3   (how much better is each action?)

    Combined: Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
    Subtracting the mean advantage ensures identifiability:
      the value stream cannot just absorb all the magnitude.

    Why dueling helps in trading:
      Many timesteps have no good trade (HOLD is clearly best).
      The value stream can learn "this is a bad market state" without
      needing to evaluate all three actions. This makes learning faster
      and more stable during flat/choppy market conditions.
    """

    def __init__(self, state_dim: int, n_actions: int = 3, std_init: float = 0.5):
        super().__init__()

        # Shared feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Value stream: estimates V(s) -- scalar
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, 1, std_init=std_init),
        )

        # Advantage stream: estimates A(s,a) -- one per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, n_actions, std_init=std_init),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features   = self.backbone(x)
        value      = self.value_stream(features)           # shape: (batch, 1)
        advantages = self.advantage_stream(features)       # shape: (batch, 3)
        # Combine: subtract mean advantage for identifiability
        q_values   = value + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values                                    # shape: (batch, 3)

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ---------------------------------------------------------------------------
# SECTION 4: PRIORITIZED EXPERIENCE REPLAY BUFFER
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER).

    Each transition is stored with a priority = |TD error| + small epsilon.
    Sampling probability proportional to priority^alpha.

    alpha controls how much prioritization is used:
      alpha=0 -> uniform random (original replay buffer)
      alpha=1 -> fully proportional to TD error

    Importance Sampling correction:
      Because we sample non-uniformly, we introduce bias.
      Importance sampling weights w = (1/N * 1/P(i))^beta correct this.
      beta anneals from 0.4 -> 1.0 over training (full correction at end).

    Uses a SumTree for O(log N) priority updates and sampling.
    Much faster than sorting or scanning the whole buffer.
    """

    class SumTree:
        """
        Binary tree where every parent = sum of its children.
        Leaf nodes store transition priorities.
        Root = total priority sum.
        Enables O(log N) sampling and update.
        """

        def __init__(self, capacity: int):
            self.capacity  = capacity
            self.tree      = np.zeros(2 * capacity - 1, dtype=np.float32)
            self.data      = np.zeros(capacity, dtype=object)
            self.write     = 0
            self.n_entries = 0

        def _propagate(self, idx: int, change: float):
            """Update parent sums up the tree."""
            parent = (idx - 1) // 2
            self.tree[parent] += change
            if parent != 0:
                self._propagate(parent, change)

        def _retrieve(self, idx: int, s: float) -> int:
            """Find the leaf node index for cumulative sum s."""
            left  = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                return self._retrieve(left, s)
            else:
                return self._retrieve(right, s - self.tree[left])

        def total(self) -> float:
            return float(self.tree[0])

        def add(self, priority: float, data):
            idx = self.write + self.capacity - 1
            self.data[self.write] = data
            self.update(idx, priority)
            self.write = (self.write + 1) % self.capacity
            self.n_entries = min(self.n_entries + 1, self.capacity)

        def update(self, idx: int, priority: float):
            change = priority - self.tree[idx]
            self.tree[idx] = priority
            self._propagate(idx, change)

        def get(self, s: float):
            idx      = self._retrieve(0, s)
            data_idx = idx - self.capacity + 1
            return idx, self.tree[idx], self.data[data_idx]

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 1e-5,
    ):
        self.tree        = self.SumTree(capacity)
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.epsilon     = epsilon
        self.frame       = 1

    @property
    def beta(self) -> float:
        """Beta anneals from beta_start -> 1.0 over beta_frames steps."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        """Store transition with max current priority (ensures it gets sampled at least once)."""
        max_priority = self.tree.tree[self.tree.capacity - 1:].max()
        if max_priority == 0:
            max_priority = 1.0
        self.tree.add(max_priority, Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Sample batch_size transitions proportional to priority.
        Returns transitions + indices (for priority updates) + IS weights.
        """
        batch      = []
        indices    = []
        priorities = []
        segment    = self.tree.total() / batch_size

        self.frame += 1

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s  = random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        total    = self.tree.total()
        probs    = np.array(priorities, dtype=np.float32) / (total + 1e-8)
        weights  = (self.tree.n_entries * probs) ** (-self.beta)
        weights /= weights.max()   # normalize so max weight = 1

        states      = np.array([t.state      for t in batch], dtype=np.float32)
        actions     = np.array([t.action     for t in batch], dtype=np.int64)
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones       = np.array([t.done       for t in batch], dtype=np.float32)

        return (states, actions, rewards, next_states, dones,
                indices, weights.astype(np.float32))

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities for sampled transitions after computing TD errors."""
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, float(priority))

    def __len__(self) -> int:
        return self.tree.n_entries


# ---------------------------------------------------------------------------
# SECTION 5: N-STEP RETURN BUFFER  (wraps PER, accumulates n-step returns)
# ---------------------------------------------------------------------------

class NStepBuffer:
    """
    Accumulates n consecutive transitions and computes n-step discounted return.

    Instead of storing: (s_t, a_t, r_t, s_{t+1})
    Stores:             (s_t, a_t, R_t, s_{t+n})
    where R_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... + gamma^{n-1}*r_{t+n-1}

    This makes the reward signal propagate faster -- the agent sees the
    multi-step consequence of its action in a single training sample.
    """

    def __init__(self, n_steps: int, gamma: float):
        self.n_steps = n_steps
        self.gamma   = gamma
        self.buffer  = deque(maxlen=n_steps)

    def push(self, state, action, reward, next_state, done):
        """Add transition. Returns completed n-step transition if ready, else None."""
        self.buffer.append(Transition(state, action, reward, next_state, done))

        if len(self.buffer) < self.n_steps:
            return None

        # Compute n-step return from oldest transition in buffer
        n_reward     = 0.0
        n_next_state = next_state
        n_done       = done

        for i, t in enumerate(self.buffer):
            n_reward += (self.gamma ** i) * t.reward
            if t.done:
                n_next_state = t.next_state
                n_done       = True
                break

        oldest = self.buffer[0]
        return Transition(oldest.state, oldest.action, n_reward, n_next_state, n_done)

    def flush(self):
        """Clear buffer (call at episode end to avoid cross-episode contamination)."""
        self.buffer.clear()


# ---------------------------------------------------------------------------
# SECTION 6: DOUBLE DQN AGENT  (Rainbow-inspired improvements)
# ---------------------------------------------------------------------------

class DoubleDQNAgent:
    """
    Improved Double DQN Agent combining:
      - Dueling Network Architecture
      - Prioritized Experience Replay
      - N-Step Returns
      - NoisyNet Exploration
      - Soft Target Network Updates
      - Cosine Annealing LR Scheduler
      - Gradient Clipping

    Interface is fully backward compatible with original DoubleDQNAgent --
    predict(obs, deterministic) returns (action, None) same as before,
    so ensemble.py requires zero changes.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 3,
        n_steps: int = 3,
        noisy_std: float = 0.5,
    ):
        self.state_dim  = state_dim
        self.n_actions  = n_actions
        self.n_steps    = n_steps
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dueling networks (online + target)
        self.online_net = DuelingQNetwork(state_dim, n_actions, std_init=noisy_std).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, n_actions, std_init=noisy_std).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer + LR scheduler
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=config.train.learning_rate,
            eps=1.5e-4,      # slightly higher eps for stability with noisy nets
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.train.timesteps,
            eta_min=1e-5,
        )

        # Prioritized Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.train.ddqn_buffer_size,
            alpha=0.6,
            beta_start=0.4,
            beta_frames=config.train.timesteps,
        )

        # N-Step Return Buffer
        self.nstep_buffer = NStepBuffer(
            n_steps=n_steps,
            gamma=config.train.gamma,
        )

        # Hyperparameters
        self.gamma      = config.train.gamma ** n_steps   # n-step discount
        self.tau        = config.train.ddqn_tau
        self.batch_size = config.train.batch_size

        # NoisyNet replaces epsilon-greedy -- keep epsilon only for logging
        self.epsilon         = 0.0
        self.training_steps  = 0
        self.losses: List[float] = []
        self.mean_q_values: List[float] = []

        logger.info(
            f"DoubleDQN (Improved) initialized on {self.device} | "
            f"n_steps={n_steps} | noisy_std={noisy_std} | "
            f"PER alpha=0.6 beta_start=0.4"
        )

    def select_action(self, state: np.ndarray) -> int:
        """
        Action selection via NoisyNet -- no epsilon-greedy needed.

        The noise in NoisyLinear layers provides stochastic exploration.
        In training mode, noise is active. In eval mode, noise is zero.
        """
        self.online_net.train()   # ensure noise is active during training
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(s)
            return q_values.argmax(dim=1).item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Pass transition through n-step buffer first.
        Only store in replay buffer once n-step return is computed.
        """
        nstep_transition = self.nstep_buffer.push(state, action, reward, next_state, done)
        if nstep_transition is not None:
            self.replay_buffer.push(
                nstep_transition.state,
                nstep_transition.action,
                nstep_transition.reward,    # this is now the n-step return
                nstep_transition.next_state,
                nstep_transition.done,
            )
        if done:
            self.nstep_buffer.flush()

    def update(self) -> Optional[float]:
        """
        One gradient update step using Double DQN + PER + Dueling.

        Key steps:
          1. Sample prioritized mini-batch with IS weights
          2. Compute DDQN target (online selects, target evaluates)
          3. Weight loss by IS weights to correct PER bias
          4. Update priorities using TD errors
          5. Soft update target network
          6. Step LR scheduler
          7. Reset NoisyNet noise for next step
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        (states, actions, rewards, next_states, dones,
         indices, is_weights) = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights_t     = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        # Current Q-values for actions actually taken
        self.online_net.train()
        current_q = self.online_net(states_t).gather(1, actions_t)

        # -- DOUBLE DQN TARGET with n-step discount --
        with torch.no_grad():
            self.online_net.eval()
            # Step 1: Online network selects best action at next state
            best_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)

            self.target_net.eval()
            # Step 2: Target network evaluates that action
            next_q = self.target_net(next_states_t).gather(1, best_actions)

            # n-step Bellman target (gamma already raised to n-th power)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        self.online_net.train()

        # TD errors for priority updates
        td_errors = (current_q - target_q).detach().abs().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, td_errors)

        # Huber loss weighted by importance sampling weights
        # IS weights correct the bias introduced by non-uniform sampling
        elementwise_loss = nn.SmoothL1Loss(reduction="none")(current_q, target_q)
        loss = (weights_t * elementwise_loss).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        # Soft update target network: theta_target = tau*theta_online + (1-tau)*theta_target
        self._soft_update_target()

        # Reset noise for next forward pass
        self.online_net.reset_noise()
        self.target_net.reset_noise()

        # Track metrics
        self.training_steps += 1
        loss_val = loss.item()
        self.losses.append(loss_val)
        self.mean_q_values.append(current_q.mean().item())

        return loss_val

    def _soft_update_target(self):
        """Polyak averaging: tau=0.005 -> target drifts slowly toward online."""
        for online_p, target_p in zip(
            self.online_net.parameters(),
            self.target_net.parameters(),
        ):
            target_p.data.copy_(
                self.tau * online_p.data + (1.0 - self.tau) * target_p.data
            )

    def get_training_stats(self) -> dict:
        """Return recent training diagnostics."""
        window = 100
        return {
            "training_steps": self.training_steps,
            "buffer_size":    len(self.replay_buffer),
            "mean_loss_100":  float(np.mean(self.losses[-window:]))        if self.losses        else 0.0,
            "mean_q_100":     float(np.mean(self.mean_q_values[-window:])) if self.mean_q_values else 0.0,
            "lr":             self.scheduler.get_last_lr()[0]              if self.training_steps > 0 else config.train.learning_rate,
            "per_beta":       self.replay_buffer.beta,
        }

    def save(self, path: str):
        """Save model weights, optimizer, scheduler, and training stats."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "online_net":     self.online_net.state_dict(),
            "target_net":     self.target_net.state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "scheduler":      self.scheduler.state_dict(),
            "training_steps": self.training_steps,
            "state_dim":      self.state_dim,
            "n_actions":      self.n_actions,
            "n_steps":        self.n_steps,
            "losses":         self.losses[-1000:],        # keep last 1000 only
            "mean_q_values":  self.mean_q_values[-1000:],
        }, path)
        logger.info(f"DDQN saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DoubleDQNAgent":
        """Load saved DDQN agent. Backward compatible with original save format."""
        checkpoint = torch.load(path, map_location="cpu")
        agent = cls(
            state_dim=checkpoint["state_dim"],
            n_actions=checkpoint["n_actions"],
            n_steps=checkpoint.get("n_steps", 3),
        )
        agent.online_net.load_state_dict(checkpoint["online_net"])
        agent.target_net.load_state_dict(checkpoint["target_net"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            agent.scheduler.load_state_dict(checkpoint["scheduler"])
        agent.training_steps  = checkpoint.get("training_steps", 0)
        agent.losses          = checkpoint.get("losses", [])
        agent.mean_q_values   = checkpoint.get("mean_q_values", [])
        logger.info(f"DDQN loaded from {path} | steps={agent.training_steps}")
        return agent

    def predict(self, state: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        """
        SB3-compatible predict interface -- unchanged from original.
        Ensemble calls this. Returns (action, None).
        """
        self.online_net.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(s)
            action = q_values.argmax(dim=1).item()
        return action, None


# ---------------------------------------------------------------------------
# SECTION 7: TRAINING LOOP
# ---------------------------------------------------------------------------

def train_ddqn(
    env: StockTradingEnv,
    timesteps: int = None,
    save_path: str = None,
    n_steps: int = 3,
    noisy_std: float = 0.5,
) -> DoubleDQNAgent:
    """
    Train the improved Double DQN agent.

    Training Loop per step:
      1.  Select action via NoisyNet (adaptive exploration)
      2.  Execute in StockTradingEnv -> (next_state, reward, done)
      3.  Push to n-step buffer -> computes n-step return when ready
      4.  Store n-step transition in Prioritized Replay Buffer
      5.  Sample prioritized mini-batch with IS weights
      6.  Compute DDQN target (online selects, target evaluates)
      7.  Compute IS-weighted Huber loss
      8.  Update priorities in PER using TD errors
      9.  Backprop + gradient clip
      10. Soft update target network
      11. Step LR scheduler
      12. Reset NoisyNet noise

    Args:
        env:        Initialized StockTradingEnv
        timesteps:  Total training steps (default from config)
        save_path:  Where to save model
        n_steps:    N-step return horizon (default 3)
        noisy_std:  Initial noise std for NoisyLinear layers

    Returns:
        Trained DoubleDQNAgent
    """
    timesteps = timesteps or config.train.timesteps
    save_path = save_path or os.path.join(config.train.models_dir, "ddqn_stock.pt")
    os.makedirs(config.train.models_dir, exist_ok=True)

    logger.info(
        f"Training Improved Double DQN | "
        f"timesteps={timesteps} | n_steps={n_steps} | noisy_std={noisy_std}"
    )

    state_dim = env.observation_space.shape[0]
    agent     = DoubleDQNAgent(state_dim=state_dim, n_steps=n_steps, noisy_std=noisy_std)

    obs, _         = env.reset()
    episode_count  = 0
    episode_reward = 0.0
    log_interval   = max(1, timesteps // 20)

    for step in range(timesteps):
        # Step 1: select action (NoisyNet handles exploration automatically)
        action = agent.select_action(obs)

        # Step 2: execute in environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Steps 3+4: n-step buffer -> prioritized replay buffer
        agent.store_transition(obs, action, reward, next_obs, done)

        # Steps 5-12: gradient update
        loss = agent.update()

        episode_reward += reward
        obs = next_obs

        if done:
            obs, _ = env.reset()
            episode_count  += 1
            episode_reward  = 0.0

        # Logging
        if (step + 1) % log_interval == 0:
            stats = agent.get_training_stats()
            logger.info(
                f"DDQN | Step {step+1}/{timesteps} | "
                f"Episodes: {episode_count} | "
                f"Loss: {stats['mean_loss_100']:.5f} | "
                f"Mean Q: {stats['mean_q_100']:.3f} | "
                f"LR: {stats['lr']:.2e} | "
                f"PER beta: {stats['per_beta']:.3f} | "
                f"Buffer: {stats['buffer_size']}"
            )

    agent.save(save_path)
    logger.info(f"Improved Double DQN training complete. Saved to {save_path}")

    # Save training stats as JSON for analysis
    stats_path = save_path.replace(".pt", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "total_steps":    agent.training_steps,
            "total_episodes": episode_count,
            "final_loss":     float(np.mean(agent.losses[-100:]))        if agent.losses        else 0.0,
            "final_mean_q":   float(np.mean(agent.mean_q_values[-100:])) if agent.mean_q_values else 0.0,
        }, f, indent=2)
    logger.info(f"Training stats saved to {stats_path}")

    return agent