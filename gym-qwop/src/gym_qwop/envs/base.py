"""Base environment interface for QWOP."""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import numpy as np
import gymnasium
from gymnasium import spaces


class BaseQWOPEnv(gymnasium.Env, ABC):
    """Abstract base class for QWOP environments."""

    metadata = {'render.modes': ['human']}

    def __init__(self):
        """Initialize observation and action spaces."""
        self.action_space = spaces.MultiBinary(4)  # Q, W, O, P keys

        # State vector: [posy, head, torso, leftarm, rightarm, leftthigh, rightthigh, leftcalf, rightcalf]
        high = np.array([np.inf] * 9)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.last_reward = 0.0
        self.last_state_time = 0.0
        self.old_posx = 0.0
        self.old_velx = 0.0

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Returns:
            obs: observation
            reward: reward
            terminated: episode finished (e.g., fell over)
            truncated: episode timeout (max steps reached)
            info: additional info dict
        """
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            obs: initial observation
            info: additional info dict
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, float]:
        """Get current state from the environment."""
        pass

    @abstractmethod
    def compute_reward(self, state: Dict[str, float] = None) -> float:
        """Compute reward based on state."""
        pass

    def obs(self, state: Dict[str, float] = None) -> np.ndarray:
        """Convert state to observation vector."""
        state = state or self.get_state()
        return np.array([
            state.get("posy", 0),
            state.get("head", 0),
            state.get("torso", 0),
            state.get("leftarm", 0),
            state.get("rightarm", 0),
            state.get("leftthigh", 0),
            state.get("rightthigh", 0),
            state.get("leftcalf", 0),
            state.get("rightcalf", 0),
        ], dtype=np.float32)

    def render(self, mode='human') -> None:
        """Render environment (optional)."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass
