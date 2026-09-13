"""Lightweight mock QWOP environment for testing without browser overhead."""
import time
import numpy as np
from typing import Dict, Tuple, Any, Optional
from .base import BaseQWOPEnv


class MockQWOPEnv(BaseQWOPEnv):
    """
    Mock QWOP environment simulating physics without browser.

    Useful for:
    - Testing training code without browser overhead
    - Development and debugging
    - CI/CD pipelines
    """

    def __init__(self, difficulty: float = 1.0, seed: int = None):
        """
        Initialize mock environment.

        Args:
            difficulty: Physics difficulty (1.0 = normal)
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.difficulty = difficulty
        self.rng = np.random.RandomState(seed)

        # Physics parameters
        self.gravity = -9.8 * difficulty
        self.joint_damping = 0.1
        self.max_joint_force = 1.0

        # Current state
        self._reset_state()
        self.max_fps = 8
        self.last_frame_time = time.time()
        self.step_count = 0
        self.max_steps = 500  # Episode length

    def _reset_state(self):
        """Reset internal state."""
        self.state = {
            "posx": 0.0,
            "posy": 0.0,
            "velx": 0.0,
            "vely": 0.0,
            "head": 0.0,
            "torso": 0.0,
            "leftarm": -0.3,
            "rightarm": -0.3,
            "leftthigh": 0.2,
            "rightthigh": 0.2,
            "leftcalf": -0.2,
            "rightcalf": -0.2,
            "terminal": False,
        }
        self.joint_velocities = {k: 0.0 for k in ["head", "torso", "leftarm", "rightarm",
                                                     "leftthigh", "rightthigh", "leftcalf", "rightcalf"]}
        self.last_state_time = time.time()
        self.old_posx = 0.0
        self.old_velx = 0.0

    def get_state(self) -> Dict[str, float]:
        """Get current state."""
        current_time = time.time()
        delta = current_time - self.last_state_time
        self.last_state_time = current_time

        # Update position tracking
        self.state["diffx"] = self.state["posx"] - self.old_posx
        self.state["velx"] = self.state["diffx"] / max(delta, 0.001)
        self.state["accelx"] = (self.state["velx"] - self.old_velx) / max(delta, 0.001)

        self.old_posx = self.state["posx"]
        self.old_velx = self.state["velx"]

        return self.state.copy()

    def compute_reward(self, state: Dict[str, float] = None) -> float:
        """
        Reward is horizontal distance traveled per timestep.
        Penalty for falling over.
        """
        state = state or self.get_state()

        current_time = time.time()
        delta = current_time - self.last_reward if hasattr(self, 'last_reward') else 0.1
        self.last_reward = current_time

        # Reward: forward progress
        reward = state["diffx"] / max(delta, 0.001)

        # Penalty for falling
        if state["terminal"]:
            reward = -1.0

        return float(reward)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the environment.

        Args:
            action: [Q, W, O, P] key presses (0 or 1)

        Returns:
            obs, reward, terminated, truncated, info
        """
        self._throttle()
        self.step_count += 1

        # Unpack action
        q_pressed, w_pressed, o_pressed, p_pressed = action

        # Simple physics: pressing keys applies forces to joints
        current_time = time.time()
        delta = current_time - self.last_state_time

        # Update joint angles based on key presses
        if q_pressed:
            self.joint_velocities["leftarm"] = min(self.joint_velocities["leftarm"] + 0.1, 0.5)
        else:
            self.joint_velocities["leftarm"] *= 0.9

        if w_pressed:
            self.joint_velocities["rightarm"] = min(self.joint_velocities["rightarm"] + 0.1, 0.5)
        else:
            self.joint_velocities["rightarm"] *= 0.9

        if o_pressed:
            self.joint_velocities["leftthigh"] = min(self.joint_velocities["leftthigh"] + 0.1, 0.5)
        else:
            self.joint_velocities["leftthigh"] *= 0.9

        if p_pressed:
            self.joint_velocities["rightthigh"] = min(self.joint_velocities["rightthigh"] + 0.1, 0.5)
        else:
            self.joint_velocities["rightthigh"] *= 0.9

        # Update joint angles
        for joint in self.state:
            if joint in self.joint_velocities:
                self.state[joint] += self.joint_velocities[joint] * delta
                self.state[joint] = np.clip(self.state[joint], -np.pi, np.pi)

        # Simple forward locomotion: pressing opposite legs moves forward
        leg_force = (o_pressed + p_pressed - abs(o_pressed - p_pressed)) * 0.5
        self.state["velx"] += leg_force * delta * self.difficulty
        self.state["posx"] += self.state["velx"] * delta

        # Vertical position (rough simulation)
        self.state["vely"] += self.gravity * delta
        self.state["posy"] += self.state["vely"] * delta

        # Collision detection (terminated = episode-ending condition)
        terminated = self.state["posy"] < -1.0

        # Episode length limit (truncated = timeout)
        truncated = self.step_count >= self.max_steps

        self.state["terminal"] = terminated or truncated

        # Get observation and reward
        state = self.get_state()
        obs = self.obs(state)
        reward = self.compute_reward(state)

        return obs, reward, terminated, truncated, {"state": state}

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        self._reset_state()
        self.step_count = 0
        self.last_reward = time.time()
        self.last_frame_time = time.time()
        return self.obs(), {}

    def _throttle(self):
        """Maintain frame rate."""
        current_time = time.time()
        delta = current_time - self.last_frame_time
        min_delta = 1.0 / self.max_fps
        if delta < min_delta:
            time.sleep(min_delta - delta)
        self.last_frame_time = time.time()

    def close(self):
        """No cleanup needed for mock environment."""
        pass
