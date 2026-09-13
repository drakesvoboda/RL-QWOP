"""QWOP Gymnasium environments with real and mock implementations."""
import gymnasium as gym
from gymnasium.envs.registration import register

# Register only if not already registered
def _register_env(id, entry_point):
    if id not in gym.registry:
        register(id=id, entry_point=entry_point)

# Mock QWOP (lightweight, no browser - great for testing/development)
_register_env('qwop-mock-v0', 'gym_qwop.envs.mock_qwop:MockQWOPEnv')

# Real QWOP (requires game server + Selenium + Firefox)
_register_env('qwop-v0', 'gym_qwop.envs.qwop_real:RealQWOPEnv')

# WebSocket QWOP (fast alternative - 10-20x speedup, requires game server with WebSocket)
_register_env('qwop-websocket-v0', 'gym_qwop.envs.qwop_websocket:WebSocketQWOPEnv')

# Aliases for backward compatibility
try:
    from gym_qwop.envs.qwop_real import RealQWOPEnv as QWOPEnv
    FrameQWOPEnv = QWOPEnv  # TODO: Implement frame observation
    MultiFrameQWOPEnv = QWOPEnv  # TODO: Implement multi-frame observation
except ImportError:
    pass

__all__ = ['register', 'QWOPEnv']