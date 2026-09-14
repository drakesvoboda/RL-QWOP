"""QWOP Gymnasium environments with real and mock implementations."""
import gymnasium as gym
from gymnasium.envs.registration import register

# Register only if not already registered
def _register_env(id, entry_point):
    if id not in gym.registry:
        register(id=id, entry_point=entry_point)

# Real QWOP (requires game server + Selenium + Firefox)
_register_env('qwop-v0', 'gym_qwop.envs.qwop_env:QWOPEnv')

# WebSocket QWOP (requires game server + Selenium + Firefox; 10-20x faster stepping than qwop-v0)
_register_env('qwop-websocket-v0', 'gym_qwop.envs.qwop_websocket:WebSocketQWOPEnv')


__all__ = ['register', 'QWOPEnv', 'WebSocketQWOPEnv']