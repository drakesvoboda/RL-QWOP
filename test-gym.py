import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2


env = make_vec_env('gym_qwop:qwop-v0', n_envs=4)
model = PPO2(MlpLstmPolicy, env, verbose=1)
model.load("ppo2_qwop")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)