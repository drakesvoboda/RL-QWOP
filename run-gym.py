import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2


env = make_vec_env('gym_qwop:qwop-v0', n_envs=4)
# env = gym.make('gym_qwop:multi-frame-qwop-v0')
model = PPO2(MlpLstmPolicy, env, verbose=1, gamma=0.99)
model.learn(total_timesteps=500000)

model.save("ppo2_qwop-v0")

