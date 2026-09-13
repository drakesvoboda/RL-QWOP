"""End-to-end tests for RL-QWOP training pipeline."""
import sys
import time
import numpy as np
import pytest

sys.path.insert(0, 'src')


class TestMockEnvironmentE2E:
    """Test mock environment without browser overhead."""

    def test_mock_env_creation(self):
        """Test mock environment can be created."""
        from gym_qwop.envs.mock_qwop import MockQWOPEnv

        env = MockQWOPEnv(seed=42)
        assert env is not None
        assert env.action_space is not None
        assert env.observation_space is not None
        env.close()

    def test_mock_env_reset(self):
        """Test mock environment reset."""
        from gym_qwop.envs.mock_qwop import MockQWOPEnv

        env = MockQWOPEnv()
        obs = env.reset()

        assert obs is not None
        assert obs.shape == (9,)
        assert isinstance(obs, np.ndarray)
        env.close()

    def test_mock_env_step(self):
        """Test mock environment step."""
        from gym_qwop.envs.mock_qwop import MockQWOPEnv

        env = MockQWOPEnv()
        env.reset()

        action = np.array([1, 0, 1, 0])  # Press Q and O
        obs, reward, done, info = env.step(action)

        assert obs.shape == (9,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, (bool, np.bool_))
        assert isinstance(info, dict)
        env.close()

    def test_mock_env_episode(self):
        """Test mock environment full episode."""
        from gym_qwop.envs.mock_qwop import MockQWOPEnv

        env = MockQWOPEnv(seed=42)
        obs = env.reset()

        total_reward = 0
        steps = 0
        max_steps = 100

        while steps < max_steps:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                break

        assert steps > 0
        assert steps <= max_steps
        assert total_reward is not None
        env.close()

    def test_mock_env_deterministic(self):
        """Test mock environment with seed is deterministic."""
        from gym_qwop.envs.mock_qwop import MockQWOPEnv

        # Run 1
        env1 = MockQWOPEnv(seed=123)
        obs1 = env1.reset()
        action = np.array([1, 0, 0, 0])
        obs1_after, reward1, _, _ = env1.step(action)
        env1.close()

        # Run 2
        env2 = MockQWOPEnv(seed=123)
        obs2 = env2.reset()
        obs2_after, reward2, _, _ = env2.step(action)
        env2.close()

        # Should be identical with same seed
        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(obs1_after, obs2_after)
        assert reward1 == reward2


class TestPPOWithMockEnv:
    """Test PPO training with mock environment."""

    def test_ppo_single_step(self):
        """Test PPO can run one step with mock env."""
        from gym_qwop.envs.mock_qwop import MockQWOPEnv
        from rlqwop.MLPActorCritic import MLPActorCritic
        import torch

        env = MockQWOPEnv(seed=42)
        obs = env.reset()

        # Create actor-critic
        ac = MLPActorCritic(env.observation_space, env.action_space)

        # Forward pass
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        pi, logp = ac.pi(obs_tensor, torch.zeros(1, 4))
        v = ac.v(obs_tensor)

        assert pi is not None
        assert logp.shape[0] == 1
        assert v.shape[0] == 1

        env.close()

    def test_ppo_buffer_collection(self):
        """Test PPO buffer collects trajectories."""
        from gym_qwop.envs.mock_qwop import MockQWOPEnv
        from rlqwop.PPO import PPOBuffer

        env = MockQWOPEnv()
        buffer = PPOBuffer(
            obs_dim=env.observation_space.shape,
            act_dim=env.action_space.shape,
            size=100,
            gamma=0.99,
            lam=0.95,
        )

        obs = env.reset()

        # Collect some transitions
        for _ in range(50):
            action = env.action_space.sample()
            obs_next, reward, done, info = env.step(action)

            buffer.store(obs, action, reward, 0.5, 0.0)

            if done:
                break
            obs = obs_next

        assert buffer.ptr > 0
        env.close()

    def test_training_loop_mock(self):
        """Test complete training loop with mock environment."""
        from gym_qwop.envs.mock_qwop import MockQWOPEnv
        from rlqwop.MLPActorCritic import MLPActorCritic
        from rlqwop.spinup_utils import EpochLogger
        import torch
        from torch.optim import Adam

        env = MockQWOPEnv(seed=42)
        logger = EpochLogger()

        # Setup
        ac = MLPActorCritic(env.observation_space, env.action_space)
        pi_optimizer = Adam(ac.pi.parameters(), lr=1e-3)
        vf_optimizer = Adam(ac.v.parameters(), lr=1e-3)

        obs = env.reset()
        total_reward = 0
        steps = 0

        # Mini training loop
        for epoch in range(2):
            for step in range(10):
                # Forward pass
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                pi, logp = ac.pi(obs_tensor, torch.zeros(1, 4))
                v = ac.v(obs_tensor)

                # Take step
                action = env.action_space.sample()
                obs_next, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1

                # Simple loss
                vf_loss = (v - torch.tensor([[reward]])).pow(2).mean()
                vf_optimizer.zero_grad()
                vf_loss.backward()
                vf_optimizer.step()

                if done:
                    obs = env.reset()
                else:
                    obs = obs_next

        assert steps > 0
        env.close()

    def test_training_with_gpu_mock(self):
        """Test training can use GPU if available."""
        from gym_qwop.envs.mock_qwop import MockQWOPEnv
        from rlqwop.MLPActorCritic import MLPActorCritic
        import torch

        env = MockQWOPEnv()
        ac = MLPActorCritic(env.observation_space, env.action_space)

        # Check device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ac = ac.to(device)

        # Test forward pass on device
        obs = env.reset()
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        pi, logp = ac.pi(obs_tensor, torch.zeros(1, 4).to(device))

        assert logp.device.type == device.split(':')[0]
        env.close()


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_mock_env_with_training_script(self):
        """Test mock env works with training script."""
        from gym_qwop.envs.mock_qwop import MockQWOPEnv
        from rlqwop.MLPActorCritic import MLPActorCritic
        import torch
        from torch.optim import Adam

        # Simulate training setup from run.py
        env = MockQWOPEnv(seed=42)

        ac = MLPActorCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            hidden_sizes=(64, 64),
        )

        pi_optimizer = Adam(ac.pi.parameters(), lr=1e-4)
        vf_optimizer = Adam(ac.v.parameters(), lr=1e-4)

        # Run few steps
        obs = env.reset()
        for step in range(20):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            pi, logp = ac.pi(obs_tensor, torch.zeros(1, 4))
            v = ac.v(obs_tensor)

            action = env.action_space.sample()
            obs_next, reward, done, info = env.step(action)

            # Simple optimization step
            loss = v.mean()
            vf_optimizer.zero_grad()
            loss.backward()
            vf_optimizer.step()

            if done:
                obs = env.reset()
            else:
                obs = obs_next

        env.close()


class TestPerformance:
    """Performance benchmarks."""

    def test_mock_env_speed(self):
        """Benchmark mock environment speed."""
        from gym_qwop.envs.mock_qwop import MockQWOPEnv

        env = MockQWOPEnv()
        env.reset()

        start = time.time()
        num_steps = 1000

        for _ in range(num_steps):
            action = env.action_space.sample()
            env.step(action)

        elapsed = time.time() - start
        steps_per_sec = num_steps / elapsed

        print(f"\nMock env: {steps_per_sec:.0f} steps/sec")

        # Should be very fast (>100 steps/sec)
        assert steps_per_sec > 100
        env.close()

    def test_actor_critic_forward_speed(self):
        """Benchmark actor-critic forward pass."""
        from gym_qwop.envs.mock_qwop import MockQWOPEnv
        from rlqwop.MLPActorCritic import MLPActorCritic
        import torch

        env = MockQWOPEnv()
        ac = MLPActorCritic(env.observation_space, env.action_space)

        obs = env.reset()
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

        start = time.time()
        num_passes = 1000

        for _ in range(num_passes):
            pi, logp = ac.pi(obs_tensor, torch.zeros(1, 4))
            v = ac.v(obs_tensor)

        elapsed = time.time() - start
        passes_per_sec = num_passes / elapsed

        print(f"Actor-critic: {passes_per_sec:.0f} forward passes/sec")

        env.close()


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '-s'])
