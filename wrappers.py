import gym
import numpy as np
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Custom wrapper for detailed logging
class LoggingWrapper(gym.Wrapper):
    def __init__(self, env):
        super(LoggingWrapper, self).__init__(env)
        self.action_descriptions = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT", 4: "RIGHTFIRE", 5: "LEFTFIRE"}
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_actions = []

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        action_str = self.action_descriptions.get(action, "UNKNOWN")
        self.episode_rewards.append(reward)
        self.episode_actions.append(action_str)
        self.episode_lengths.append(1)  # Counting each step as 1
        if done:
            survived = info.get('lives', 0) > 0
            average_reward = np.mean(self.episode_rewards)
            total_length = sum(self.episode_lengths)
            action_counts = {action: self.episode_actions.count(action) for action in set(self.episode_actions)}
            print("Episode Summary:")
            print("Total Reward: ", sum(self.episode_rewards))
            print("Average Reward: ", average_reward)
            print("Total Length: ", total_length)
            print("Action Distribution: ", action_counts)
            print("Survived the episode:", survived)
            # Reset logs for the next episode
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_actions = []
        return observation, reward, done, truncated, info

    def reset(self, **kwargs):
        # Ensure logs are cleared on reset
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_actions = []
        return self.env.reset(**kwargs)

class CustomMetricsCallback(BaseCallback):
    def __init__(self, check_freq, writer):
        super(CustomMetricsCallback, self).__init__()
        self.check_freq = check_freq
        self.writer = writer

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Loss
            loss = self.model.logger.get_value('train/loss')
            if loss is not None:
                self.writer.add_scalar('Loss', loss, self.n_calls)

            # Learning Rate
            learning_rate = self.model.lr_schedule(self.n_calls / self.total_timesteps)
            self.writer.add_scalar('Learning Rate', learning_rate, self.n_calls)

            # Epsilon (Exploration Rate)
            epsilon = self.model.exploration_rate
            self.writer.add_scalar('Epsilon', epsilon, self.n_calls)

            # Average Reward
            if len(self.model.ep_info_buffer) > 0:
                average_reward = np.mean([info['r'] for info in self.model.ep_info_buffer])
                self.writer.add_scalar('Average Reward', average_reward, self.n_calls)

            # Average Episode Length
            if len(self.model.ep_info_buffer) > 0:
                average_length = np.mean([info['l'] for info in self.model.ep_info_buffer])
                self.writer.add_scalar('Average Episode Length', average_length, self.n_calls)

            # Entropy - if the entropy is logged by the model or accessible from the policy
            entropy = self.model.logger.get_value('entropy')
            if entropy is not None:
                self.writer.add_scalar('Entropy', entropy, self.n_calls)

        return True

def make_custom_atari_env(env_id, seed, n_stack):
    def _init():
        env = gym.make(env_id)
        env = LoggingWrapper(env)
        env.seed(seed)
        return env
    env = DummyVecEnv([_init])
    env = VecFrameStack(env, n_stack=n_stack)
    return env

def learning_rate_schedule(progress_remaining):
    initial_lr = 0.003  # Starting learning rate
    final_lr = 0.00015   # Final learning rate vc
    return initial_lr + (final_lr - initial_lr) * (1 - progress_remaining)