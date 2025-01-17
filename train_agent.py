from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import imageio
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from env import DinoGame
import gymnasium as gym
import pygame
# Tool to track the learning progress
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super(RewardLoggerCallback, self).__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        # Record the reward of the current step
        self.rewards.append(self.locals["rewards"])
        return True

# Check-in. Are the rewards going up with each iteration? (It's okay if there's some up and down)
# Register the environment

gym.envs.registration.register(
    id='DinoGame-v0',
    entry_point=DinoGame,
)

# TODO: Think back to the Q-Learning notebook. How can we initialize the environment?
env = gym.make("DinoGame")

# Set up the DQN model
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.0001, buffer_size=100000, batch_size=32)

# TODO: Try some timesteps. How long does 10 steps take. 100? 1000? Then finally try 100000.
timesteps = 800000

# Training the model
reward_callback = RewardLoggerCallback()
model.learn(total_timesteps=timesteps, callback=reward_callback)

# Step 5: Save the model
model_path = "game"
model.save(model_path)

# Check-in. Are the rewards going up with each iteration? (It's okay if there's some up and down)