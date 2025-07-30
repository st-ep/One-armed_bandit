import gymnasium as gym
import panda_gym
import torch

env = gym.make('PandaReach-v3')
print("Environment created successfully!")
print("CUDA available:", torch.cuda.is_available())
env.close()