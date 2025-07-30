import gymnasium as gym
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize  # Added for normalization
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your custom env
from env.custom_sort_env import CustomSortEnv  # Adjusted path

# Setup logging
logger = configure("./training/logs/", ["stdout", "tensorboard"])

# Create vectorized environments (increased to 8 for stability)
vec_env = make_vec_env(CustomSortEnv, n_envs=8, env_kwargs=dict(render_mode="rgb_array"))

# Normalize observations and rewards (key fix for divergence)
env = VecNormalize(vec_env)

# Evaluation callback: Evaluate every 10k steps, log best model
eval_callback = EvalCallback(env, best_model_save_path="./training/best_model",
                             log_path="./training/logs/", eval_freq=10000,
                             deterministic=True, render=False)

# Initialize PPO model with adjustments for stability
model = PPO(
    "MultiInputPolicy",  # Policy for dict observations
    env,
    verbose=1,
    tensorboard_log="./training/tensorboard/",
    learning_rate=1e-4,  # Lowered for stability
    batch_size=64,
    n_steps=2048,
    clip_range=0.1,  # Tighter clipping to prevent large updates
    target_kl=0.01,  # Early stop if KL divergence too high
    device="cuda"  # Use your GPUs
)
model.set_logger(logger)

# Train for 100k timesteps (adjust up to 1M for better performance)
model.learn(total_timesteps=100000, callback=eval_callback)

# Save the final model (and normalized env stats)
model.save("training/ppo_sort_red")
env.save("training/vec_normalize.pkl")  # Save normalizer for testing/loading

# Quick test: Load and run (use normalized env for consistency)
model = PPO.load("training/ppo_sort_red")
norm_env = VecNormalize.load("training/vec_normalize.pkl", vec_env)  # Load normalizer
obs, _ = norm_env.reset()
for _ in range(1000):  # Run one episode
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, infos = norm_env.step(action)
    if dones.any() or truncated.any():
        break
norm_env.close()