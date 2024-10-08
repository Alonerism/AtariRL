import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from wrappers import make_custom_atari_env, learning_rate_schedule
from torch.utils.tensorboard import SummaryWriter

# DQN training setup
env_id = 'SpaceInvaders-v4'
seed = 33
n_stack = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

env = make_custom_atari_env(env_id, seed, n_stack)
eval_env = make_custom_atari_env(env_id, seed, n_stack)

# Define a unique identifier for this training session to differentiate it in logs and saved models
model_version = "v2"  # Update this as needed for new versions
name_prefix = f"dqn_model_{model_version}"

# Model and TensorBoard logging setup
checkpoint_dir = './training/saved_models'  # Same directory as before
tensorboard_log_dir = './tensorboard_logs'  # All logs under the same main directory

# Ensure directories exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

# Initialize the model
model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    device=device,
    tensorboard_log=tensorboard_log_dir, 
    learning_rate=learning_rate_schedule,
    buffer_size=10000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=10000,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.07
)

# Function to train the model
def train_model(model, env, total_timesteps):
    save_path = checkpoint_dir
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_path, name_prefix=name_prefix)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback], tb_log_name=name_prefix)

train_model(model, env, total_timesteps=120000)
model.save(os.path.join(checkpoint_dir, f"{name_prefix}_Final"))

env.close()
eval_env.close()

#Place this in terminal to see tensorboard
#tensorboard --logdir=./tensorboard_logs