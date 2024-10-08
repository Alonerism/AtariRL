import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.evaluation import evaluate_policy
from wrappers import make_custom_atari_env, CustomMetricsCallback

# Set up the environment
env_id = 'SpaceInvaders-v4'
seed = 33
n_stack = 4
env = make_custom_atari_env(env_id, seed, n_stack)
eval_env = make_custom_atari_env(env_id, seed, n_stack)

writer = SummaryWriter(log_dir='./tensorboard_logs')

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs",
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.97,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            seed=seed,
            device='auto',
            _init_setup_model=True)

# Callbacks for training
checkpoint_dir = './training/saved_models'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_dir, name_prefix='ppo_checkpoint')
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=checkpoint_dir,
    log_path=checkpoint_dir,
    eval_freq=5000,
    deterministic=True,
    render=False
)
custom_metrics_callback = CustomMetricsCallback(check_freq=1000, writer=writer)

# Train the model with callbacks
total_timesteps = 120000
model.learn(total_timesteps=int(total_timesteps), callback=[checkpoint_callback, eval_callback, custom_metrics_callback])

# Save the trained model
model.save("PPO_3_final")

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Close the environments and writer
env.close()
eval_env.close()
writer.close()


#Place this in terminal to see tensorboard
#tensorboard --logdir=./tensorboard_logs