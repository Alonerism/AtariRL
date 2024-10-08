# AtariRL: Deep Reinforcement Learning for Atari Games

This repository contains implementations of reinforcement learning agents using Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO) to play Atari's Space Invaders. Both agents are trained using `stable-baselines3` with custom environment wrappers and logging callbacks.

## Repository Structure

### 1. **DQN.py**
   - Implements a Deep Q-Network (DQN) agent using a convolutional neural network (CNN) policy.
   - Includes TensorBoard logging, model checkpointing, and evaluation.
   - Training is performed on the Atari game Space Invaders (`SpaceInvaders-v4`).

### 2. **PPO.py**
   - Implements a Proximal Policy Optimization (PPO) agent with custom hyperparameters.
   - Supports TensorBoard logging and model checkpointing.
   - Evaluates the trained PPO model on Space Invaders and logs key metrics.

### 3. **wrappers.py**
   - Contains custom wrappers for logging detailed information during training, such as action distribution, episode rewards, and lengths.
   - Includes a `CustomMetricsCallback` for logging loss, learning rate, and other training metrics.
   - Implements a helper function `make_custom_atari_env` to set up the Atari environment with frame stacking.

## Requirements
- Python 3.7+
- `stable-baselines3`
- `gym`
- `torch`
- `tensorboard`

## How to Use

1. **Train DQN**:
   Run the following to train a DQN agent:
   ```bash
   python DQN.py
