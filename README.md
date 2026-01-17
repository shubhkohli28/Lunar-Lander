# Deep Q-Network (DQN) Agent for Lunar Lander

## Overview

This project presents an end-to-end implementation of a **Deep Q-Network (DQN)** agent trained to solve the **Lunar Lander** control task.  
The agent learns optimal landing behavior through reinforcement learning by interacting directly with the environment, without relying on labeled data.

The entire pipeline is implemented from scratch in **PyTorch**, following best practices for stability, reproducibility, and clean engineering suitable for research and production-grade portfolios.

The implementation is provided as a well-documented Jupyter Notebook for clarity and ease of reproducibility.

---

## Problem Description

The Lunar Lander task is a continuous-state, discrete-action control problem modeled as a **Markov Decision Process (MDP)**.

The objective is to autonomously control a spacecraft to:
- Achieve a safe and stable landing between designated markers
- Avoid crashes
- Minimize fuel consumption

The agent must learn a policy that maximizes cumulative reward under these constraints.

---

## Methodology

The problem is solved using **Deep Q-Learning**, where a neural network approximates the optimal action-value function:

Q(s, a) = E[ sum_t γ^t r_t | s_0 = s, a_0 = a ]

To ensure stable learning, the following techniques are employed:

- Experience Replay to decorrelate training samples
- Target Network to stabilize bootstrapped value updates
- ε-Greedy exploration strategy to balance exploration and exploitation
- Soft target updates (Polyak averaging) for smooth synchronization
- Gradient clipping to prevent exploding gradients

---

## Environment Specifications

- Environment: LunarLander-v3 (Gymnasium)
- State Space: 8 continuous variables representing position, velocity, orientation, and leg contact
- Action Space: 4 discrete actions corresponding to engine controls
- Reward Structure:
  - Positive reward for successful landing
  - Large negative reward for crashing
  - Small penalties for fuel usage

---

## Neural Network Architecture

The Q-network is a fully connected feed-forward neural network:

Input Layer:        8 units  
Hidden Layer 1:   128 units (ReLU)  
Hidden Layer 2:   128 units (ReLU)  
Output Layer:       4 units (Q-values)

No output activation function is used, as Q-values are unbounded.

---

## Training Details

- Algorithm: Deep Q-Network (DQN)
- Framework: PyTorch
- Training Platform: Google Colab

### Hyperparameters

- Discount factor (γ): 0.99
- Learning rate: 1e-3
- Replay buffer size: 100,000
- Batch size: 64
- Target update coefficient (τ): 0.005
- Number of episodes: ~800

Training updates are performed at every environment step.

---

## Evaluation Protocol

Evaluation is conducted using a pure greedy policy (ε = 0) with learning disabled.

Performance is measured using:
- Average reward over multiple evaluation episodes
- Visual inspection via recorded gameplay episodes

### Success Criterion

Average evaluation reward ≥ 200

---

## Results

The trained agent demonstrates:
- Stable descent control
- Consistent landings within the target area
- Significant performance improvement over a random policy

Recorded evaluation videos are included for qualitative validation.

---

## Project Structure

Lunar-Lander/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── dqn_lunarlander_training.ipynb
│
├── videos/
│   ├── dqn_lunarlander_success.mp4
│   └── dqn_lunarlander_early_training.mp4

---

## How to Run

### Install Dependencies
pip install -r requirements.txt

### Train the Agent
python train.py

### Evaluate and Record Video
python evaluate.py

---

## Key Contributions

This project demonstrates:
- A correct and stable implementation of Deep Q-Learning
- Strong understanding of reinforcement learning fundamentals
- Ability to design, train, evaluate, and debug end-to-end RL systems
- Clean separation between training and evaluation pipelines

---

## Future Work

Potential extensions include:
- Double DQN to reduce overestimation bias
- Dueling DQN architectures
- Prioritized experience replay
- Comparison with policy-gradient methods such as PPO

---

## Author

Shubh  
B.Tech Computer Science  
Interests: Machine Learning, Reinforcement Learning, Applied AI Systems
