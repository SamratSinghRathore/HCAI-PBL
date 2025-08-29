# Project 5: Reinforcement Learning with Human Feedback

This project implements a Reinforcement Learning (RL) system where a mouse learns to navigate a grid to find cheese. The initial policy is trained using the REINFORCE algorithm, and then refined using human feedback through a Bradley-Terry model.

## Directory Structure

```
project5/
├── static/                 # Static files (if any, e.g., CSS, JS)
├── templates/
│   └── project5/
│       └── index.html      # Main interface for training, feedback, and retraining
├── urls.py                 # URL routing for the project
├── views.py                # Backend logic for RL, feedback, and retraining
└── README.md               # This file
```

## Tasks Overview

The project is divided into three main tasks:

1.  **Task 1: REINFORCE Algorithm Training**
    - An initial policy is trained for a mouse in a 5x5 grid environment.
    - The mouse learns to collect cheese (+10 reward) while avoiding traps (-50 reward) using the REINFORCE algorithm.
    - The policy is implemented as a PyTorch neural network.

2.  **Task 2: Human Feedback and Reward Modeling**
    - A user provides feedback by choosing between two demonstrated trajectories.
    - The collected preferences are used to train a reward model based on the Bradley-Terry model of preferences. This allows the system to learn a reward function that reflects human goals.

3.  **Task 3: Policy Retraining with Learned Reward**
    - The original policy is re-trained using the REINFORCE algorithm, but this time with the reward signal coming from the learned reward model.
    - A KL-divergence penalty is included in the loss function to ensure the updated policy does not stray too far from the original, stable policy.

## Pages and Buttons

The entire user interaction happens on a single page, `index.html`, which is divided into sections corresponding to the project tasks.

### 1. Initial Policy Training (Task 1)

- **Purpose**: Train the base RL model using the predefined reward function.
- **Buttons and Interactions**:
  - **Start Initial Training**: Initializes and trains the policy network for a set number of episodes. A progress bar and a chart show the training progress and rewards over time.
  - **Train Single Episode / Train 10 Episodes**: Allows for incremental training of the policy, one episode or a small batch at a time.
  - **Reset**: Clears the trained model and all statistics, allowing the training to start from scratch.

### 2. Human Feedback Collection (Task 2)

- **Purpose**: Collect user preferences between pairs of trajectories to learn a new reward function.
- **Buttons and Interactions**:
  - **Generate Trajectories for Feedback**: Simulates two different paths (trajectories) using the currently trained policy and displays them side-by-side.
  - **Trajectory Selection**: The user can click on one of the two displayed trajectories to select it as their preference.
  - **🎬 Show Animation**: For each trajectory, this button opens a modal window that plays an animation of the mouse's movement step-by-step, showing the grid, action taken, and rewards at each step.
  - **✅ Submit My Preference**: After selecting a preferred trajectory, the user clicks this button to send their feedback to the server. The server uses this data to update the Bradley-Terry reward model.
  - **🔄 Generate New Trajectories**: Loads a new pair of trajectories for the user to compare.

### 3. Policy Retraining with Human Feedback (Task 3)

- **Purpose**: Fine-tune the policy using the reward model learned from human feedback.
- **Buttons and Interactions**:
  - **Retrain Policy with Learned Reward**: Initiates the retraining process. It uses the REINFORCE algorithm again, but instead of the original hard-coded rewards, it uses the rewards predicted by the model trained on user feedback.
  - **Retrain with KL Penalty**: A separate button that performs the retraining while also applying a KL-divergence penalty. This ensures that the new policy improves based on feedback without deviating too drastically from the original, stable policy. This button is enabled only after at least one piece of feedback has been submitted.

