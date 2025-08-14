# Project 5: Reinforcement Learning with Human Feedback

## Overview

This project implements a comprehensive reinforcement learning system with a focus on the REINFORCE policy gradient algorithm. The environment is a 5×5 grid world where a mouse agent learns to navigate, collect cheese, and avoid traps.

## Environment

### Grid World Specifications
- **Size**: 5×5 grid
- **Agent**: Mouse (M)
- **Objectives**: Cheese (C) and Organic Cheese (O)
- **Obstacles**: Traps (X) and Walls (#)
- **Empty cells**: (.)

### Reward Structure
- **+10 points**: Collecting cheese (normal or organic)
- **-50 points**: Entering a trap
- **-0.2 points**: Moving to empty cell or hitting wall/boundary

### State Representation
- 4-channel representation:
  1. Mouse position
  2. Cheese locations (with different values for normal vs organic)
  3. Trap locations
  4. Wall locations

## REINFORCE Algorithm

### Implementation Details
- **Policy Network**: Multi-layer perceptron (MLP)
  - Input: Flattened 4-channel grid representation (100 features)
  - Hidden layers: Configurable size (default 128)
  - Output: 4 action probabilities (UP, DOWN, LEFT, RIGHT)
  - Activation: ReLU for hidden layers, Softmax for output

### Training Process
1. **Episode Rollout**: Agent follows current policy to generate trajectory
2. **Return Calculation**: Compute discounted returns with γ = 0.99
3. **Policy Update**: Gradient ascent on expected return
4. **Exploration**: Epsilon-greedy with decay (ε₀ = 0.3 → ε_final = 0.01)

### Hyperparameters
- **Learning Rate**: 3e-4 (configurable)
- **Discount Factor**: 0.99
- **Hidden Size**: 128 (configurable)
- **Optimizer**: Adam
- **Episodes**: 500-2000 (configurable)

## Features

### 1. Interactive Environment Demo
- Real-time grid visualization
- Manual action controls (arrow keys)
- Action history logging
- Environment state display

### 2. Policy Simulation & Comparison
- **Random Policy**: Baseline random action selection
- **Greedy Cheese Policy**: Always moves toward nearest cheese
- **Trap Avoidance Policy**: Random actions avoiding visible traps
- **REINFORCE Policy**: Trained neural network policy

### 3. REINFORCE Training Interface
- Web-based training configuration
- Real-time training progress (when implemented)
- Performance metrics and visualizations
- Model saving and loading capabilities

## File Structure

```
project5/
├── grid_world.py              # Environment implementation
├── reinforce_algorithm.py     # REINFORCE algorithm
├── train_reinforce.py         # Training scripts and utilities
├── views.py                   # Django views
├── urls.py                    # URL routing
├── models.py                  # Django models
├── forms.py                   # Django forms
├── templates/project5/        # HTML templates
│   ├── index.html            # Main project page
│   ├── environment_demo.html # Interactive environment
│   ├── policy_simulation.html# Policy comparison
│   └── reinforce_training.html# Training interface
└── static/project5/          # Static files (CSS, JS, images)
```

## Dependencies

### Required Packages
```
django                 # Web framework
numpy                 # Numerical computations
torch                 # Deep learning framework
matplotlib            # Plotting and visualization
```

### Optional Packages (for extended functionality)
```
torchvision           # Additional PyTorch utilities
tensorboard           # Training visualization
seaborn               # Enhanced plotting
```

## Usage

### Running the Environment
```python
from project5.grid_world import GridWorld, Action

# Create environment
env = GridWorld(random_seed=42)

# Take actions
action = Action.UP
new_pos, reward, done, info = env.step(action)

# Render current state
print(env.render())
```

### Training REINFORCE Agent
```python
from project5.reinforce_algorithm import REINFORCE

# Initialize agent
agent = REINFORCE(
    grid_size=5,
    learning_rate=3e-4,
    gamma=0.99,
    hidden_size=128
)

# Train the agent
episode_rewards, episode_lengths, losses = agent.train(
    num_episodes=1000,
    max_steps_per_episode=50
)

# Evaluate performance
eval_stats = agent.evaluate(num_episodes=20)
```

### Web Interface
1. Navigate to `/project5/` for the main project page
2. Visit `/project5/environment/` for interactive environment demo
3. Go to `/project5/policy-simulation/` for policy comparisons
4. Access `/project5/reinforce-training/` for training interface

## Performance Metrics

### Evaluation Criteria
- **Success Rate**: Percentage of episodes where all cheese is collected
- **Average Reward**: Mean cumulative reward per episode
- **Episode Length**: Average steps to completion
- **Learning Curve**: Reward progression over training episodes

### Baseline Performance
- **Random Policy**: ~-15 to -25 average reward, <5% success rate
- **Greedy Cheese Policy**: ~5 to 15 average reward, 15-30% success rate
- **Trained REINFORCE**: Target >20 average reward, >60% success rate

## Technical Notes

### State Space
- **Continuous**: Grid positions (discrete but high-dimensional)
- **Observable**: Full environment state available
- **Deterministic**: Actions have consistent outcomes

### Action Space
- **Discrete**: 4 actions (UP, DOWN, LEFT, RIGHT)
- **Deterministic**: Actions always attempt the same movement
- **Bounded**: Invalid moves (hitting walls) result in staying in place

### Episode Termination
- **Success**: All cheese collected
- **Failure**: Enter trap
- **Timeout**: Maximum steps reached (default: 100)

## Extensions and Future Work

### Potential Improvements
1. **Curriculum Learning**: Start with simpler environments
2. **Actor-Critic Methods**: Add value function estimation
3. **Experience Replay**: Store and reuse past experiences
4. **Multi-Agent**: Multiple mice or competitive scenarios
5. **Human Feedback Integration**: Direct policy shaping through human input

### Advanced Features
- **Attention Mechanisms**: Focus on relevant grid regions
- **Recurrent Networks**: Memory for partially observable environments
- **Meta-Learning**: Quick adaptation to new environment configurations
- **Hierarchical RL**: High-level navigation + low-level actions

## Troubleshooting

### Common Issues
1. **CUDA/GPU Issues**: Set device manually in REINFORCE initialization
2. **Memory Errors**: Reduce batch size or network hidden size
3. **Slow Training**: Check PyTorch installation and consider GPU usage
4. **Import Errors**: Ensure all dependencies are installed

### Performance Tips
- Use GPU when available for faster training
- Adjust learning rate if training is unstable
- Monitor gradient norms for optimization debugging
- Use tensorboard for detailed training analysis

## References

- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms.
