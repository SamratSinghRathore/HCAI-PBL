import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Grid parameters
GRID_SIZE = 5

# Elements on the grid
EMPTY = 0
MOUSE = 1
CHEESE = 2
TRAP = 3
WALL = 4
ORGANIC_CHEESE = 5

# Number of elements
NUM_TRAPS = 2
NUM_WALLS = 2
NUM_ORGANIC_CHEESE = 1
NUM_CHEESE = 2

ACTIONS = ['up', 'down', 'left', 'right']
ACTION_TO_DELTA = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1),
}

class MouseEnvironment:
    def __init__(self, grid_size=GRID_SIZE, num_traps=NUM_TRAPS, num_walls=NUM_WALLS, 
                 num_cheese=NUM_CHEESE, num_organic_cheese=NUM_ORGANIC_CHEESE):
        self.grid_size = grid_size
        self.num_traps = num_traps
        self.num_walls = num_walls
        self.num_cheese = num_cheese
        self.num_organic_cheese = num_organic_cheese
        self.reset()
    
    def reset(self):
        """Reset environment and return initial state"""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Randomly place mouse
        while True:
            self.mouse_pos = tuple(np.random.randint(0, self.grid_size, size=2))
            if self.grid[self.mouse_pos] == EMPTY:
                self.grid[self.mouse_pos] = MOUSE
                break

        # Place normal cheese
        self.cheese_positions = []
        for _ in range(self.num_cheese):
            while True:
                cheese_pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if self.grid[cheese_pos] == EMPTY:
                    self.grid[cheese_pos] = CHEESE
                    self.cheese_positions.append(cheese_pos)
                    break

        # Place organic cheese
        self.organic_cheese_positions = []
        for _ in range(self.num_organic_cheese):
            while True:
                pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if self.grid[pos] == EMPTY:
                    self.grid[pos] = ORGANIC_CHEESE
                    self.organic_cheese_positions.append(pos)
                    break

        # Place traps
        for _ in range(self.num_traps):
            while True:
                trap_pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if self.grid[trap_pos] == EMPTY:
                    self.grid[trap_pos] = TRAP
                    break

        # Place walls
        for _ in range(self.num_walls):
            while True:
                wall_pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if self.grid[wall_pos] == EMPTY:
                    self.grid[wall_pos] = WALL
                    break

        self.done = False
        self.steps = 0
        return self.get_state()
    
    def get_state(self):
        """Convert grid to state representation for neural network"""
        # Create one-hot encoded channels for each element type
        state = np.zeros((6, self.grid_size, self.grid_size), dtype=np.float32)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                element = self.grid[i, j]
                if element < 6:  # Valid element
                    state[element, i, j] = 1.0
        
        return state
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        if self.done:
            return self.get_state(), 0, True
            
        delta = ACTION_TO_DELTA[ACTIONS[action]]
        new_pos = (self.mouse_pos[0] + delta[0], self.mouse_pos[1] + delta[1])
        
        # Check bounds
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            reward = -0.2  # Hit boundary
            self.steps += 1
            if self.steps >= 50:  # Max steps
                self.done = True
            return self.get_state(), reward, self.done
        
        # Check if new position is a wall
        if self.grid[new_pos] == WALL:
            reward = -0.2  # Hit wall
            self.steps += 1
            if self.steps >= 50:
                self.done = True
            return self.get_state(), reward, self.done
        
        # Valid move - update position
        self.grid[self.mouse_pos] = EMPTY
        reward = self.get_reward(new_pos)
        
        # Handle different cell types
        cell_content = self.grid[new_pos]
        
        if cell_content == CHEESE:
            self.grid[new_pos] = EMPTY  # Remove cheese
            if new_pos in self.cheese_positions:
                self.cheese_positions.remove(new_pos)
            reward = 10
            print(f"Cheese collected! Reward: {reward}, Remaining cheese: {len(self.cheese_positions) + len(self.organic_cheese_positions)}")
            
        elif cell_content == ORGANIC_CHEESE:
            self.grid[new_pos] = EMPTY  # Remove organic cheese
            if new_pos in self.organic_cheese_positions:
                self.organic_cheese_positions.remove(new_pos)
            reward = 10
            print(f"Organic cheese collected! Reward: {reward}, Remaining cheese: {len(self.cheese_positions) + len(self.organic_cheese_positions)}")
            
        elif cell_content == TRAP:
            reward = -50
            self.done = True  # Game over - hit trap
            print(f"Trap hit! Game over. Reward: {reward}")
        
        else:
            reward = -0.2  # Empty cell movement penalty
        
        # Update mouse position
        self.mouse_pos = new_pos
        self.grid[new_pos] = MOUSE
        
        # Check win condition - all cheese collected
        if len(self.cheese_positions) == 0 and len(self.organic_cheese_positions) == 0:
            self.done = True
            print("All cheese collected! Episode complete.")
        
        self.steps += 1
        if self.steps >= 50:  # Max steps limit
            self.done = True
            print("Maximum steps reached. Episode complete.")
            
        return self.get_state(), reward, self.done
    
    def get_reward(self, pos):
        """Calculate reward for position"""
        element = self.grid[pos]
        if element == CHEESE or element == ORGANIC_CHEESE:
            return 10
        elif element == TRAP:
            return -50
        else:
            return -0.2
    
    def render(self):
        """Print current grid state"""
        symbols = {
            EMPTY: '.',
            MOUSE: 'M',
            CHEESE: 'C',
            TRAP: 'T',
            WALL: '#',
            ORGANIC_CHEESE: 'O'
        }
        grid_str = []
        for row in self.grid:
            grid_str.append(' '.join(symbols[cell] for cell in row))
        return '\n'.join(grid_str)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(16 * 5 * 5, 64)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(64, 4)  # 4 actions
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.flatten(x)
        x = self.relu2(self.linear1(x))
        x = self.linear2(x)
        return self.softmax(x)


class REINFORCEAgent:
    def __init__(self, learning_rate=0.001):
        self.device = torch.device('cpu')
        self.policy_net = PolicyNetwork()
        self.policy_net.to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.saved_log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        """Select action using current policy"""
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Remove torch.no_grad() for training - we need gradients!
            probs = self.policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            self.saved_log_probs.append(action_dist.log_prob(action))
            return action.item()
        except Exception as e:
            print(f"Error in select_action: {e}")
            return np.random.randint(0, 4)
    
    def select_action_no_grad(self, state):
        """Select action without gradient tracking (for inference)"""
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = self.policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return action.item()
        except Exception as e:
            print(f"Error in select_action_no_grad: {e}")
            return np.random.randint(0, 4)
    
    def update_policy(self, gamma=0.99):
        """Update policy using REINFORCE algorithm"""
        try:
            if len(self.rewards) == 0:
                return 0.0
                
            discounted_rewards = []
            cumulative_reward = 0
            
            # Calculate discounted rewards
            for reward in reversed(self.rewards):
                cumulative_reward = reward + gamma * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)
            
            # Convert to tensor and normalize
            discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
            if len(discounted_rewards) > 1:
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
            
            # Calculate policy loss
            policy_loss = []
            for log_prob, reward in zip(self.saved_log_probs, discounted_rewards):
                policy_loss.append(-log_prob * reward)
            
            if len(policy_loss) > 0:
                policy_loss = torch.stack(policy_loss).sum()
                
                # Update policy
                self.optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Add gradient clipping
                self.optimizer.step()
                
                loss_value = policy_loss.item()
            else:
                loss_value = 0.0
            
            # Clear saved values
            self.saved_log_probs = []
            self.rewards = []
            
            return loss_value
            
        except Exception as e:
            print(f"Error in update_policy: {e}")
            self.saved_log_probs = []
            self.rewards = []
            return 0.0
    
    def get_state_dict(self):
        return self.policy_net.state_dict()
    
    def load_state_dict(self, state_dict):
        self.policy_net.load_state_dict(state_dict)