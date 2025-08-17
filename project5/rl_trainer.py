import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .mouse_env import MouseEnvironment, REINFORCEAgent
from .models import GameSession, Trajectory
import json

class RLTrainer:
    def __init__(self, session_id=None):
        self.env = MouseEnvironment()
        self.agent = REINFORCEAgent()
        self.session_id = session_id
        
        if session_id:
            try:
                session = GameSession.objects.get(session_id=session_id)
                weights = session.get_weights()
                if weights:
                    # Convert weights back to tensors
                    state_dict = {}
                    for key, value in weights.items():
                        state_dict[key] = torch.FloatTensor(value)
                    self.agent.load_state_dict(state_dict)
            except GameSession.DoesNotExist:
                pass
    
    def train_episode(self):
        """Train one episode using REINFORCE"""
        state = self.env.reset()
        states, actions, rewards = [], [], []
        total_reward = 0
        
        while not self.env.done:
            action = self.agent.select_action(state)
            next_state, reward, done = self.env.step(action)
            
            states.append(state.tolist())
            actions.append(action)
            rewards.append(reward)
            self.agent.rewards.append(reward)
            
            total_reward += reward
            state = next_state
        
        # Update policy
        loss = self.agent.update_policy()
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'total_reward': total_reward,
            'loss': loss
        }
    
    def generate_trajectory(self):
        """Generate a single trajectory without training"""
        state = self.env.reset()
        states, actions, rewards = [], [], []
        total_reward = 0
        
        while not self.env.done:
            # Use the no-gradient method for trajectory generation
            action = self.agent.select_action_no_grad(state)
            next_state, reward, done = self.env.step(action)
            
            states.append(state.tolist())
            actions.append(action)
            rewards.append(reward)
            
            total_reward += reward
            state = next_state
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'total_reward': total_reward,
            'grid_sequence': self.get_grid_sequence(states, actions, rewards)  # Passing rewards here
        }
    
    def get_grid_sequence(self, states, actions, rewards):
        """Convert state sequence to readable grid sequence with actual rewards"""
        grids = []
        for i, state in enumerate(states):
            # Convert state back to grid for visualization
            state_array = np.array(state)
            grid = np.zeros((5, 5), dtype=int)
            
            for element_type in range(6):
                positions = np.where(state_array[element_type] == 1)
                for row, col in zip(positions[0], positions[1]):
                    grid[row, col] = element_type
            
            symbols = {0: '.', 1: 'M', 2: 'C', 3: 'T', 4: '#', 5: 'O'}
            grid_str = '\n'.join([' '.join([symbols[cell] for cell in row]) for row in grid])
            
            action_name = ['up', 'down', 'left', 'right'][actions[i]] if i < len(actions) else 'done'
            actual_reward = rewards[i] if i < len(rewards) else 0
            
            grids.append({
                'grid': grid_str,
                'action': action_name,
                'step': i,
                'reward': actual_reward  # Include actual reward
            })
        
        return grids
    
    def save_session(self):
        """Save current agent state to database"""
        if self.session_id:
            session, created = GameSession.objects.get_or_create(session_id=self.session_id)
            
            # Convert tensors to lists for JSON serialization
            state_dict = self.agent.get_state_dict()
            weights_dict = {}
            for key, tensor in state_dict.items():
                weights_dict[key] = tensor.tolist()
            
            session.set_weights(weights_dict)
            session.save()
            return session
        return None
    
    def train_batch(self, num_episodes=10):
        """Train multiple episodes"""
        results = []
        for episode in range(num_episodes):
            result = self.train_episode()
            result['episode'] = episode
            results.append(result)
        
        # Save session after training
        self.save_session()
        
        return results

class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(32 * 5 * 5, 64)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(64, 1)  # Single reward value per state
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        x = self.relu3(self.linear1(x))
        x = self.linear2(x)
        return x

class BradleyTerryTrainer:
    def __init__(self, session_id):
        self.session_id = session_id
        self.reward_model = RewardModel()
        self.optimizer = optim.Adam(self.reward_model.parameters(), lr=0.001)
        
    def get_trajectory_reward(self, trajectory):
        """Calculate total reward for a trajectory using learned reward model"""
        states = json.loads(trajectory.states)
        total_reward = 0
        
        for state in states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                reward = self.reward_model(state_tensor).item()
                total_reward += reward
                
        return total_reward
    
    def train_reward_model(self, num_epochs=100):
        """Train reward model using Bradley-Terry model on human feedback"""
        session = GameSession.objects.get(session_id=self.session_id)
        feedbacks = HumanFeedback.objects.filter(session=session)
        
        if len(feedbacks) < 5:
            raise ValueError(f"Need at least 5 feedback samples, got {len(feedbacks)}")
        
        print(f"Training reward model with {len(feedbacks)} feedback samples")
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for feedback in feedbacks:
                # Get trajectory data
                states1 = json.loads(feedback.trajectory1.states)
                states2 = json.loads(feedback.trajectory2.states)
                
                # Calculate rewards for both trajectories
                reward1 = 0
                reward2 = 0
                
                # Sum rewards over trajectory
                for state in states1:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    reward1 += self.reward_model(state_tensor).squeeze()
                
                for state in states2:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    reward2 += self.reward_model(state_tensor).squeeze()
                
                # Bradley-Terry loss
                if feedback.preferred_trajectory == 1:
                    # Preference for trajectory 1
                    loss = -torch.log_softmax(torch.stack([reward1, reward2]), dim=0)[0]
                else:
                    # Preference for trajectory 2
                    loss = -torch.log_softmax(torch.stack([reward1, reward2]), dim=0)[1]
                
                total_loss += loss
            
            # Backpropagation
            if len(feedbacks) > 0:
                avg_loss = total_loss / len(feedbacks)
                self.optimizer.zero_grad()
                avg_loss.backward()
                self.optimizer.step()
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}, Loss: {avg_loss.item():.4f}")
        
        print("Reward model training completed!")
        return self.reward_model
    
    def retrain_policy_with_learned_reward(self, original_trainer):
        """Retrain policy using learned reward function"""
        from .rl_trainer import RLTrainer
        
        # Train reward model first
        try:
            self.train_reward_model()
        except ValueError as e:
            print(f"Cannot train reward model: {e}")
            return original_trainer
        
        # Create new trainer with learned reward
        enhanced_trainer = EnhancedRLTrainer(self.session_id, self.reward_model)
        
        # Copy original policy weights
        enhanced_trainer.agent.load_state_dict(original_trainer.agent.get_state_dict())
        
        # Train with learned rewards for several episodes
        print("Retraining policy with learned reward...")
        for episode in range(50):
            enhanced_trainer.train_episode_with_learned_reward()
            
            if episode % 10 == 0:
                print(f"Enhanced training episode {episode}")
        
        return enhanced_trainer

class EnhancedRLTrainer:
    def __init__(self, session_id, reward_model):
        from .rl_trainer import RLTrainer
        self.base_trainer = RLTrainer(session_id)
        self.reward_model = reward_model
        self.session_id = session_id
        
    def __getattr__(self, name):
        # Delegate to base trainer
        return getattr(self.base_trainer, name)
    
    def train_episode_with_learned_reward(self):
        """Train episode using learned reward model"""
        state = self.base_trainer.env.reset()
        states, actions, rewards = [], [], []
        total_reward = 0
        
        while not self.base_trainer.env.done:
            action = self.base_trainer.agent.select_action(state)
            next_state, env_reward, done = self.base_trainer.env.step(action)
            
            # Use learned reward instead of environment reward
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                learned_reward = self.reward_model(state_tensor).item()
            
            # Combine learned reward with small environment penalty to avoid traps
            combined_reward = learned_reward + (0.1 * env_reward if env_reward == -50 else 0)
            
            states.append(state.tolist())
            actions.append(action)
            rewards.append(combined_reward)
            self.base_trainer.agent.rewards.append(combined_reward)
            
            total_reward += combined_reward
            state = next_state
        
        # Update policy
        loss = self.base_trainer.agent.update_policy()
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'total_reward': total_reward,
            'loss': loss
        }