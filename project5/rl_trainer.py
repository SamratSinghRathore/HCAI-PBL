import torch
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
            action = self.agent.select_action(state)
            next_state, reward, done = self.env.step(action)
            
            states.append(state.tolist())
            actions.append(action)
            rewards.append(reward)
            
            total_reward += reward
            state = next_state
        
        # Clear saved log probs since we're not training
        self.agent.saved_log_probs = []
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'total_reward': total_reward,
            'grid_sequence': self.get_grid_sequence(states, actions)
        }
    
    def get_grid_sequence(self, states, actions):
        """Convert state sequence to readable grid sequence"""
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
            grids.append({
                'grid': grid_str,
                'action': action_name,
                'step': i
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