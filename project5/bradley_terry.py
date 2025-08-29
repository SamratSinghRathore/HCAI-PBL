import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from .models import GameSession, Trajectory, HumanFeedback
from .mouse_env import REINFORCEAgent

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
        self.device = torch.device('cpu')
        self.reward_model.to(self.device)
        self.training_history = []  # Track training progress
        
    def get_trajectory_reward(self, trajectory):
        """Calculate total reward for a trajectory using learned reward model"""
        states = json.loads(trajectory.states)
        total_reward = 0
        
        for state in states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                reward = self.reward_model(state_tensor).item()
                total_reward += reward
                
        return total_reward
    
    def train_reward_model(self, num_epochs=100):
        """Train reward model using Bradley-Terry model on human feedback"""
        try:
            session = GameSession.objects.get(session_id=self.session_id)
            feedbacks = HumanFeedback.objects.filter(session=session)
            
            if len(feedbacks) < 5:
                raise ValueError(f"Need at least 1 feedback sample, got {len(feedbacks)}")
            
            print(f"Training reward model with {len(feedbacks)} feedback samples")
            
            training_results = {
                'epochs': [],
                'losses': [],
                'accuracy': [],
                'feedback_count': len(feedbacks),
                'convergence_info': {}
            }
            
            best_loss = float('inf')
            epochs_without_improvement = 0
            patience = 20
            
            for epoch in range(num_epochs):
                total_loss = 0
                num_comparisons = 0
                correct_predictions = 0
                
                for feedback in feedbacks:
                    try:
                        # Get trajectory data
                        states1 = json.loads(feedback.trajectory1.states)
                        states2 = json.loads(feedback.trajectory2.states)
                        
                        # Calculate rewards for both trajectories
                        reward1 = torch.tensor(0.0, requires_grad=True)
                        reward2 = torch.tensor(0.0, requires_grad=True)
                        
                        # Sum rewards over trajectory 1
                        for state in states1:
                            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                            reward1 = reward1 + self.reward_model(state_tensor).squeeze()
                        
                        # Sum rewards over trajectory 2
                        for state in states2:
                            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                            reward2 = reward2 + self.reward_model(state_tensor).squeeze()
                        
                        # Bradley-Terry loss
                        if feedback.preferred_trajectory == 1:
                            loss = -torch.log(torch.sigmoid(reward1 - reward2))
                            # Check prediction accuracy
                            if reward1.item() > reward2.item():
                                correct_predictions += 1
                        else:
                            loss = -torch.log(torch.sigmoid(reward2 - reward1))
                            # Check prediction accuracy
                            if reward2.item() > reward1.item():
                                correct_predictions += 1
                        
                        total_loss += loss
                        num_comparisons += 1
                        
                    except Exception as e:
                        print(f"Error processing feedback {feedback.id}: {e}")
                        continue
                
                # Backpropagation
                if num_comparisons > 0:
                    avg_loss = total_loss / num_comparisons
                    accuracy = correct_predictions / num_comparisons if num_comparisons > 0 else 0
                    
                    self.optimizer.zero_grad()
                    avg_loss.backward()
                    self.optimizer.step()
                    
                    # Track training progress
                    loss_value = avg_loss.item()
                    training_results['epochs'].append(epoch)
                    training_results['losses'].append(loss_value)
                    training_results['accuracy'].append(accuracy)
                    
                    # Early stopping check
                    if loss_value < best_loss:
                        best_loss = loss_value
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    
                    if epoch % 10 == 0 or epoch < 10:
                        print(f"Epoch {epoch:3d}: Loss = {loss_value:.4f}, Accuracy = {accuracy:.3f}")
                    
                    # Early stopping
                    if epochs_without_improvement >= patience and epoch > 30:
                        print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                        training_results['convergence_info']['early_stopped'] = True
                        training_results['convergence_info']['final_epoch'] = epoch
                        break
                    if int(loss_value)==0 and int(accuracy)==1:
                        break
                else:
                    print(f"No valid comparisons in epoch {epoch}")
            
            # Final training statistics
            if training_results['losses']:
                final_loss = training_results['losses'][-1]
                final_accuracy = training_results['accuracy'][-1]
                improvement = ((training_results['losses'][0] - final_loss) / training_results['losses'][0]) * 100
                
                training_results['convergence_info'].update({
                    'final_loss': final_loss,
                    'final_accuracy': final_accuracy,
                    'loss_improvement_pct': improvement,
                    'converged': epochs_without_improvement < patience
                })
                
                print(f"\nTraining Summary:")
                print(f"  Final Loss: {final_loss:.4f}")
                print(f"  Final Accuracy: {final_accuracy:.3f}")
                print(f"  Loss Improvement: {improvement:.1f}%")
                print(f"  Training completed in {len(training_results['epochs'])} epochs")
            
            self.training_history.append(training_results)
            print("Reward model training completed!")
            return self.reward_model, training_results
            
        except Exception as e:
            print(f"Error training reward model: {e}")
            raise e
    
    def retrain_policy_with_learned_reward(self, original_trainer):
        """Retrain policy using learned reward function"""
        try:
            # Train reward model first
            reward_model, training_results = self.train_reward_model()
            
            # Create enhanced trainer
            enhanced_trainer = EnhancedRLTrainer(self.session_id, reward_model, original_trainer)
            
            # Train with learned rewards for several episodes
            print("Retraining policy with learned reward...")
            retraining_results = []
            
            for episode in range(20):
                result = enhanced_trainer.train_episode_with_learned_reward()
                retraining_results.append(result)
                
                if episode % 5 == 0:
                    print(f"Enhanced training episode {episode}, reward: {result['total_reward']:.2f}")
            
            # Save the enhanced policy
            enhanced_trainer.save_session()
            
            # Calculate retraining statistics
            rewards = [r['total_reward'] for r in retraining_results]
            avg_reward = sum(rewards) / len(rewards)
            reward_std = np.std(rewards)
            
            # Compare with original policy performance (if available)
            improvement_info = {
                'avg_reward_after_retraining': avg_reward,
                'reward_std': reward_std,
                'episodes_trained': len(retraining_results),
                'reward_trend': 'improving' if rewards[-1] > rewards[0] else 'declining'
            }
            
            print(f"Retraining completed! Average reward: {avg_reward:.2f} ± {reward_std:.2f}")
            
            return enhanced_trainer, {
                'bradley_terry_training': training_results,
                'policy_retraining': improvement_info,
                'detailed_rewards': rewards
            }
            
        except Exception as e:
            print(f"Error in retrain_policy_with_learned_reward: {e}")
            raise e

class EnhancedRLTrainer:
    def __init__(self, session_id, reward_model, original_trainer):
        self.session_id = session_id
        self.reward_model = reward_model
        self.device = torch.device('cpu')
        
        # Copy the original trainer's components
        from .rl_trainer import RLTrainer
        from .mouse_env import MouseEnvironment
        
        self.env = MouseEnvironment()
        self.agent = REINFORCEAgent()
        
        # Copy the trained policy weights from original trainer
        if original_trainer and hasattr(original_trainer, 'agent'):
            try:
                original_state_dict = original_trainer.agent.get_state_dict()
                self.agent.load_state_dict(original_state_dict)
                print("Successfully loaded original policy weights")
            except Exception as e:
                print(f"Could not load original weights: {e}")
    
    def train_episode_with_learned_reward(self):
        """Train episode using learned reward model with KL penalty"""
        state = self.env.reset()
        states, actions, rewards = [], [], []
        total_reward = 0
        
        while not self.env.done:
            action = self.agent.select_action(state)
            next_state, env_reward, done = self.env.step(action)
            
            # Use learned reward instead of environment reward
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                learned_reward = self.reward_model(state_tensor).item()
            
            # Combine learned reward with safety penalty to avoid catastrophic failures
            if env_reward == -50:  # Hit trap - strong penalty
                combined_reward = learned_reward - 10
            elif env_reward == 10:  # Got cheese - boost learned reward
                combined_reward = learned_reward + 2
            else:
                combined_reward = learned_reward
            
            states.append(state.tolist())
            actions.append(action)
            rewards.append(combined_reward)
            self.agent.rewards.append(combined_reward)
            
            total_reward += combined_reward
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
    
    def save_session(self):
        """Save current agent state to database"""
        try:
            session, created = GameSession.objects.get_or_create(session_id=self.session_id)
            
            # Convert tensors to lists for JSON serialization
            state_dict = self.agent.get_state_dict()
            weights_dict = {}
            for key, tensor in state_dict.items():
                weights_dict[key] = tensor.tolist()
            
            session.set_weights(weights_dict)
            session.save()
            print(f"Session {self.session_id} saved successfully")
            return session
        except Exception as e:
            print(f"Error saving session: {e}")
            return None