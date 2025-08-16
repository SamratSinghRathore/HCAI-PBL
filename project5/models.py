from django.db import models
import json

class GameSession(models.Model):
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    current_episode = models.IntegerField(default=0)
    policy_weights = models.TextField(blank=True, null=True)  # Store as JSON
    
    def set_weights(self, weights_dict):
        self.policy_weights = json.dumps(weights_dict)
    
    def get_weights(self):
        if self.policy_weights:
            return json.loads(self.policy_weights)
        return None

class Trajectory(models.Model):
    session = models.ForeignKey(GameSession, on_delete=models.CASCADE)
    episode = models.IntegerField()
    states = models.TextField()  # JSON encoded
    actions = models.TextField()  # JSON encoded
    rewards = models.TextField()  # JSON encoded
    total_reward = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def set_trajectory_data(self, states, actions, rewards):
        self.states = json.dumps(states)
        self.actions = json.dumps(actions)
        self.rewards = json.dumps(rewards)
        self.total_reward = sum(rewards)
    
    def get_trajectory_data(self):
        return {
            'states': json.loads(self.states),
            'actions': json.loads(self.actions),
            'rewards': json.loads(self.rewards)
        }

class HumanFeedback(models.Model):
    session = models.ForeignKey(GameSession, on_delete=models.CASCADE)
    trajectory1 = models.ForeignKey(Trajectory, related_name='feedback_as_first', on_delete=models.CASCADE)
    trajectory2 = models.ForeignKey(Trajectory, related_name='feedback_as_second', on_delete=models.CASCADE)
    preferred_trajectory = models.IntegerField(choices=[(1, 'Trajectory 1'), (2, 'Trajectory 2')])
    created_at = models.DateTimeField(auto_now_add=True)