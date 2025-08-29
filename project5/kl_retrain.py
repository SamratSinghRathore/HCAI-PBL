import copy
import json
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import GameSession, Trajectory
from .bradley_terry import BradleyTerryTrainer, RewardModel

try:
    from .mouse_env import MouseEnvironment, PolicyNetwork
except Exception:
    from .mouse_env import MouseEnvironment
    from .nets import PolicyNetwork


class KLRegularizedAgent:
    def __init__(self, lr: float = 1e-3, device: str = "cpu"):
        self.device = torch.device(device)
        self.policy = PolicyNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.saved_log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def select_action(self, state):
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        out = self.policy(st)
        # If output not a prob. distribution, softmax it
        if (out <= 0).any() or (out.sum(dim=-1) - 1).abs().mean() > 1e-3:
            probs = F.softmax(out, dim=-1)
        else:
            probs = out
        dist = torch.distributions.Categorical(probs=probs)
        a = dist.sample()
        self.saved_log_probs.append(dist.log_prob(a))
        return int(a.item())

    def update_with_kl(self, gamma: float, states_for_kl: List, ref_policy: nn.Module, kl_beta: float) -> float:
        if not self.rewards:
            return 0.0

        # Returns
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # PG loss
        pg_terms = [-lp * Gt for lp, Gt in zip(self.saved_log_probs, returns_t)]
        pg_loss = torch.stack(pg_terms).sum() if pg_terms else torch.tensor(0.0, device=self.device)

        # KL penalty
        eps = 1e-8
        ref_policy.eval()
        ref_probs_cache = []
        with torch.no_grad():
            for s in states_for_kl:
                st = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                r_out = ref_policy(st)
                if (r_out <= 0).any() or (r_out.sum(dim=-1) - 1).abs().mean() > 1e-3:
                    r_probs = F.softmax(r_out, dim=-1)
                else:
                    r_probs = r_out
                ref_probs_cache.append(torch.clamp(r_probs, min=eps, max=1.0))

        kls = []
        for s, ref_probs in zip(states_for_kl, ref_probs_cache):
            st = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            c_out = self.policy(st)
            if (c_out <= 0).any() or (c_out.sum(dim=-1) - 1).abs().mean() > 1e-3:
                c_probs = F.softmax(c_out, dim=-1)
            else:
                c_probs = c_out
            c_probs = torch.clamp(c_probs, min=eps, max=1.0)
            kls.append(torch.sum(c_probs * (torch.log(c_probs) - torch.log(ref_probs))))
        kl_loss = torch.stack(kls).mean() if kls else torch.tensor(0.0, device=self.device)

        total_loss = pg_loss + kl_beta * kl_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        self.saved_log_probs.clear()
        self.rewards.clear()
        return float(total_loss.item())

    def load_state_dict(self, sd):
        self.policy.load_state_dict(sd)

    def state_dict(self):
        return self.policy.state_dict()


class KLEnhancedRLTrainer:
    def __init__(self, session_id: str, reward_model: RewardModel, original_policy_state: Dict,
                 kl_beta: float = 0.1, gamma: float = 0.99):
        self.session_id = session_id
        self.reward_model = reward_model.eval()
        self.env = MouseEnvironment()
        self.gamma = gamma
        self.kl_beta = kl_beta
        self.device = torch.device("cpu")

        self.agent = KLRegularizedAgent()
        self.agent.load_state_dict(original_policy_state)

        self.reference_policy = PolicyNetwork()
        self.reference_policy.load_state_dict(copy.deepcopy(original_policy_state))
        for p in self.reference_policy.parameters():
            p.requires_grad = False
        self.reference_policy.eval()

    def train_episode_with_learned_reward(self) -> Dict:
        state = self.env.reset()
        states, actions, rewards = [], [], []
        total_reward = 0.0

        while not self.env.done:
            action = self.agent.select_action(state)
            next_state, env_reward, done = self.env.step(action)

            st = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                learned_r = float(self.reward_model(st).item())

            combined = learned_r
            if env_reward == -50:
                combined = learned_r - 10.0
            elif env_reward == 10:
                combined = learned_r + 2.0

            states.append(state)
            actions.append(action)
            rewards.append(combined)
            self.agent.rewards.append(combined)

            total_reward += combined
            state = next_state

        loss = self.agent.update_with_kl(
            gamma=self.gamma,
            states_for_kl=states,
            ref_policy=self.reference_policy,
            kl_beta=self.kl_beta
        )

        return {
            "states": [s.tolist() if hasattr(s, "tolist") else s for s in states],
            "actions": actions,
            "rewards": rewards,
            "total_reward": total_reward,
            "loss": loss
        }

    def train_episodes(self, num_episodes: int = 10) -> Dict:
        session = GameSession.objects.get(session_id=self.session_id)
        results = []
        for _ in range(num_episodes):
            ep = self.train_episode_with_learned_reward()
            Trajectory.objects.create(
                session=session,
                episode=session.current_episode,
                states=json.dumps(ep["states"]),
                actions=json.dumps(ep["actions"]),
                rewards=json.dumps(ep["rewards"]),
                total_reward=ep["total_reward"]
            )
            session.current_episode += 1
            session.save(update_fields=["current_episode"])
            results.append(ep)

        avg_reward = float(sum(e["total_reward"] for e in results) / max(1, len(results)))
        avg_loss = float(sum(e["loss"] for e in results) / max(1, len(results)))
        return {
            "episodes": len(results),
            "avg_reward": avg_reward,
            "avg_loss": avg_loss,
            "details": results
        }


def retrain_policy_with_feedback_kl(session_id: str, original_trainer,
                                    bt_epochs: int = 100, kl_beta: float = 0.1,
                                    retrain_episodes: int = 10) -> Dict:
    bt = BradleyTerryTrainer(session_id)
    reward_model, bt_stats = bt.train_reward_model(num_epochs=bt_epochs)

    if not hasattr(original_trainer, "agent"):
        raise ValueError("original_trainer missing agent")

    original_state = original_trainer.agent.get_state_dict()

    kl_trainer = KLEnhancedRLTrainer(
        session_id=session_id,
        reward_model=reward_model,
        original_policy_state=original_state,
        kl_beta=kl_beta,
        gamma=0.99
    )
    retrain_stats = kl_trainer.train_episodes(num_episodes=retrain_episodes)

    return {
        "bradley_terry_training": bt_stats,
        "policy_retraining": retrain_stats
    }