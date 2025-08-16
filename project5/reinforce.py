import torch
import numpy as np
from .env import MouseEnv, grid_to_state, ACTIONS, GRID_SIZE
from .nets import PolicyNet, RewardNet

GAMMA = 0.99
BETA = 0.1  # KL penalty coefficient
N_TRAJ = 32  # Number of trajectories per update
NUM_ITERS = 500  # Training iterations
LR = 1e-3
MAX_STEPS = 50

def generate_trajectory(policy_net_path, for_display=False):
    policy_net = PolicyNet()
    policy_net.load_state_dict(torch.load(policy_net_path))
    policy_net.eval()

    env = MouseEnv(max_steps=MAX_STEPS)
    renders = []
    data = []
    state = env.reset()
    grid = env.grid.tolist()
    if for_display:
        renders.append(env.render())
    done = False
    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = policy_net(state_t)
        action_idx = torch.multinomial(probs, num_samples=1).item()
        action = ACTIONS[action_idx]
        next_state, _, done = env.step(action)
        data.append({'grid': grid, 'action': action_idx})
        grid = env.grid.tolist()
        if for_display:
            renders.append(env.render())
        state = next_state
    return renders, data if for_display else data

def train_initial_policy():
    policy_net = PolicyNet()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    for iter in range(NUM_ITERS):
        trajectories = []
        for _ in range(N_TRAJ):
            env = MouseEnv(max_steps=MAX_STEPS)
            state = env.reset()
            traj = []
            done = False
            while not done:
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                probs = policy_net(state_t)
                a = torch.multinomial(probs, 1).item()
                next_state, r, done = env.step(ACTIONS[a])
                traj.append({'state': state, 'a': a, 'r': r})
                state = next_state
            trajectories.append(traj)
        loss_pg = 0.0
        for traj in trajectories:
            G = 0.0
            for t in reversed(range(len(traj))):
                G = traj[t]['r'] + GAMMA * G
                state_t = torch.tensor(traj[t]['state'], dtype=torch.float32).unsqueeze(0)
                probs = policy_net(state_t)
                log_prob = torch.log(probs[0, traj[t]['a']])
                loss_pg -= log_prob * G
        loss_pg /= N_TRAJ
        optimizer.zero_grad()
        loss_pg.backward()
        optimizer.step()
    torch.save(policy_net.state_dict(), 'project5/initial_policy.pth')

def train_reward_model():
    reward_net = RewardNet()
    optimizer = torch.optim.Adam(reward_net.parameters(), lr=LR)
    from .models import Preference
    preferences = Preference.objects.all()
    if len(preferences) < 10:
        raise ValueError("Not enough preferences collected.")
    for epoch in range(100):
        loss_total = 0.0
        for pref in preferences:
            traj1 = pref.traj1.data
            traj2 = pref.traj2.data
            R1 = 0.0
            R2 = 0.0
            gamma_pow = 1.0
            for step in traj1:
                state = grid_to_state(step['grid'])
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                r = reward_net(state_t)[0, step['action']]
                R1 += gamma_pow * r
                gamma_pow *= GAMMA
            gamma_pow = 1.0
            for step in traj2:
                state = grid_to_state(step['grid'])
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                r = reward_net(state_t)[0, step['action']]
                R2 += gamma_pow * r
                gamma_pow *= GAMMA
            if pref.choice == 1:
                loss = -torch.log(torch.sigmoid(R1 - R2))
            else:
                loss = -torch.log(torch.sigmoid(R2 - R1))
            loss_total += loss
        loss_total /= len(preferences)
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
    torch.save(reward_net.state_dict(), 'project5/reward.pth')

def retrain_policy():
    policy_net = PolicyNet()
    policy_net.load_state_dict(torch.load('project5/initial_policy.pth'))
    pi0 = PolicyNet()
    pi0.load_state_dict(torch.load('project5/initial_policy.pth'))
    pi0.eval()
    reward_net = RewardNet()
    reward_net.load_state_dict(torch.load('project5/reward.pth'))
    reward_net.eval()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    for iter in range(NUM_ITERS):
        trajectories = []
        for _ in range(N_TRAJ):
            env = MouseEnv(max_steps=MAX_STEPS)
            state = env.reset()
            traj = []
            done = False
            while not done:
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                probs = policy_net(state_t)
                a = torch.multinomial(probs, 1).item()
                next_state, _, done = env.step(ACTIONS[a])
                r = reward_net(state_t)[0, a].item()
                traj.append({'state': state, 'a': a, 'r': r})
                state = next_state
            trajectories.append(traj)
        loss_pg = 0.0
        kl_total = 0.0
        num_steps = 0
        for traj in trajectories:
            G = 0.0
            for t in reversed(range(len(traj))):
                G = traj[t]['r'] + GAMMA * G
                state_t = torch.tensor(traj[t]['state'], dtype=torch.float32).unsqueeze(0)
                probs = policy_net(state_t)
                log_prob = torch.log(probs[0, traj[t]['a']])
                loss_pg -= log_prob * G
            for t in range(len(traj)):
                state_t = torch.tensor(traj[t]['state'], dtype=torch.float32).unsqueeze(0)
                probs = policy_net(state_t).softmax(-1)
                probs0 = pi0(state_t).softmax(-1)
                kl_total += torch.sum(probs * torch.log(probs / probs0))
                num_steps += 1
        loss_pg /= N_TRAJ
        kl_total /= num_steps if num_steps > 0 else 1
        total_loss = loss_pg + BETA * kl_total
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    torch.save(policy_net.state_dict(), 'project5/refined_policy.pth')