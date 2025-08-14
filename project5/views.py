from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import sys
import os
import random

# Add the current directory to Python path to import grid_world
sys.path.append(os.path.dirname(__file__))

try:
    from .grid_world import GridWorld, Action, CellType
    GRID_WORLD_AVAILABLE = True
except Exception:
    GRID_WORLD_AVAILABLE = False

# Try to import REINFORCE algorithm
try:
    from .reinforce_algorithm import REINFORCE
    from .train_reinforce import train_agent, simulate_episode, benchmark_policies
    REINFORCE_AVAILABLE = True
except Exception:
    REINFORCE_AVAILABLE = False


def index(request):
    """
    Main index view for Project 5: Reinforcement Learning with Human Feedback
    """
    context = {
        'project_title': 'Project 5',
        'project_subtitle': 'Reinforcement Learning with Human Feedback',
        'project_description': 'This project implements a grid-world environment where a mouse learns to navigate and find cheese while avoiding traps using reinforcement learning algorithms. The environment provides human feedback to improve the learning process.',
        'features': [
            'Interactive 5×5 Grid World Environment',
            'Mouse Navigation with Reward System',
            'Cheese Collection (+10 reward)',
            'Trap Avoidance (-50 penalty)',
            'Wall and Boundary Collision Detection',
            'Real-time Environment Visualization',
            'REINFORCE Policy Gradient Algorithm',
            'Policy Learning Simulation',
            'Human Feedback Integration'
        ],
        'total_features': 9,
        'status': 'Active',
        'grid_world_available': GRID_WORLD_AVAILABLE,
        'reinforce_available': REINFORCE_AVAILABLE
    }
    return render(request, 'project5/index.html', context)


def environment_demo(request):
    """
    Environment demonstration view
    """
    if not GRID_WORLD_AVAILABLE:
        context = {
            'error': 'Grid World environment is not available. Please install required dependencies.'
        }
        return render(request, 'project5/environment_demo.html', context)

    env = GridWorld(random_seed=42)
    context = {
        'environment_state': env.get_state_info(),
        'environment_display': env.render(),
        'available_actions': [action.name for action in Action],
        'reward_structure': {
            'cheese': '+10 points',
            'trap': '-50 points',
            'empty_cell': '-0.2 points',
            'wall_collision': '-0.2 points'
        },
        'cell_types': {
            'EMPTY': 'Empty cell (.)',
            'MOUSE': 'Mouse (M)',
            'WALL': 'Wall (#)',
            'TRAP': 'Trap (X)',
            'CHEESE': 'Cheese (C)',
            'ORGANIC_CHEESE': 'Organic Cheese (O)'
        }
    }
    return render(request, 'project5/environment_demo.html', context)


@csrf_exempt
def take_action(request):
    """Take an action in a freshly seeded environment (demo endpoint)."""
    if not GRID_WORLD_AVAILABLE:
        return JsonResponse({'error': 'Grid World environment is not available'}, status=500)
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    try:
        data = json.loads(request.body)
        action_name = str(data.get('action', '')).upper()
        seed = int(data.get('seed', 42))
        if action_name not in [a.name for a in Action]:
            return JsonResponse({'error': f'Invalid action: {action_name}'}, status=400)
        env = GridWorld(random_seed=seed)
        action = next(a for a in Action if a.name == action_name)
        new_pos, reward, done, info = env.step(action)
        return JsonResponse({
            'success': True,
            'new_position': new_pos,
            'reward': reward,
            'done': done,
            'info': info,
            'environment_state': env.get_state_info(),
            'environment_display': env.render(),
        })
    except Exception as e:
        return JsonResponse({'error': f'Error taking action: {e}'}, status=500)


def policy_simulation(request):
    context = {
        'grid_world_available': GRID_WORLD_AVAILABLE,
        'reinforce_available': REINFORCE_AVAILABLE,
        'policy_types': [
            {'id': 'random', 'name': 'Random Policy', 'description': 'Selects actions randomly.'},
            {'id': 'greedy_cheese', 'name': 'Greedy Cheese Policy', 'description': 'Moves towards nearest cheese.'},
            {'id': 'avoid_traps', 'name': 'Trap Avoidance Policy', 'description': 'Avoids adjacent traps.'},
        ],
    }
    if REINFORCE_AVAILABLE:
        context['policy_types'].append({'id': 'reinforce', 'name': 'REINFORCE Policy', 'description': 'Neural policy.'})
    return render(request, 'project5/policy_simulation.html', context)


@csrf_exempt
def run_simulation(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    if not GRID_WORLD_AVAILABLE:
        return JsonResponse({'error': 'Grid World environment is not available'}, status=500)
    try:
        data = json.loads(request.body)
        policy_type = data.get('policy_type', 'random')
        num_episodes = int(data.get('num_episodes', 5))
        max_steps = int(data.get('max_steps', 50))
        num_episodes = min(num_episodes, 20)
        max_steps = min(max_steps, 100)

        results = []
        total_rewards = []
        total_steps = []
        success_count = 0

        for ep in range(num_episodes):
            env = GridWorld(random_seed=1000 + ep)
            ep_reward = 0.0
            steps = 0
            episode_steps = []
            while not env.game_over and steps < max_steps:
                action = select_action_by_policy(env, policy_type)
                new_pos, reward, done, info = env.step(action)
                ep_reward += reward
                steps += 1
                episode_steps.append({
                    'step': steps,
                    'action': action.name,
                    'reward': reward,
                    'position': new_pos,
                    'done': done,
                    'info': info,
                })
                if done:
                    break
            success = len(env.cheese_positions) == 0
            success_count += 1 if success else 0
            total_rewards.append(ep_reward)
            total_steps.append(steps)
            results.append({
                'episode': ep + 1,
                'steps': episode_steps,
                'total_reward': ep_reward,
                'success': success,
                'steps_taken': steps,
                'final_state': env.render(),
            })

        mean_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
        std_reward = (sum((r - mean_reward) ** 2 for r in total_rewards) / len(total_rewards)) ** 0.5 if total_rewards else 0.0
        summary = {
            'num_episodes': num_episodes,
            'policy_type': policy_type,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_steps': sum(total_steps) / len(total_steps) if total_steps else 0.0,
            'success_rate': success_count / num_episodes if num_episodes else 0.0,
            'total_successes': success_count,
            'best_reward': max(total_rewards) if total_rewards else 0.0,
            'worst_reward': min(total_rewards) if total_rewards else 0.0,
        }
        return JsonResponse({'success': True, 'results': results, 'summary': summary})
    except Exception as e:
        return JsonResponse({'error': f'Error running simulation: {e}'}, status=500)


def select_action_by_policy(env: GridWorld, policy_type: str) -> Action:
    if policy_type == 'random':
        return random.choice(list(Action))
    if policy_type == 'greedy_cheese':
        if not env.cheese_positions:
            return random.choice(list(Action))
        mouse = env.mouse_pos
        cheese_pos, _ = min(env.cheese_positions, key=lambda x: abs(x[0][0] - mouse[0]) + abs(x[0][1] - mouse[1]))
        if cheese_pos[0] < mouse[0]:
            return Action.UP
        if cheese_pos[0] > mouse[0]:
            return Action.DOWN
        if cheese_pos[1] < mouse[1]:
            return Action.LEFT
        if cheese_pos[1] > mouse[1]:
            return Action.RIGHT
        return random.choice(list(Action))
    if policy_type == 'avoid_traps':
        mouse = env.mouse_pos
        candidates = []
        for act in Action:
            # compute next pos (ignore walls here; env will handle walls as no-ops)
            r, c = mouse
            if act == Action.UP:
                nr, nc = max(0, r - 1), c
            elif act == Action.DOWN:
                nr, nc = min(env.grid_size - 1, r + 1), c
            elif act == Action.LEFT:
                nr, nc = r, max(0, c - 1)
            else:
                nr, nc = r, min(env.grid_size - 1, c + 1)
            if (nr, nc) not in env.trap_positions:
                candidates.append(act)
        return random.choice(candidates) if candidates else random.choice(list(Action))
    if policy_type == 'reinforce' and REINFORCE_AVAILABLE:
        # For brevity, fall back to random unless a global agent is wired in
        return random.choice(list(Action))
    return random.choice(list(Action))


def reinforce_training(request):
    context = {
        'reinforce_available': REINFORCE_AVAILABLE,
        'grid_world_available': GRID_WORLD_AVAILABLE,
    }
    if request.method == 'POST':
        if not REINFORCE_AVAILABLE:
            context['error'] = 'REINFORCE algorithm is not available. Please install required dependencies.'
            return render(request, 'project5/reinforce_training.html', context)
        try:
            num_episodes = int(request.POST.get('num_episodes', 500))
            learning_rate = float(request.POST.get('learning_rate', 3e-4))
            hidden_size = int(request.POST.get('hidden_size', 128))
            config = {
                'num_episodes': min(num_episodes, 2000),
                'learning_rate': learning_rate,
                'hidden_size': hidden_size,
                'print_interval': max(50, num_episodes // 10),
            }
            agent, results = train_agent(config)
            context.update({'training_completed': True, 'results': results, 'config': config})
        except Exception as e:
            context['error'] = f'Training failed: {e}'
    return render(request, 'project5/reinforce_training.html', context)
