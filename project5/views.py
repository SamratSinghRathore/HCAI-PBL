from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import uuid
from .models import GameSession, Trajectory, HumanFeedback
from .rl_trainer import RLTrainer
from .mouse_env import MouseEnvironment

def index(request):
    """Main page for Project 5"""
    return render(request, 'project5/index.html')

def start_training(request):
    """Initialize a new training session"""
    session_id = str(uuid.uuid4())
    request.session['rl_session_id'] = session_id
    
    # Create new game session
    game_session = GameSession.objects.create(session_id=session_id)
    
    return JsonResponse({'session_id': session_id, 'status': 'initialized'})

def train_episode(request):
    """Train a single episode"""
    session_id = request.session.get('rl_session_id')
    if not session_id:
        return JsonResponse({'error': 'No active session'}, status=400)
    
    trainer = RLTrainer(session_id)
    result = trainer.train_episode()
    
    # Save trajectory to database
    session = GameSession.objects.get(session_id=session_id)
    trajectory = Trajectory.objects.create(
        session=session,
        episode=session.current_episode
    )
    trajectory.set_trajectory_data(
        result['states'],
        result['actions'],
        result['rewards']
    )
    trajectory.save()
    
    session.current_episode += 1
    session.save()
    
    return JsonResponse({
        'episode': session.current_episode - 1,
        'total_reward': result['total_reward'],
        'loss': result['loss'],
        'steps': len(result['actions'])
    })

def train_batch(request):
    """Train multiple episodes"""
    if request.method == 'POST':
        data = json.loads(request.body)
        num_episodes = data.get('num_episodes', 10)
        
        session_id = request.session.get('rl_session_id')
        if not session_id:
            return JsonResponse({'error': 'No active session'}, status=400)
        
        trainer = RLTrainer(session_id)
        results = trainer.train_batch(num_episodes)
        
        # Save trajectories
        session = GameSession.objects.get(session_id=session_id)
        for result in results:
            trajectory = Trajectory.objects.create(
                session=session,
                episode=session.current_episode + result['episode']
            )
            trajectory.set_trajectory_data(
                result['states'],
                result['actions'],
                result['rewards']
            )
            trajectory.save()
        
        session.current_episode += num_episodes
        session.save()
        
        avg_reward = sum(r['total_reward'] for r in results) / len(results)
        avg_loss = sum(r['loss'] for r in results) / len(results)
        
        return JsonResponse({
            'num_episodes': num_episodes,
            'average_reward': avg_reward,
            'average_loss': avg_loss,
            'total_episodes': session.current_episode
        })
    
    return JsonResponse({'error': 'POST method required'}, status=405)

def generate_trajectories_for_feedback(request):
    """Generate two trajectories for human feedback comparison"""
    session_id = request.session.get('rl_session_id')
    if not session_id:
        return JsonResponse({'error': 'No active session'}, status=400)
    
    trainer = RLTrainer(session_id)
    
    # Generate two trajectories
    traj1 = trainer.generate_trajectory()
    traj2 = trainer.generate_trajectory()
    
    # Save trajectories
    session = GameSession.objects.get(session_id=session_id)
    
    trajectory1 = Trajectory.objects.create(session=session, episode=-1)  # -1 for feedback trajectories
    trajectory1.set_trajectory_data(traj1['states'], traj1['actions'], traj1['rewards'])
    trajectory1.save()
    
    trajectory2 = Trajectory.objects.create(session=session, episode=-1)
    trajectory2.set_trajectory_data(traj2['states'], traj2['actions'], traj2['rewards'])
    trajectory2.save()
    
    return JsonResponse({
        'trajectory1': {
            'id': trajectory1.id,
            'total_reward': traj1['total_reward'],
            'steps': len(traj1['actions']),
            'grid_sequence': traj1['grid_sequence']
        },
        'trajectory2': {
            'id': trajectory2.id,
            'total_reward': traj2['total_reward'],
            'steps': len(traj2['actions']),
            'grid_sequence': traj2['grid_sequence']
        }
    })

@csrf_exempt
def submit_feedback(request):
    """Submit human feedback preference"""
    if request.method == 'POST':
        data = json.loads(request.body)
        trajectory1_id = data.get('trajectory1_id')
        trajectory2_id = data.get('trajectory2_id')
        preferred = data.get('preferred')  # 1 or 2
        
        session_id = request.session.get('rl_session_id')
        if not session_id:
            return JsonResponse({'error': 'No active session'}, status=400)
        
        session = GameSession.objects.get(session_id=session_id)
        trajectory1 = Trajectory.objects.get(id=trajectory1_id)
        trajectory2 = Trajectory.objects.get(id=trajectory2_id)
        
        feedback = HumanFeedback.objects.create(
            session=session,
            trajectory1=trajectory1,
            trajectory2=trajectory2,
            preferred_trajectory=preferred
        )
        
        return JsonResponse({'status': 'feedback_saved', 'feedback_id': feedback.id})
    
    return JsonResponse({'error': 'POST method required'}, status=405)

def get_training_stats(request):
    """Get training statistics"""
    session_id = request.session.get('rl_session_id')
    if not session_id:
        return JsonResponse({'error': 'No active session'}, status=400)
    
    try:
        session = GameSession.objects.get(session_id=session_id)
        trajectories = Trajectory.objects.filter(session=session, episode__gte=0).order_by('episode')
        feedbacks = HumanFeedback.objects.filter(session=session)
        
        rewards = [t.total_reward for t in trajectories]
        episodes = [t.episode for t in trajectories]
        
        return JsonResponse({
            'episodes': episodes,
            'rewards': rewards,
            'total_episodes': session.current_episode,
            'total_feedbacks': feedbacks.count(),
            'session_id': session_id
        })
    except GameSession.DoesNotExist:
        return JsonResponse({'error': 'Session not found'}, status=404)

def reset_training(request):
    """Reset training session"""
    session_id = request.session.get('rl_session_id')
    if session_id:
        try:
            session = GameSession.objects.get(session_id=session_id)
            # Delete all related data
            Trajectory.objects.filter(session=session).delete()
            HumanFeedback.objects.filter(session=session).delete()
            session.delete()
        except GameSession.DoesNotExist:
            pass
    
    # Remove from Django session
    if 'rl_session_id' in request.session:
        del request.session['rl_session_id']
    
    return JsonResponse({'status': 'reset_complete'})