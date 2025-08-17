from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import uuid
from .models import GameSession, Trajectory, HumanFeedback

def get_trainer(session_id):
    try:
        from .rl_trainer import RLTrainer
        return RLTrainer(session_id)
    except ImportError as e:
        raise ImportError(f"Please install required dependencies: torch, numpy. Error: {e}")

def index(request):
    """Main page for Project 5"""
    return render(request, 'project5/index.html')

@csrf_exempt
def start_training(request):
    """Initialize a new training session"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
        
    session_id = str(uuid.uuid4())
    request.session['rl_session_id'] = session_id
    
    game_session = GameSession.objects.create(session_id=session_id)
    return JsonResponse({'session_id': session_id, 'status': 'initialized'})

@csrf_exempt
def train_episode(request):
    """Train a single episode"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
        
    session_id = request.session.get('rl_session_id')
    if not session_id:
        return JsonResponse({'error': 'No active session'}, status=400)
    
    try:
        trainer = get_trainer(session_id)
        result = trainer.train_episode()
        
        session = GameSession.objects.get(session_id=session_id)
        trajectory = Trajectory.objects.create(
            session=session,
            episode=session.current_episode,
            states=json.dumps(result['states']),
            actions=json.dumps(result['actions']),
            rewards=json.dumps(result['rewards']),
            total_reward=result['total_reward']
        )
        
        session.current_episode += 1
        session.save()
        
        return JsonResponse({
            'episode': session.current_episode - 1,
            'total_reward': result['total_reward'],
            'loss': result['loss'],
            'steps': len(result['actions'])
        })
    except Exception as e:
        return JsonResponse({'error': f'Training error: {str(e)}'}, status=500)

@csrf_exempt
def train_batch(request):
    """Train multiple episodes"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
        
    try:
        data = json.loads(request.body)
        num_episodes = data.get('num_episodes', 10)
        
        session_id = request.session.get('rl_session_id')
        if not session_id:
            return JsonResponse({'error': 'No active session'}, status=400)
        
        trainer = get_trainer(session_id)
        results = trainer.train_batch(num_episodes)
        
        session = GameSession.objects.get(session_id=session_id)
        for result in results:
            Trajectory.objects.create(
                session=session,
                episode=session.current_episode + result['episode'],
                states=json.dumps(result['states']),
                actions=json.dumps(result['actions']),
                rewards=json.dumps(result['rewards']),
                total_reward=result['total_reward']
            )
        
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
    except Exception as e:
        return JsonResponse({'error': f'Batch training error: {str(e)}'}, status=500)

@csrf_exempt
def generate_trajectories_for_feedback(request):
    """Generate two trajectories for human feedback comparison"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
        
    session_id = request.session.get('rl_session_id')
    if not session_id:
        return JsonResponse({'error': 'No active session'}, status=400)
    
    try:
        trainer = get_trainer(session_id)
        
        traj1 = trainer.generate_trajectory()
        traj2 = trainer.generate_trajectory()
        
        session = GameSession.objects.get(session_id=session_id)
        
        trajectory1 = Trajectory.objects.create(
            session=session, 
            episode=-1,
            states=json.dumps(traj1['states']),
            actions=json.dumps(traj1['actions']),
            rewards=json.dumps(traj1['rewards']),
            total_reward=traj1['total_reward']
        )
        
        trajectory2 = Trajectory.objects.create(
            session=session, 
            episode=-1,
            states=json.dumps(traj2['states']),
            actions=json.dumps(traj2['actions']),
            rewards=json.dumps(traj2['rewards']),
            total_reward=traj2['total_reward']
        )
        
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
    except Exception as e:
        return JsonResponse({'error': f'Trajectory generation error: {str(e)}'}, status=500)

@csrf_exempt
def submit_feedback(request):
    """Submit human feedback preference"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
        
    try:
        data = json.loads(request.body)
        trajectory1_id = data.get('trajectory1_id')
        trajectory2_id = data.get('trajectory2_id')
        preferred = data.get('preferred')
        
        session_id = request.session.get('rl_session_id')
        if not session_id:
            return JsonResponse({'error': 'No active session'}, status=400)
        
        if not trajectory1_id or not trajectory2_id or preferred not in [1, 2]:
            return JsonResponse({'error': 'Invalid feedback data'}, status=400)
        
        session = GameSession.objects.get(session_id=session_id)
        trajectory1 = Trajectory.objects.get(id=trajectory1_id, session=session)
        trajectory2 = Trajectory.objects.get(id=trajectory2_id, session=session)
        
        feedback = HumanFeedback.objects.create(
            session=session,
            trajectory1=trajectory1,
            trajectory2=trajectory2,
            preferred_trajectory=preferred
        )
        
        return JsonResponse({
            'status': 'feedback_saved', 
            'feedback_id': feedback.id,
            'preferred': preferred
        })
    except Exception as e:
        return JsonResponse({'error': f'Feedback submission error: {str(e)}'}, status=500)

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
    except Exception as e:
        return JsonResponse({'error': f'Stats error: {str(e)}'}, status=500)

@csrf_exempt
def reset_training(request):
    """Reset training session"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
        
    session_id = request.session.get('rl_session_id')
    if session_id:
        try:
            session = GameSession.objects.get(session_id=session_id)
            Trajectory.objects.filter(session=session).delete()
            HumanFeedback.objects.filter(session=session).delete()
            session.delete()
        except GameSession.DoesNotExist:
            pass
    
    if 'rl_session_id' in request.session:
        del request.session['rl_session_id']
    
    return JsonResponse({'status': 'reset_complete'})