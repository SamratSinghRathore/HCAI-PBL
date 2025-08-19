from django.urls import path
from . import views

app_name = 'project5' 

urlpatterns = [
    path('', views.index, name='index'),
    path('start_training/', views.start_training, name='start_training'),
    path('train_episode/', views.train_episode, name='train_episode'),
    path('train_batch/', views.train_batch, name='train_batch'),
    path('generate_trajectories/', views.generate_trajectories_for_feedback, name='generate_trajectories'),
    path('submit_feedback/', views.submit_feedback, name='submit_feedback'),
    path('training_stats/', views.get_training_stats, name='training_stats'),
    path('reset_training/', views.reset_training, name='reset_training'),
    path('retrain_with_feedback/', views.retrain_with_feedback, name='retrain_with_feedback'),
    path('debug_feedback/', views.debug_feedback, name='debug_feedback'),
]