from django.urls import path
from . import views

app_name = 'project5'

urlpatterns = [
    path('', views.index, name='index'),
    path('environment/', views.environment_demo, name='environment_demo'),
    path('policy-simulation/', views.policy_simulation, name='policy_simulation'),
    path('reinforce-training/', views.reinforce_training, name='reinforce_training'),
    path('api/take-action/', views.take_action, name='take_action'),
    path('api/run-simulation/', views.run_simulation, name='run_simulation'),
]
