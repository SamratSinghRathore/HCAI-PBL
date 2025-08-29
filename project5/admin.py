from django.contrib import admin
from .models import GameSession, Trajectory, HumanFeedback

@admin.register(GameSession)
class GameSessionAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'created_at', 'current_episode']
    list_filter = ['created_at']
    search_fields = ['session_id']

@admin.register(Trajectory)
class TrajectoryAdmin(admin.ModelAdmin):
    list_display = ['session', 'episode', 'total_reward', 'created_at']
    list_filter = ['created_at', 'episode']
    search_fields = ['session__session_id']

@admin.register(HumanFeedback)
class HumanFeedbackAdmin(admin.ModelAdmin):
    list_display = ['session', 'preferred_trajectory', 'created_at']
    list_filter = ['preferred_trajectory', 'created_at']