from django.urls import path
from . import views

app_name = 'project4'
urlpatterns = [
    path('', views.index, name='index'),
    path('study/', views.study_landing, name='study_landing'),
    path('start-study/', views.start_study, name='start_study'),
    path('submit-survey/', views.submit_survey, name='submit-survey'),  # Ensure this is present
    path('survey/', views.survey, name='survey'),
    path('store-recommendations/', views.store_recommendations, name='store_recommendations'),
    path('cold-start/', views.cold_start, name='cold_start'),
    path('cold-start/<str:group>/', views.cold_start, name='cold_start_with_group'),
    path('next-questions/', views.next_questions, name='next_questions'),
    path('submit-ratings/', views.submit_ratings, name='submit_ratings'),
]