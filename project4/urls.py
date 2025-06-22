from django.urls import path
from . import views

app_name = 'project4'
urlpatterns = [
    path('', views.index, name='index'),
    path('cold-start/', views.cold_start, name='cold_start'),
    path('submit-ratings/', views.submit_ratings, name='submit_ratings'),
    path('next-questions/', views.next_questions, name='next_questions'),
]