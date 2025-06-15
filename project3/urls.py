from django.urls import path
from . import views

app_name = 'project3'

urlpatterns = [
    path('', views.index, name='index'),
    path('decision-tree/', views.decision_tree, name='decision_tree'),
    path('logistic-regression/', views.logistic_regression, name='logistic_regression'),
    path('counterfactual/', views.counterfactual, name='counterfactual'),
]