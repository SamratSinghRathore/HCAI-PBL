<<<<<<< HEAD
from django.urls import path
from . import views

app_name = 'project3'

urlpatterns = [
    path('', views.index, name='index'),
    path('decision-tree/', views.decision_tree, name='decision_tree'),
    path('logistic-regression/', views.logistic_regression, name='logistic_regression'),
=======
from django.urls import path
from . import views

app_name = 'project3'

urlpatterns = [
    path('', views.index, name='index'),
    path('decision-tree/', views.decision_tree, name='decision_tree'),
    path('logistic-regression/', views.logistic_regression, name='logistic_regression'),
>>>>>>> 23af6a68fd1263c6fef8c68b65a2b9ea76ec0811
]