from django.urls import path
from . import views
app_name = "project2"
urlpatterns = [
    path('', views.index, name='index'),
    # path('upload/', views.upload_csv, name='upload'),
    # path('plot/', views.generate_plot, name='plot'),
    path('train_model/', views.train_model, name='train_model'),
    path('load_model/', views.load_model, name='load_model')
]