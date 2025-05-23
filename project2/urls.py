from django.urls import path
from . import views
app_name = "project2"
urlpatterns = [
    path('', views.index, name='index'),
    # path('upload/', views.upload_csv, name='upload'),
    # path('plot/', views.generate_plot, name='plot'),
    path('train_model/', views.train_model, name='train_model'),
    path('load_model/', views.load_model, name='load_model'),
    path('start_active_learning/', views.start_active_learning, name='start_active_learning'),
    path('submit_label/', views.submit_label, name='submit_label'),
    path('get_next_sample/', views.get_next_sample, name='get_next_sample'),
    path('train_on_labeled_data/', views.train_on_labeled_data, name='train_on_labeled_data'),
]