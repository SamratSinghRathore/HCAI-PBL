"""
URL configuration for pbl project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import include, path
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", include("home.urls")), # ADDED THE ROUTE TO DIRET TO HOMEPAGE FROM THE START
    path("home/", include("home.urls")),
    path("admin/", admin.site.urls),
    path("demos/", include("demos.urls")),
    path("project1/", include("project1.urls")), # # Add this line
    path("project2/", include("project2.urls")), # # Project 2
    path("project3/", include("project3.urls")), # # Project 3
    path("project4/", include("project4.urls")), # # Project 4
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)