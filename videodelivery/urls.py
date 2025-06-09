from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('process_frame/', views.process_frame, name='process_frame'),
    path('measurements/', views.measurements, name='measurements'),
]
