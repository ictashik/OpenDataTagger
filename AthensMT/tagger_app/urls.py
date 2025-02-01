# tagger_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_file_view, name='upload_file'),      # Screen 1
    path('define-columns/', views.define_columns_view, name='define_columns'),  # Screen 2
    path('tagging/', views.tagging_view, name='tagging'),      # Screen 3
    path('tagging/progress/', views.tagging_progress_view, name='tagging_progress'),  # ✅ FIXED: Add progress route

    path('results/', views.results_view, name='results'),      # Screen 4
]