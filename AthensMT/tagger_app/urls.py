from django.conf import settings
from django.urls import path
from . import views

BASE_URL = settings.BASE_URL.strip("/")  # Remove trailing slashes if exists

urlpatterns = [
    path(f'{BASE_URL}/', views.upload_file_view, name='upload_file'),
    path(f'{BASE_URL}/define-columns/', views.define_columns_view, name='define_columns'),
    path(f'{BASE_URL}/tagging/', views.tagging_view, name='tagging'),
    path(f'{BASE_URL}/tagging/progress/', views.tagging_progress_view, name='tagging_progress'),
    path(f"{BASE_URL}/llm_status/", views.llm_status_view, name="llm_status"),
    path(f'{BASE_URL}/results/', views.results_view, name='results'),
]