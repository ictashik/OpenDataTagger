from django.conf import settings
from django.urls import path
from . import views

BASE_URL = settings.BASE_URL.strip("/")

urlpatterns = [
    path(f'{BASE_URL}/',                           views.home_view,               name='home'),
    path(f'{BASE_URL}/upload/',                    views.upload_file_view,        name='upload_file'),
    path(f'{BASE_URL}/define-columns/',            views.define_columns_view,     name='define_columns'),
    path(f'{BASE_URL}/tagging/',                   views.tagging_view,            name='tagging'),
    path(f'{BASE_URL}/tagging/progress/',          views.tagging_progress_view,   name='tagging_progress'),
    path(f'{BASE_URL}/tagging/pause/',             views.pause_tagging_view,      name='pause_tagging'),
    path(f'{BASE_URL}/tagging/resume/',            views.resume_tagging_view,     name='resume_tagging'),
    path(f'{BASE_URL}/llm_status/',                views.llm_status_view,         name='llm_status'),
    path(f'{BASE_URL}/results/',                   views.results_view,            name='results'),
    path(f'{BASE_URL}/connection/',                views.connection_editor_view,  name='connection_editor'),
    path(f'{BASE_URL}/connection/test/',           views.test_connection_view,    name='test_connection'),
    path(f'{BASE_URL}/project/<str:project_id>/open/', views.project_open_view,  name='project_open'),
]
