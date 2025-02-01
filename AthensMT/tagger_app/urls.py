from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_file_view, name='upload_file'),
    path('define-columns/', views.define_columns_view, name='define_columns'),
    path('tagging/', views.tagging_view, name='tagging'),
    path('results/', views.results_view, name='results'),
]