from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect

def upload_file_view(request):
    return render(request, 'upload.html')

def define_columns_view(request):
    return render(request, 'define_columns.html')

def tagging_view(request):
    return render(request, 'tagging.html')

def results_view(request):
    return render(request, 'results.html')