# tagger_app/views.py
import threading
from django.shortcuts import render, redirect
# tagger_app/views.py

from django.core.files.storage import FileSystemStorage
import os
import pandas as pd
from .forms import UploadForm
from django.http import JsonResponse
from .utils import (
    PROGRESS_STATUS,
    row_by_row_tagger,
    load_config_file
)
import uuid
from .utils import save_config_file

def upload_file_view(request):
    """ Screen 1: Upload CSV & (Optionally) Config File """
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Get the files
            csv_file = form.cleaned_data['csv_file']
            config_file = form.cleaned_data.get('config_file')

            # Save them using FileSystemStorage
            fs = FileSystemStorage(location='media/')  # or settings.MEDIA_ROOT
            csv_filename = fs.save(csv_file.name, csv_file)

            # If there's a config file, save it with a similar approach
            if config_file:
                config_filename = fs.save(config_file.name, config_file)
            else:
                config_filename = None

            # Store file paths in session (or some global dictionary)
            request.session['csv_filepath'] = os.path.join('media', csv_filename)
            request.session['config_filepath'] = os.path.join('media', config_filename) if config_filename else None

            # Redirect to define-columns step
            return redirect('define_columns')
    else:
        form = UploadForm()

    return render(request, 'upload.html', {'form': form})

# tagger_app/views.py



def define_columns_view(request):
    """ Screen 2: Define Input/Output columns & prompts """
    csv_path = request.session.get('csv_filepath')
    config_path = request.session.get('config_filepath')

    if not csv_path or not os.path.exists(csv_path):
        # If no CSV found, redirect back to upload
        return redirect('upload_file')

    # --- Load the CSV columns ---
    df = pd.read_csv(csv_path)
    all_columns = df.columns.tolist()

    # --- Load existing config data (prompts) ---
    config_data = load_config_file(config_path)  # returns list of dicts

    if request.method == 'POST':
        # 1. Get the selected input columns from form
        input_cols = request.POST.getlist('input_columns')

        # 2. Build updated config data from the POST
        #    We'll look for "output_column" and "prompt_template" lists
        output_cols = request.POST.getlist('output_column')
        prompts = request.POST.getlist('prompt_template')

        new_config = []
        for oc, pt in zip(output_cols, prompts):
            if oc.strip() and pt.strip():
                new_config.append({
                    "OutputColumn": oc.strip(),
                    "PromptTemplate": pt.strip()
                })

        # 3. Save config file (if we have any data)
        #    If no config_path was provided (i.e. user didn't upload),
        #    we can create one ourselves.
        if not config_path:
            # By convention, let's create a config path next to the CSV
            base, ext = os.path.splitext(csv_path)
            config_path = base + "_config.csv"
            request.session['config_filepath'] = config_path

        save_config_file(config_path, new_config)

        # 4. Store the selected input columns & updated config in session
        request.session['input_columns'] = input_cols

        # 5. Redirect to tagging
        return redirect('tagging')

    else:
        # Render the page with columns + existing config
        context = {
            'columns': all_columns,
            'config_data': config_data,  # e.g. [{'OutputColumn':..., 'PromptTemplate':...}, ...]
        }
        return render(request, 'define_columns.html', context)

# tagger_app/views.py



def tagging_view(request):
    """ Screen 3: Start the tagging process in background, show real-time progress """
    csv_path = request.session.get('csv_filepath')
    config_path = request.session.get('config_filepath')
    input_columns = request.session.get('input_columns', [])

    if not csv_path:
        return redirect('upload_file')

    # Load config data (OutputColumn, PromptTemplate)
    config_data = load_config_file(config_path) if config_path else []
    
    # Generate a unique key for this "tagging session"
    session_key = str(uuid.uuid4())
    request.session['tagging_session_key'] = session_key

    # Start the background thread
    t = threading.Thread(
        target=row_by_row_tagger,
        args=(session_key, csv_path, config_path, input_columns, config_data),
        daemon=True
    )
    t.start()

    # Render the template which will poll for progress
    return render(request, 'tagging.html', {})

from django.http import JsonResponse
from .utils import PROGRESS_STATUS

from django.http import JsonResponse
import pandas as pd
import os
from .utils import PROGRESS_STATUS

def tagging_progress_view(request):
    """
    Returns JSON progress for the tagging process, including real-time logs.
    """
    session_key = request.session.get('tagging_session_key')
    if not session_key or session_key not in PROGRESS_STATUS:
        return JsonResponse({'error': 'No progress data found'}, status=400)

    progress_data = PROGRESS_STATUS[session_key]

    # Read logs (if exists) to get latest logs
    logs = []
    logs_path = progress_data.get("logs_file", "")
    if logs_path and os.path.exists(logs_path):
        try:
            df_logs = pd.read_csv(logs_path)
            logs = df_logs.tail(10).to_dict('records')  # Get the last 10 logs
        except Exception as e:
            logs = [{"error": f"Failed to read logs: {str(e)}"}]

    return JsonResponse({
        "done": progress_data["done"],
        "total": progress_data["total"],
        "status": progress_data["status"],
        "logs": logs
    })

def results_view(request):
    """ Screen 4: Download the tagged CSV & logs """
    session_key = request.session.get('tagging_session_key')
    if not session_key or session_key not in PROGRESS_STATUS:
        return render(request, 'results.html', {
            'tagged_file': None,
            'logs_file': None,
        })

    data = PROGRESS_STATUS[session_key]
    if data["status"] != "finished":
        # Not finished or error
        return render(request, 'results.html', {
            'tagged_file': None,
            'logs_file': None,
        })

    return render(request, 'results.html', {
        'tagged_file': data["tagged_file"],
        'logs_file': data["logs_file"],
    })