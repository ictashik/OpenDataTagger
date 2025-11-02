# tagger_app/views.py
import threading
from django.shortcuts import render, redirect
from django.conf import settings
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
from django.core.cache import cache
from .utils import LLM_MODEL_NAME, LLM_CACHE_KEYS
import time

def llm_status_view(request):
    """Returns real-time LLM usage statistics."""
    request_count = cache.get(LLM_CACHE_KEYS["requests"], 0)
    total_time = cache.get(LLM_CACHE_KEYS["total_time"], 0.0)
    avg_speed = (total_time / request_count) if request_count > 0 else 0.0

    return JsonResponse({
        "status": "Connected" if request_count > 0 else "Idle",
        "model": LLM_MODEL_NAME,
        "requests": request_count,
        "total_time": f"{total_time:.2f} sec",
        "avg_speed": f"{avg_speed:.2f} sec/request"
    })
def upload_file_view(request):
    """Handles file upload and ensures consistent naming."""
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = form.cleaned_data['csv_file']
            config_file = form.cleaned_data.get('config_file')

            # 🔹 Use FileSystemStorage but overwrite duplicates
            fs = FileSystemStorage(location='media/')
            csv_filename = csv_file.name  # Keep original filename
            csv_path = os.path.join('media', csv_filename)

            # Delete existing file (to prevent auto-renaming)
            if os.path.exists(csv_path):
                os.remove(csv_path)

            # Save the file
            fs.save(csv_filename, csv_file)

            # Handle config file
            if config_file:
                config_filename = config_file.name
                config_path = os.path.join('media', config_filename)
                
                # Delete existing config if needed
                if os.path.exists(config_path):
                    os.remove(config_path)

                fs.save(config_filename, config_file)
            else:
                config_path = None

            # Store file paths in session
            request.session['csv_filepath'] = csv_path
            request.session['config_filepath'] = config_path

            # Redirect to column definition
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

def tagging_progress_view(request):
    """
    Returns JSON progress for the tagging process, including real-time logs.
    Enhanced with persistent file checking.
    """
    session_key = request.session.get('tagging_session_key')
    if not session_key or session_key not in PROGRESS_STATUS:
        return JsonResponse({'error': 'No progress data found'}, status=400)

    progress_data = PROGRESS_STATUS[session_key]

    # 🔹 DEBUG: Print progress updates to check if "finished" is ever set
    print(f"DEBUG: Progress for {session_key} -> {progress_data}")

    logs = []

    # ✅ Try to get logs file from multiple sources
    logs_path = None
    
    # First, try from cache
    cached_logs_path = cache.get(f"logs_file_{session_key}")
    if cached_logs_path and os.path.exists(cached_logs_path):
        logs_path = cached_logs_path
    else:
        # Fall back to session-based path
        csv_filepath = request.session.get("csv_filepath", "")
        if csv_filepath:
            logs_path = csv_filepath.replace(".csv", "_logs.csv")

    if logs_path and os.path.exists(logs_path):
        try:
            time.sleep(0.1)  # Ensures logs are fully written before reading
            df_logs = pd.read_csv(logs_path)

            # Ensure logs have correct structure
            required_columns = {"row_index", "column", "prompt", "best_answer", "explanation"}
            if required_columns.issubset(df_logs.columns):
                logs = df_logs.tail(10).to_dict('records')
            else:
                logs = [{"error": "Log format incorrect. Missing columns."}]
        except Exception as e:
            logs = [{"error": f"Failed to read logs: {str(e)}"}]
    else:
        logs = [{"info": "Logs file not yet available or still being created."}]

    # ✅ Add file persistence status
    tagged_file_path = cache.get(f"tagged_file_{session_key}")
    files_saved = tagged_file_path and os.path.exists(tagged_file_path)

    return JsonResponse({
        "done": progress_data["done"],
        "total": progress_data["total"],
        "status": progress_data["status"],
        "logs": logs,
        "files_saved": files_saved,
        "last_save": f"Row {progress_data['done']}" if files_saved else "Not saved yet"
    })
def results_view(request):
    """Screen 4: Show tagged CSV results + logs + download options"""

    session_key = request.session.get('tagging_session_key')
    
    # ✅ Retrieve paths from Django cache
    tagged_file = cache.get(f"tagged_file_{session_key}")
    logs_file = cache.get(f"logs_file_{session_key}")

    # ✅ Debugging: Print session contents
    print(f"DEBUG: Session Data in results_view -> {dict(request.session)}")
    print(f"DEBUG: Cached files -> {tagged_file}, {logs_file}")

    # ✅ RECOVERY MECHANISM: If cache is empty, try to find auto-saved files
    if not tagged_file or not os.path.exists(tagged_file):
        csv_path = request.session.get('csv_filepath')
        if csv_path and os.path.exists(csv_path):
            base, ext = os.path.splitext(csv_path)
            auto_tagged_file = base + "_tagged.csv"
            auto_logs_file = base + "_logs.csv"
            
            if os.path.exists(auto_tagged_file):
                tagged_file = auto_tagged_file
                logs_file = auto_logs_file if os.path.exists(auto_logs_file) else None
                
                # Update cache with recovered files (24 hours for long jobs)
                cache.set(f"tagged_file_{session_key}", tagged_file, timeout=86400)
                if logs_file:
                    cache.set(f"logs_file_{session_key}", logs_file, timeout=86400)
                    
                print(f"DEBUG: Recovered auto-saved files -> {tagged_file}, {logs_file}")
            else:
                print("🚨 Error: No tagged file found (neither cached nor auto-saved)! Redirecting...")
                return redirect('tagging')
        else:
            print("🚨 Error: No CSV path in session! Redirecting to upload...")
            return redirect('upload_file')

    # Load first 10 rows for preview
    df = pd.read_csv(tagged_file)
    table_columns = df.columns.tolist()
    table_data = df.head(10).values.tolist()

    # Generate proper URLs for download
    tagged_file_url = settings.MEDIA_URL + os.path.basename(tagged_file) if os.path.exists(tagged_file) else None
    logs_file_url = settings.MEDIA_URL + os.path.basename(logs_file) if logs_file and os.path.exists(logs_file) else None

    return render(request, 'results.html', {
        "tagged_file_url": tagged_file_url,
        "logs_file_url": logs_file_url,
        "table_columns": table_columns,
        "table_data": table_data
    })