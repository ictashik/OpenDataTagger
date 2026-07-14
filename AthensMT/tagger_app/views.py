# tagger_app/views.py
import threading
import json
import uuid
import os
import time
from itertools import zip_longest

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.core.cache import cache
import pandas as pd

from .forms import UploadForm
from .utils import (
    PROGRESS_STATUS,
    PAUSE_FLAGS,
    row_by_row_tagger,
    load_config_file,
    save_config_file,
    LLM_CACHE_KEYS,
    IMAGE_CACHE_KEYS,
    get_active_connection,
    load_connections,
    save_connection,
    load_projects,
    save_project,
    update_project,
    delete_project,
    get_host_stats,
    read_csv_safe,
    convert_upload_to_csv,
    load_image_connections,
    save_image_connection,
    get_active_image_connection,
    get_image_capability,
    get_image_models,
    get_downloaded_image_models,
    start_image_download,
    image_download_status,
)


def _project_mode(request):
    """Authoritative mode ('text' | 'image') for the session's current project."""
    pid = request.session.get('project_id')
    if pid:
        proj = next((p for p in load_projects() if str(p['project_id']) == str(pid)), None)
        if proj:
            return proj.get('mode', 'text') or 'text'
    return request.session.get('project_mode', 'text') or 'text'


# ─── LLM status (sidebar poll) ───────────────────────────────────────────────

def llm_status_view(request):
    request_count = cache.get(LLM_CACHE_KEYS["requests"], 0)
    total_time    = cache.get(LLM_CACHE_KEYS["total_time"], 0.0)
    avg_speed     = (total_time / request_count) if request_count > 0 else 0.0
    return JsonResponse({
        "status":     "Connected" if request_count > 0 else "Idle",
        "model":      get_active_connection()['model'],
        "requests":   request_count,
        "total_time": f"{total_time:.2f} sec",
        "avg_speed":  f"{avg_speed:.2f} sec/request",
    })


# ─── Home / Projects dashboard ───────────────────────────────────────────────

def home_view(request):
    projects = load_projects()
    for p in projects:
        sk = p.get('session_key', '')
        ps = PROGRESS_STATUS.get(sk) if sk else None
        if ps:
            p['live_done']   = ps['done']
            p['live_total']  = ps['total']
            p['live_status'] = ps['status']
            p['is_live']     = True
        else:
            p['live_done']   = p.get('done_rows', 0)
            p['live_total']  = p.get('total_rows', 0)
            p['live_status'] = p.get('status', 'idle')
            p['is_live']     = False

    return render(request, 'home.html', {
        'projects':         projects,
        'host_stats':       get_host_stats(),
        'active_connection': get_active_connection(),
    })


def project_open_view(request, project_id):
    """Set session context for a project and redirect to the right page."""
    projects = load_projects()
    project  = next((p for p in projects if str(p['project_id']) == project_id), None)
    if not project:
        return redirect('home')

    request.session['csv_filepath']    = project['csv_path']
    request.session['config_filepath'] = project['config_path'] or None
    request.session['project_id']      = project['project_id']
    request.session['project_mode']    = project.get('mode', 'text')

    action = request.GET.get('action', 'auto')

    if action == 'edit':
        return redirect('define_columns')

    if action == 'results':
        if project.get('session_key'):
            request.session['tagging_session_key'] = project['session_key']
        return redirect('results')

    if action == 'monitor':
        sk = project.get('session_key', '')
        if sk and sk in PROGRESS_STATUS:
            request.session['tagging_session_key'] = sk
            return redirect('tagging')
        # Session not live — fall through to auto
        action = 'auto'

    # auto: pick best action based on status
    status = project.get('status', 'idle')
    if status in ('running', 'paused') and project.get('session_key'):
        request.session['tagging_session_key'] = project['session_key']
        return redirect('tagging')
    if status == 'finished' and project.get('session_key'):
        request.session['tagging_session_key'] = project['session_key']
        return redirect('results')
    return redirect('define_columns')


def delete_project_view(request, project_id):
    if request.method == 'POST':
        delete_project(project_id)
    return redirect('home')


# ─── Upload ──────────────────────────────────────────────────────────────────

def upload_file_view(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file    = form.cleaned_data['csv_file']
            config_file = form.cleaned_data.get('config_file')
            mode        = 'image' if request.POST.get('mode') == 'image' else 'text'

            fs       = FileSystemStorage(location='media/')
            csv_path = os.path.join('media', csv_file.name)
            if os.path.exists(csv_path):
                os.remove(csv_path)
            fs.save(csv_file.name, csv_file)
            csv_path = convert_upload_to_csv(csv_path)

            if config_file:
                config_path = os.path.join('media', config_file.name)
                if os.path.exists(config_path):
                    os.remove(config_path)
                fs.save(config_file.name, config_file)
                config_path = convert_upload_to_csv(config_path)
            else:
                config_path = None

            request.session['csv_filepath']    = csv_path
            request.session['config_filepath'] = config_path
            request.session['project_mode']    = mode

            # Register project
            project_id = str(uuid.uuid4())
            request.session['project_id'] = project_id
            save_project(
                project_id=project_id,
                name=os.path.splitext(csv_file.name)[0],
                csv_path=csv_path,
                config_path=config_path or '',
                mode=mode,
            )

            return redirect('define_columns')
    else:
        form = UploadForm()

    return render(request, 'upload.html', {'form': form})


# ─── Define columns ──────────────────────────────────────────────────────────

def define_columns_view(request):
    csv_path    = request.session.get('csv_filepath')
    config_path = request.session.get('config_filepath')

    if not csv_path or not os.path.exists(csv_path):
        return redirect('upload_file')

    df          = read_csv_safe(csv_path)
    all_columns = df.columns.tolist()
    config_data = load_config_file(config_path)
    mode        = _project_mode(request)

    if request.method == 'POST':
        input_cols       = request.POST.getlist('input_columns')
        output_cols      = request.POST.getlist('output_column')
        prompts          = request.POST.getlist('prompt_template')
        condition_fields = request.POST.getlist('condition_field')
        condition_ops    = request.POST.getlist('condition_op')
        condition_values = request.POST.getlist('condition_value')
        default_values   = request.POST.getlist('default_value')
        send_contexts    = request.POST.getlist('send_context')
        tag_input_cols   = request.POST.getlist('tag_input_cols')
        image_params     = request.POST.getlist('image_params')

        new_config = []
        # zip_longest: image_params is absent in text mode (fills to ''); other
        # arrays are always one-per-card thanks to the mirrored-hidden inputs.
        for oc, pt, cf, cop, cv, dv, sc, tic, ip in zip_longest(
            output_cols, prompts,
            condition_fields, condition_ops, condition_values, default_values,
            send_contexts, tag_input_cols, image_params,
            fillvalue='',
        ):
            if (oc or '').strip() and (pt or '').strip():
                new_config.append({
                    "OutputColumn":   oc.strip(),
                    "PromptTemplate": pt.strip(),
                    "ConditionField": (cf or '').strip(),
                    "ConditionOp":    (cop or '').strip(),
                    "ConditionValue": (cv or '').strip(),
                    "DefaultValue":   (dv or '').strip(),
                    "SendContext":    (sc or '').strip(),
                    "InputColumns":   (tic or '').strip(),
                    "ImageParams":    (ip or '').strip(),
                })

        if not config_path:
            base, _ = os.path.splitext(csv_path)
            config_path = base + "_config.csv"
            request.session['config_filepath'] = config_path

        save_config_file(config_path, new_config)
        request.session['input_columns'] = input_cols

        # Update project with config path
        project_id = request.session.get('project_id')
        if project_id:
            update_project(project_id, config_path=config_path)

        return redirect('tagging')

    context = {
        'columns':      all_columns,
        'columns_json': json.dumps(all_columns),
        'config_data':  config_data,
        'mode':         mode,
        'image_models': [],
    }
    if mode == 'image':
        context['image_models'] = get_downloaded_image_models()
        context['active_image_connection'] = get_active_image_connection()
    return render(request, 'define_columns.html', context)


# ─── Tagging ─────────────────────────────────────────────────────────────────

def tagging_view(request):
    csv_path       = request.session.get('csv_filepath')
    config_path    = request.session.get('config_filepath')
    input_columns  = request.session.get('input_columns', [])
    project_id     = request.session.get('project_id')
    existing_key   = request.session.get('tagging_session_key')

    if not csv_path:
        return redirect('upload_file')

    # Monitor-only mode: if there's already an active session, just render the UI
    if existing_key and existing_key in PROGRESS_STATUS:
        ps = PROGRESS_STATUS[existing_key]
        if ps['status'] not in ('finished',) and not ps['status'].startswith('error'):
            return render(request, 'tagging.html', {'mode': _project_mode(request)})

    config_data = load_config_file(config_path) if config_path else []
    mode        = _project_mode(request)
    session_key = str(uuid.uuid4())
    request.session['tagging_session_key'] = session_key

    t = threading.Thread(
        target=row_by_row_tagger,
        args=(session_key, csv_path, config_path, input_columns, config_data),
        kwargs={'project_id': project_id, 'mode': mode},
        daemon=True,
    )
    t.start()

    return render(request, 'tagging.html', {'mode': mode})


def tagging_progress_view(request):
    session_key = request.session.get('tagging_session_key')
    if not session_key or session_key not in PROGRESS_STATUS:
        return JsonResponse({'error': 'No progress data found'}, status=400)

    progress_data = PROGRESS_STATUS[session_key]
    since         = int(request.GET.get('since', 0))
    live_logs     = progress_data.get("live_logs", [])
    new_logs      = live_logs[since:]

    tagged_file_path = cache.get(f"tagged_file_{session_key}")
    files_saved      = bool(tagged_file_path and os.path.exists(tagged_file_path))

    done      = progress_data["done"]
    total     = progress_data["total"]
    elapsed   = time.time() - progress_data.get("start_time", time.time())
    remaining = ((elapsed / done) * (total - done)) if done > 0 and total > done else None

    return JsonResponse({
        "done":        done,
        "total":       total,
        "status":      progress_data["status"],
        "paused":      PAUSE_FLAGS.get(session_key, False),
        "logs":        new_logs,
        "log_total":   len(live_logs),
        "files_saved": files_saved,
        "elapsed":     elapsed,
        "remaining":   remaining,
    })


def pause_tagging_view(request):
    session_key = request.session.get('tagging_session_key')
    if session_key:
        PAUSE_FLAGS[session_key] = True
    return JsonResponse({'paused': True})


def resume_tagging_view(request):
    session_key = request.session.get('tagging_session_key')
    if session_key:
        PAUSE_FLAGS[session_key] = False
    return JsonResponse({'paused': False})


# ─── Results ─────────────────────────────────────────────────────────────────

def results_view(request):
    session_key = request.session.get('tagging_session_key')
    tagged_file = cache.get(f"tagged_file_{session_key}")
    logs_file   = cache.get(f"logs_file_{session_key}")

    if not tagged_file or not os.path.exists(tagged_file):
        csv_path = request.session.get('csv_filepath')
        if csv_path and os.path.exists(csv_path):
            base, _ = os.path.splitext(csv_path)
            auto_tagged = base + "_tagged.csv"
            auto_logs   = base + "_logs.csv"
            if os.path.exists(auto_tagged):
                tagged_file = auto_tagged
                logs_file   = auto_logs if os.path.exists(auto_logs) else None
                cache.set(f"tagged_file_{session_key}", tagged_file, timeout=86400)
                if logs_file:
                    cache.set(f"logs_file_{session_key}", logs_file, timeout=86400)
            else:
                return redirect('tagging')
        else:
            return redirect('upload_file')

    df            = pd.read_csv(tagged_file)
    table_columns = df.columns.tolist()

    # Build cells tagged with whether they point at a generated image, so the
    # template can render a thumbnail instead of a raw path.
    image_exts = ('.png', '.jpg', '.jpeg', '.webp')
    table_rows = []
    for row in df.head(10).values.tolist():
        cells = []
        for cell in row:
            if isinstance(cell, float) and pd.isna(cell):
                cells.append({'value': '', 'is_image': False, 'url': ''})
                continue
            sval   = str(cell)
            is_img = sval.lower().endswith(image_exts)
            cells.append({
                'value':    sval,
                'is_image': is_img,
                'url':      (settings.MEDIA_URL + sval) if is_img else '',
            })
        table_rows.append(cells)

    tagged_file_url = settings.MEDIA_URL + os.path.basename(tagged_file)
    logs_file_url   = (settings.MEDIA_URL + os.path.basename(logs_file)) if logs_file and os.path.exists(logs_file) else None

    return render(request, 'results.html', {
        "tagged_file_url": tagged_file_url,
        "logs_file_url":   logs_file_url,
        "table_columns":   table_columns,
        "table_rows":      table_rows,
        "mode":            _project_mode(request),
    })


# ─── Connection ──────────────────────────────────────────────────────────────

def connection_editor_view(request):
    if request.method == 'POST':
        host  = request.POST.get('host', '').strip()
        port  = request.POST.get('port', '').strip()
        model = request.POST.get('model', '').strip()
        if host and port and model:
            save_connection(host, port, model)
            return JsonResponse({'success': True, 'message': 'Connection saved.'})
        return JsonResponse({'success': False, 'message': 'Host, port, and model are all required.'}, status=400)

    connections   = load_connections()
    active        = get_active_connection()
    unique_models = list(dict.fromkeys(c['model'] for c in connections))
    return render(request, 'connection.html', {
        'connections':   connections,
        'active':        active,
        'unique_models': unique_models,
    })


def test_connection_view(request):
    import urllib.request
    import json as _json
    host = request.GET.get('host', '').strip()
    port = request.GET.get('port', '').strip()
    if not host or not port:
        return JsonResponse({'success': False, 'error': 'Host and port are required.'}, status=400)
    url = f"http://{host}:{port}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=4) as resp:
            data   = _json.loads(resp.read())
            models = [m['name'] for m in data.get('models', [])]
            return JsonResponse({'success': True, 'models': models})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


# ─── Image backend (Stable Diffusion server) ─────────────────────────────────

def image_backend_view(request):
    """GET: model catalog + capability UI. POST: save the SD server host/port."""
    if request.method == 'POST':
        host  = request.POST.get('host', '').strip()
        port  = request.POST.get('port', '').strip()
        model = request.POST.get('model', '').strip()
        if host and port:
            save_image_connection(host, port, model)
            return JsonResponse({'success': True, 'message': 'Image backend saved.'})
        return JsonResponse({'success': False, 'message': 'Host and port are required.'}, status=400)

    return render(request, 'image_backend.html', {
        'connections': load_image_connections(),
        'active':      get_active_image_connection(),
    })


def image_capability_view(request):
    try:
        return JsonResponse({'success': True, 'capability': get_image_capability()})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


def image_models_view(request):
    try:
        return JsonResponse({'success': True, **get_image_models()})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


def image_download_view(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required.'}, status=400)
    model_id = request.POST.get('model_id', '').strip()
    hf_token = request.POST.get('hf_token', '').strip() or None
    if not model_id:
        return JsonResponse({'success': False, 'error': 'model_id is required.'}, status=400)
    try:
        return JsonResponse({'success': True, **start_image_download(model_id, hf_token)})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


def image_download_status_view(request):
    job_id = request.GET.get('job_id', '').strip()
    if not job_id:
        return JsonResponse({'success': False, 'error': 'job_id is required.'}, status=400)
    try:
        return JsonResponse({'success': True, **image_download_status(job_id)})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


def image_status_view(request):
    """Sidebar poll — active SD server + image-generation counters."""
    conn          = get_active_image_connection()
    request_count = cache.get(IMAGE_CACHE_KEYS["requests"], 0)
    total_time    = cache.get(IMAGE_CACHE_KEYS["total_time"], 0.0)
    return JsonResponse({
        "host":       conn.get('host', ''),
        "port":       conn.get('port', ''),
        "model":      conn.get('model', '') or '—',
        "requests":   request_count,
        "total_time": f"{total_time:.1f} sec",
        "status":     "Active" if request_count > 0 else "Idle",
    })
