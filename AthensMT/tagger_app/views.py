# tagger_app/views.py
import threading
import json
import uuid
import os
import re
import time
from itertools import zip_longest

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, HttpResponse
from django.core.cache import cache
import pandas as pd

from .forms import UploadForm
from .utils import (
    PROGRESS_STATUS,
    PAUSE_FLAGS,
    CANCEL_FLAGS,
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
    get_project,
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
    get_image_loras,
    get_downloaded_image_loras,
    start_image_lora_download,
    get_image_schedulers,
    get_image_server_health,
    get_image_server_status,
    get_image_server_metrics,
    request_image_generation_cancel,
    get_disk_usage,
    summarize_image_run_settings,
    get_aspect_ratio_presets,
    load_style_presets,
    render_tag_prompt,
    estimate_image_generation,
    regenerate_image_cell,
    list_image_candidates,
    select_image_candidate,
    load_review_state,
    set_review_state,
    BULK_RETRY_STATUS,
    bulk_retry_errors,
    bulk_retry_selected,
    compare_models_generate,
    images_dir_for_tagged_path,
    tagged_path_for_project,
    build_gallery_items,
    build_gallery_zip,
)


def _resolve_project_context(request):
    """(tagged_file, config_path, session_key, project_id) — resolved from an
    explicit `project_id` (POST or GET; used by the Gallery, which has no
    active Django session for whichever project's cards it's showing) or,
    when absent, from the current session (Results page). Returns all-None
    when neither resolves to an existing tagged file."""
    project_id = (request.POST.get('project_id') or request.GET.get('project_id') or '').strip()
    if project_id:
        proj = get_project(project_id)
        if not proj:
            return None, None, None, None
        tagged_file = tagged_path_for_project(proj)
        if not tagged_file or not os.path.exists(tagged_file):
            return None, None, None, None
        return tagged_file, proj.get('config_path', ''), proj.get('session_key') or None, project_id

    session_key = request.session.get('tagging_session_key')
    tagged_file = cache.get(f"tagged_file_{session_key}") if session_key else None
    if not tagged_file or not os.path.exists(tagged_file):
        return None, None, None, None
    return tagged_file, request.session.get('config_filepath'), session_key, request.session.get('project_id')


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

            # Each project gets its own media/<project_id>/ folder so two
            # uploads sharing a filename (e.g. two "test.csv" runs) never
            # collide on disk or bleed stale rows/images into each other.
            project_id  = str(uuid.uuid4())
            project_dir = os.path.join('media', project_id)
            os.makedirs(project_dir, exist_ok=True)

            fs       = FileSystemStorage(location=project_dir)
            csv_path = os.path.join(project_dir, csv_file.name)
            fs.save(csv_file.name, csv_file)
            csv_path = convert_upload_to_csv(csv_path)

            if config_file:
                config_path = os.path.join(project_dir, config_file.name)
                fs.save(config_file.name, config_file)
                config_path = convert_upload_to_csv(config_path)
            else:
                config_path = None

            request.session['csv_filepath']    = csv_path
            request.session['config_filepath'] = config_path
            request.session['project_mode']    = mode
            request.session['project_id']      = project_id
            # A brand-new upload must never resume a previous project's
            # tagging run — without this, tagging_view's "monitor-only"
            # branch would find the old (possibly paused/deleted) session
            # still alive in PROGRESS_STATUS and reattach to it instead of
            # starting a fresh run for this project.
            request.session.pop('tagging_session_key', None)

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
        image_naming_col = request.POST.get('image_naming_column', '').strip()
        image_format     = (request.POST.get('image_format', '').strip() or 'png').lower()

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

        # Update project with config path (+ image-mode naming/format settings)
        project_id = request.session.get('project_id')
        if project_id:
            update_kwargs = {'config_path': config_path}
            if mode == 'image':
                update_kwargs['image_naming_column'] = image_naming_col
                update_kwargs['image_format'] = image_format
            update_project(project_id, **update_kwargs)

        return redirect('tagging')

    context = {
        'columns':       all_columns,
        'columns_json':  json.dumps(all_columns),
        'config_data':   config_data,
        'mode':          mode,
        'image_models':  [],
        'image_loras':   [],
        'style_presets': [],
        'schedulers':    [],
        'aspect_presets': [],
        'row_count':     len(df),
    }
    if mode == 'image':
        context['image_models']   = get_downloaded_image_models()
        context['image_loras']    = get_downloaded_image_loras()
        context['style_presets']  = load_style_presets()
        context['schedulers']     = get_image_schedulers()
        context['aspect_presets'] = get_aspect_ratio_presets()
        context['active_image_connection'] = get_active_image_connection()

        # Naming column duplicate counts — surfaced next to the naming-column
        # picker so users see the collision risk before they start a run.
        dup_counts = {
            col: int(df[col].astype(str).duplicated(keep=False).sum())
            for col in all_columns
        }
        context['dup_counts_json'] = json.dumps(dup_counts)

        project_id = request.session.get('project_id')
        proj = get_project(project_id) if project_id else None
        context['image_naming_column'] = (proj.get('image_naming_column', '') if proj else '') or ''
        context['image_format'] = (proj.get('image_format', '') if proj else '') or 'png'
    return render(request, 'define_columns.html', context)


def estimate_image_view(request):
    """'Preview & Estimate' — run ONE real generation with the tag's current
    (possibly unsaved) settings against row 0 of the CSV, then extrapolate a
    total-time estimate from that real per-image cost."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required.'}, status=400)

    csv_path = request.session.get('csv_filepath')
    if not csv_path or not os.path.exists(csv_path):
        return JsonResponse({'success': False, 'error': 'No CSV loaded.'}, status=400)

    prompt_template = request.POST.get('prompt_template', '').strip()
    if not prompt_template:
        return JsonResponse({'success': False, 'error': 'Prompt template is required.'}, status=400)

    tag_input_cols = request.POST.get('tag_input_cols', '').strip()
    global_cols     = [c.strip() for c in request.POST.get('global_input_columns', '').split(',') if c.strip()]

    try:
        image_params = json.loads(request.POST.get('image_params') or '{}')
        if not isinstance(image_params, dict):
            image_params = {}
    except (ValueError, TypeError):
        image_params = {}

    df = read_csv_safe(csv_path)
    if df.empty:
        return JsonResponse({'success': False, 'error': 'CSV has no rows.'}, status=400)

    row0 = df.iloc[0]
    context_cols = [c for c in global_cols if c in df.columns] if global_cols else list(df.columns)
    full_context = {c: row0[c] for c in context_cols}
    all_context  = {c: row0[c] for c in df.columns}
    definition   = {'PromptTemplate': prompt_template, 'InputColumns': tag_input_cols}
    rendered_prompt, _ = render_tag_prompt(definition, full_context, all_context)

    elapsed_sec, image_url, error = estimate_image_generation(rendered_prompt, image_params)
    if error:
        return JsonResponse({'success': False, 'error': error})

    num_images = max(1, int(image_params.get('num_images') or 1))
    row_count  = len(df)

    return JsonResponse({
        'success':            True,
        'elapsed_sec':        round(elapsed_sec, 2),
        'image_url':          image_url,
        'row_count':          row_count,
        'num_images':         num_images,
        'total_estimate_sec': round(elapsed_sec * row_count * num_images, 1),
        'rendered_prompt':    rendered_prompt,
    })


# ─── Tagging ─────────────────────────────────────────────────────────────────

def tagging_view(request):
    csv_path       = request.session.get('csv_filepath')
    config_path    = request.session.get('config_filepath')
    input_columns  = request.session.get('input_columns', [])
    project_id     = request.session.get('project_id')
    existing_key   = request.session.get('tagging_session_key')

    if not csv_path:
        return redirect('upload_file')

    mode        = _project_mode(request)
    config_data = load_config_file(config_path) if config_path else []
    context     = {'mode': mode}
    if mode == 'image':
        context['run_info'] = summarize_image_run_settings(config_data, get_active_image_connection())

    # Monitor-only mode: if there's already an active session for THIS
    # project, just render the UI. The project_id check matters — a stale
    # tagging_session_key left over from a different (e.g. deleted) project
    # must never be reattached to here, or the page would show that other
    # project's stale/broken progress instead of starting a fresh run.
    if existing_key and existing_key in PROGRESS_STATUS:
        ps = PROGRESS_STATUS[existing_key]
        same_project = ps.get('project_id') == project_id
        if same_project and ps['status'] not in ('finished', 'cancelled') and not ps['status'].startswith('error'):
            return render(request, 'tagging.html', context)

    session_key = str(uuid.uuid4())
    request.session['tagging_session_key'] = session_key

    t = threading.Thread(
        target=row_by_row_tagger,
        args=(session_key, csv_path, config_path, input_columns, config_data),
        kwargs={'project_id': project_id, 'mode': mode},
        daemon=True,
    )
    t.start()

    return render(request, 'tagging.html', context)


def tagging_image_status_view(request):
    """Live SD-server + disk status for the tagging page's status panel
    (image mode only) — polled independently of the progress endpoint since
    it hits the SD server and disk, and shouldn't slow down the 1s progress
    poll or run on every other page's sidebar."""
    health  = get_image_server_health()
    status  = get_image_server_status()
    disk    = get_disk_usage()
    metrics = get_image_server_metrics()
    gpu     = metrics.get('gpu', {})
    return JsonResponse({
        'loaded_model':      health.get('loaded_model') or '',
        'server_reachable':  bool(health),
        'server_state':      status.get('state', ''),
        'server_detail':     status.get('detail', ''),
        'server_since':      status.get('since'),
        'server_logs':       status.get('logs', []),
        'disk_free_human':   disk['free_human']   if disk else None,
        'disk_total_human':  disk['total_human']  if disk else None,
        'disk_percent_free': disk['percent_free'] if disk else None,
        'cpu_percent':       metrics.get('cpu_percent'),
        'ram_used_mb':       metrics.get('ram_used_mb'),
        'ram_total_mb':      metrics.get('ram_total_mb'),
        'ram_percent':       metrics.get('ram_percent'),
        'gpu_backend':       gpu.get('backend'),
        'gpu_device_name':   gpu.get('device_name'),
        'gpu_util_percent':  gpu.get('utilization_percent'),
        'vram_used_mb':      gpu.get('vram_used_mb'),
        'vram_total_mb':     gpu.get('vram_total_mb'),
        'gpu_temp_c':        gpu.get('temperature_c'),
    })


def stop_tagging_view(request):
    """Hard stop, as opposed to pause: cancels the run outright instead of
    just blocking between tags. Reuses the same CANCEL_FLAGS mechanism
    delete_project() uses to unstick a paused thread whose project got
    deleted — here it's the same idea, just user-initiated. Also asks the
    SD server to interrupt an in-flight generation (best-effort — see
    request_image_generation_cancel for what it can and can't stop)."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required.'}, status=400)

    session_key = request.session.get('tagging_session_key')
    project_id  = request.session.get('project_id')
    if session_key and session_key in PROGRESS_STATUS:
        CANCEL_FLAGS[session_key] = True
        PAUSE_FLAGS[session_key] = False
        PROGRESS_STATUS[session_key]['status'] = 'stopping'
        if project_id:
            update_project(project_id, status='stopping')
        if _project_mode(request) == 'image':
            request_image_generation_cancel()

    return JsonResponse({'stopped': True})


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

    if not tagged_file or not os.path.exists(tagged_file):
        csv_path = request.session.get('csv_filepath')
        if csv_path and os.path.exists(csv_path):
            base, _ = os.path.splitext(csv_path)
            auto_tagged = base + "_tagged.csv"
            if os.path.exists(auto_tagged):
                tagged_file = auto_tagged
                cache.set(f"tagged_file_{session_key}", tagged_file, timeout=86400)
            else:
                return redirect('tagging')
        else:
            return redirect('upload_file')

    df            = pd.read_csv(tagged_file)
    table_columns = df.columns.tolist()
    mode          = _project_mode(request)
    review_state  = load_review_state(tagged_file) if mode == 'image' else {}
    review_filter = (request.GET.get('filter', 'all') or 'all') if mode == 'image' else 'all'

    # Build cells tagged with whether they point at a generated image, so the
    # template can render a thumbnail (+ grid picker / regenerate) instead of
    # a raw path.
    image_exts = ('.png', '.jpg', '.jpeg', '.webp')

    def build_row(row_index, row):
        cells = []
        row_matches = (review_filter == 'all')
        for col in table_columns:
            cell = row[col]
            if isinstance(cell, float) and pd.isna(cell):
                cells.append({'value': '', 'is_image': False, 'url': '', 'candidates_json': '[]',
                              'column': col, 'row_index': int(row_index), 'review': ''})
                continue
            sval   = str(cell)
            is_img = sval.lower().endswith(image_exts)
            is_err = sval.startswith('ERROR:')
            review = review_state.get(f"{row_index}:{col}", '') if (mode == 'image' and is_img) else ''
            candidates = list_image_candidates(tagged_file, row_index, col) if (mode == 'image' and (is_img or is_err)) else []
            if mode == 'image' and review_filter != 'all' and (is_img or is_err):
                if review_filter == 'error' and is_err:
                    row_matches = True
                elif review_filter == 'unreviewed' and is_img and not review:
                    row_matches = True
                elif review_filter in ('approved', 'rejected') and review == review_filter:
                    row_matches = True
            cells.append({
                'value':          sval,
                'is_image':       is_img,
                'is_image_error': is_err and mode == 'image',
                'url':            (settings.MEDIA_URL + sval) if is_img else '',
                'candidates_json': json.dumps(candidates),
                'column':         col,
                'row_index':      int(row_index),
                'review':         review,
            })
        return cells, row_matches

    table_rows = []
    if mode == 'image':
        MAX_ROWS = 50
        for row_index, row in df.iterrows():
            cells, matches = build_row(row_index, row)
            if matches:
                table_rows.append(cells)
            if len(table_rows) >= MAX_ROWS:
                break
        err_mask = df[table_columns].astype(str).apply(lambda col: col.str.startswith('ERROR:'))
        error_cell_count = int(err_mask.values.sum())
    else:
        for row_index, row in df.head(10).iterrows():
            cells, _ = build_row(row_index, row)
            table_rows.append(cells)
        error_cell_count = 0

    def _media_rel(p):
        return os.path.relpath(p, settings.MEDIA_ROOT).replace(os.sep, '/')

    tagged_file_url = settings.MEDIA_URL + _media_rel(tagged_file)

    return render(request, 'results.html', {
        "tagged_file_url":   tagged_file_url,
        "table_columns":     table_columns,
        "table_rows":        table_rows,
        "mode":              mode,
        "review_filter":     review_filter,
        "error_cell_count":  error_cell_count,
        "total_rows":        len(df),
        "shown_rows":        len(table_rows),
        "analytics_columns": _build_yes_no_analytics(df, table_columns),
    })


# Categorical breakdown shown on the Results "Analytics" tab: any column
# (over the FULL tagged dataset, not just the preview rows) whose non-empty
# values are entirely YES/NO/N-A. Colors mirror the app's status palette so
# YES/NO read as affirmative/negative at a glance.
_ANALYTICS_CATEGORIES = ['YES', 'NO', 'N/A']
_ANALYTICS_COLORS = {'YES': '#0ca30c', 'NO': '#d03b3b', 'N/A': '#898781'}


def _build_yes_no_analytics(df, table_columns):
    analytics = []
    for col in table_columns:
        if col.endswith('_exp'):
            continue
        values = df[col].dropna().astype(str).str.strip().str.upper()
        values = values[values != '']
        values = values.replace({'NA': 'N/A'})
        if values.empty:
            continue
        uniques = set(values.unique())
        if not uniques <= set(_ANALYTICS_CATEGORIES) or not (uniques & {'YES', 'NO'}):
            continue

        total = len(values)
        segments = []
        for cat in _ANALYTICS_CATEGORIES:
            count = int((values == cat).sum())
            if count:
                segments.append({
                    'label': cat,
                    'count': count,
                    'pct':   round(count / total * 100, 1),
                    'color': _ANALYTICS_COLORS[cat],
                })
        analytics.append({'column': col, 'total': total, 'segments': segments})
    return analytics


def set_review_view(request):
    """Lightweight approve/reject — one click, no CSV write, sidecar JSON only."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required.'}, status=400)

    tagged_file, _config_path, _session_key, _project_id = _resolve_project_context(request)
    if not tagged_file:
        return JsonResponse({'success': False, 'error': 'No tagged file for this session.'}, status=400)

    try:
        row_index = int(request.POST.get('row_index'))
    except (TypeError, ValueError):
        return JsonResponse({'success': False, 'error': 'row_index is required.'}, status=400)
    column = request.POST.get('column', '').strip()
    status = request.POST.get('status', '').strip()
    if not column:
        return JsonResponse({'success': False, 'error': 'column is required.'}, status=400)
    if status not in ('approved', 'rejected', ''):
        return JsonResponse({'success': False, 'error': 'status must be approved, rejected, or empty.'}, status=400)

    set_review_state(tagged_file, row_index, column, status)
    return JsonResponse({'success': True, 'status': status})


def bulk_retry_view(request):
    """Kick off a background job retrying every image cell holding an
    'ERROR: ...' value. Mirrors tagging_view's session_key + thread pattern."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required.'}, status=400)

    session_key = request.session.get('tagging_session_key')
    tagged_file = cache.get(f"tagged_file_{session_key}") if session_key else None
    if not tagged_file or not os.path.exists(tagged_file):
        return JsonResponse({'success': False, 'error': 'No tagged file for this session.'}, status=400)

    config_path = request.session.get('config_filepath')
    config_data = load_config_file(config_path) if config_path else []
    images_dir, images_rel = images_dir_for_tagged_path(tagged_file)
    os.makedirs(images_dir, exist_ok=True)

    job_key = str(uuid.uuid4())
    request.session['bulk_retry_job_key'] = job_key
    t = threading.Thread(
        target=bulk_retry_errors,
        args=(job_key, tagged_file, config_data, images_dir, images_rel),
        kwargs={'session_key': session_key, 'project_id': request.session.get('project_id')},
        daemon=True,
    )
    t.start()
    return JsonResponse({'success': True, 'job_key': job_key})


def bulk_retry_status_view(request):
    job_key = request.GET.get('job_key', '').strip() or request.session.get('bulk_retry_job_key', '')
    if not job_key or job_key not in BULK_RETRY_STATUS:
        return JsonResponse({'success': False, 'error': 'No such job.'}, status=400)
    return JsonResponse({'success': True, **BULK_RETRY_STATUS[job_key]})


def regenerate_image_view(request):
    """'Retry' — re-run generation for one row/column. Fresh random seed by
    default; pass lock_seed=1 to reuse the seed behind the currently-shown
    candidate instead (for iterating on the prompt/settings without losing
    the composition)."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required.'}, status=400)

    tagged_file, config_path, session_key, project_id = _resolve_project_context(request)
    if not tagged_file:
        return JsonResponse({'success': False, 'error': 'No tagged file for this session.'}, status=400)

    try:
        row_index = int(request.POST.get('row_index'))
    except (TypeError, ValueError):
        return JsonResponse({'success': False, 'error': 'row_index is required.'}, status=400)
    out_col = request.POST.get('column', '').strip()
    if not out_col:
        return JsonResponse({'success': False, 'error': 'column is required.'}, status=400)
    lock_seed = request.POST.get('lock_seed', '').strip() == '1'

    config_data = load_config_file(config_path) if config_path else []
    images_dir, images_rel = images_dir_for_tagged_path(tagged_file)
    os.makedirs(images_dir, exist_ok=True)

    try:
        saved_rel, seed_used = regenerate_image_cell(
            tagged_file, config_data, row_index, out_col, images_dir, images_rel,
            session_key=session_key, project_id=project_id,
            lock_seed=lock_seed,
        )
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({
        'success':    True,
        'image_url':  settings.MEDIA_URL + saved_rel[0],
        'image_urls': [settings.MEDIA_URL + p for p in saved_rel],
        'seed_used':  seed_used,
    })


def select_image_view(request):
    """Grid picker — point a cell at a different already-generated candidate."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required.'}, status=400)

    tagged_file, _config_path, _session_key, _project_id = _resolve_project_context(request)
    if not tagged_file:
        return JsonResponse({'success': False, 'error': 'No tagged file for this session.'}, status=400)

    try:
        row_index = int(request.POST.get('row_index'))
    except (TypeError, ValueError):
        return JsonResponse({'success': False, 'error': 'row_index is required.'}, status=400)
    out_col  = request.POST.get('column', '').strip()
    rel_path = request.POST.get('rel_path', '').strip()
    if not out_col or not rel_path:
        return JsonResponse({'success': False, 'error': 'column and rel_path are required.'}, status=400)

    try:
        select_image_candidate(tagged_file, row_index, out_col, rel_path)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': True})


# ─── Gallery ─────────────────────────────────────────────────────────────────

def _slug(name):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(name)).strip('_') or 'project'


def gallery_view(request, project_id=None):
    """Per-project (project_id given) or cross-project (project_id=None)
    image gallery — resolved straight from projects.csv, independent of
    whatever session/project the user currently has open."""
    image_projects = [p for p in load_projects() if p.get('mode') == 'image']

    if project_id:
        current_project = next((p for p in image_projects if str(p['project_id']) == str(project_id)), None)
        if not current_project:
            return redirect('home')
        target_projects = [current_project]
    else:
        current_project = None
        project_filter = request.GET.get('project', '').strip()
        target_projects = (
            [p for p in image_projects if str(p['project_id']) == project_filter]
            if project_filter else image_projects
        )

    all_items = []
    for proj in target_projects:
        proj_items, _tagged, _dir, _rel = build_gallery_items(proj)
        all_items.extend(proj_items)

    error_count = sum(1 for it in all_items if it['is_error'])
    review_filter = request.GET.get('filter', 'all') or 'all'

    if review_filter == 'all':
        items = all_items
    else:
        def matches(it):
            if review_filter == 'error':
                return it['is_error']
            if review_filter == 'unreviewed':
                return it['is_image'] and not it['review']
            return it['is_image'] and it['review'] == review_filter
        items = [it for it in all_items if matches(it)]

    MAX_ITEMS = 300
    truncated = len(items) > MAX_ITEMS
    items = items[:MAX_ITEMS]

    return render(request, 'gallery.html', {
        'items':              items,
        'is_cross_project':   project_id is None,
        'project':            current_project,
        'projects_available': image_projects,
        'project_filter':     request.GET.get('project', ''),
        'review_filter':      review_filter,
        'error_count':        error_count,
        'total_items':        len(all_items),
        'shown_items':        len(items),
        'truncated':          truncated,
    })


def gallery_retry_view(request):
    """Multi-select 'retry' — regenerate an explicit set of (project, row,
    column) cells, possibly spanning several projects, reusing each cell's
    own config/prompt (via regenerate_image_cell)."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required.'}, status=400)

    try:
        payload = json.loads(request.body or '{}')
    except ValueError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON body.'}, status=400)

    raw_items = payload.get('items') or []
    lock_seed = bool(payload.get('lock_seed'))
    if not raw_items:
        return JsonResponse({'success': False, 'error': 'No items selected.'}, status=400)

    by_project = {}
    for it in raw_items:
        pid = str(it.get('project_id', '')).strip()
        col = str(it.get('column', '')).strip()
        try:
            row_index = int(it.get('row_index'))
        except (TypeError, ValueError):
            continue
        if not pid or not col:
            continue
        by_project.setdefault(pid, []).append((row_index, col))

    groups = []
    for pid, targets in by_project.items():
        proj = get_project(pid)
        if not proj:
            continue
        tagged_file = tagged_path_for_project(proj)
        if not tagged_file or not os.path.exists(tagged_file):
            continue
        config_data = load_config_file(proj.get('config_path', ''))
        images_dir, images_rel = images_dir_for_tagged_path(tagged_file)
        os.makedirs(images_dir, exist_ok=True)
        groups.append({
            'tagged_path': tagged_file, 'config_data': config_data,
            'images_dir': images_dir, 'images_rel': images_rel,
            'session_key': proj.get('session_key') or None, 'project_id': pid,
            'targets': targets,
        })

    if not groups:
        return JsonResponse({'success': False, 'error': 'No valid items to retry.'}, status=400)

    job_key = str(uuid.uuid4())
    t = threading.Thread(
        target=bulk_retry_selected,
        args=(job_key, groups),
        kwargs={'lock_seed': lock_seed},
        daemon=True,
    )
    t.start()
    return JsonResponse({'success': True, 'job_key': job_key})


def gallery_retry_status_view(request):
    job_key = request.GET.get('job_key', '').strip()
    if not job_key or job_key not in BULK_RETRY_STATUS:
        return JsonResponse({'success': False, 'error': 'No such job.'}, status=400)
    return JsonResponse({'success': True, **BULK_RETRY_STATUS[job_key]})


def gallery_zip_view(request):
    """GET ?project_id=<id> -> whole-project zip; GET with no project_id ->
    every image-mode project zipped, one subfolder each; POST {items: [...]}
    -> just the selected images, grouped by project."""
    selected_items = None
    if request.method == 'POST':
        try:
            payload = json.loads(request.body or '{}')
        except ValueError:
            payload = {}
        selected_items = payload.get('items')

    if selected_items:
        by_project = {}
        for it in selected_items:
            pid = str(it.get('project_id', '')).strip()
            rel = str(it.get('rel_path', '')).strip()
            if pid and rel:
                by_project.setdefault(pid, []).append(rel)
        entries = []
        for pid, rels in by_project.items():
            proj = get_project(pid)
            if proj:
                entries.append((proj, rels))
        if not entries:
            return JsonResponse({'success': False, 'error': 'No valid items selected.'}, status=400)
        fname = f"gallery_selected_{int(time.time())}.zip"
    else:
        project_id = request.GET.get('project_id', '').strip()
        if project_id:
            proj = get_project(project_id)
            if not proj:
                return JsonResponse({'success': False, 'error': 'No such project.'}, status=404)
            entries = [(proj, None)]
            fname = f"{_slug(proj.get('name') or project_id)}_images_{int(time.time())}.zip"
        else:
            projects = [p for p in load_projects() if p.get('mode') == 'image']
            entries = [(p, None) for p in projects]
            fname = f"gallery_all_{int(time.time())}.zip"

    zip_bytes = build_gallery_zip(entries)
    response = HttpResponse(zip_bytes, content_type='application/zip')
    response['Content-Disposition'] = f'attachment; filename="{fname}"'
    return response


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
    """Downloads a base model by default; pass kind=lora to download a LoRA
    instead (tracked separately server-side since both are plain HF repos)."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required.'}, status=400)
    model_id = request.POST.get('model_id', '').strip()
    hf_token = request.POST.get('hf_token', '').strip() or None
    kind     = request.POST.get('kind', 'model').strip() or 'model'
    if not model_id:
        return JsonResponse({'success': False, 'error': 'model_id is required.'}, status=400)
    try:
        if kind == 'lora':
            result = start_image_lora_download(model_id, hf_token)
        else:
            result = start_image_download(model_id, hf_token)
        return JsonResponse({'success': True, **result})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


def image_loras_view(request):
    try:
        return JsonResponse({'success': True, **get_image_loras()})
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


MAX_COMPARE_MODELS = 6


def compare_models_view(request):
    """Same prompt/settings, one image per selected model — side-by-side."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required.'}, status=400)

    prompt = request.POST.get('prompt', '').strip()
    if not prompt:
        return JsonResponse({'success': False, 'error': 'Prompt is required.'}, status=400)

    model_ids = [m.strip() for m in request.POST.getlist('model_ids') if m.strip()]
    if not model_ids:
        return JsonResponse({'success': False, 'error': 'Select at least one model.'}, status=400)
    if len(model_ids) > MAX_COMPARE_MODELS:
        return JsonResponse({'success': False,
                             'error': f'Compare at most {MAX_COMPARE_MODELS} models at a time.'}, status=400)

    params = {
        'negative_prompt': request.POST.get('negative_prompt', ''),
        'width':    request.POST.get('width', 512),
        'height':   request.POST.get('height', 512),
        'steps':    request.POST.get('steps', 30),
        'guidance': request.POST.get('guidance', 7.5),
        'seed':     request.POST.get('seed', -1),
    }
    results = compare_models_generate(prompt, params, model_ids)
    return JsonResponse({'success': True, 'results': results})
