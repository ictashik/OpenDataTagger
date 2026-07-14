import threading
import openai
import pandas as pd
import os
import re
import time
import json
import base64
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime
from django.core.cache import cache
from django.conf import settings

LLM_CACHE_KEYS = {
    "requests": "llm_request_count",
    "total_time": "llm_total_inference_time",
}
IMAGE_CACHE_KEYS = {
    "requests": "img_request_count",
    "total_time": "img_total_inference_time",
}

_base = os.path.dirname(os.path.abspath(__file__))
CONNECTIONS_CSV       = os.path.join(_base, '..', 'connections.csv')
IMAGE_CONNECTIONS_CSV = os.path.join(_base, '..', 'image_connections.csv')
PROJECTS_CSV          = os.path.join(_base, '..', 'projects.csv')
STATS_CSV             = os.path.join(_base, '..', 'stats.csv')

PAUSE_FLAGS = {}     # session_key -> bool  (True = paused)
PROGRESS_STATUS = {} # session_key -> dict

_stats_lock    = threading.Lock()
_projects_lock = threading.Lock()

_DEFAULT_CONNECTION = {
    'host': '10.60.23.102',
    'port': '11434',
    'model': 'gemma3:27b',
}

# Fallback SD server if image_connections.csv is empty (local server on localhost).
_DEFAULT_IMAGE_CONNECTION = {
    'host':  getattr(settings, 'SD_SERVER_DEFAULT', {}).get('host', 'localhost'),
    'port':  getattr(settings, 'SD_SERVER_DEFAULT', {}).get('port', '7860'),
    'model': '',
}

# Image generation can take a while (large models / many steps).
SD_TIMEOUT = 600


def read_csv_safe(path, **kwargs):
    """Read a user-supplied CSV trying common encodings before giving up.

    Uploaded files are frequently exported from Excel as Windows-1252/Latin-1
    rather than UTF-8, which raises UnicodeDecodeError with plain read_csv.
    """
    for encoding in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    # latin-1 maps every byte 0-255, so this line is effectively unreachable,
    # but fall back to it explicitly with replacement as a last resort.
    return pd.read_csv(path, encoding='latin-1', encoding_errors='replace', **kwargs)


def convert_upload_to_csv(path):
    """If an uploaded file is actually an Excel workbook, convert it to CSV.

    The upload form's file picker only suggests .csv (accept=".csv"), but
    browsers let users pick "All files" and select an .xlsx/.xls export
    instead. Reading that binary data as text raises UnicodeDecodeError, so
    detect the real file type from its signature (not just its extension)
    and transparently convert it, returning the (possibly new) path.
    """
    try:
        with open(path, 'rb') as f:
            sig = f.read(8)
    except OSError:
        return path

    is_xlsx = sig[:4] == b'PK\x03\x04'
    is_xls  = sig[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'
    if not (is_xlsx or is_xls):
        return path

    df = pd.read_excel(path)
    new_path = os.path.splitext(path)[0] + '.csv'
    df.to_csv(new_path, index=False)
    if os.path.normpath(new_path) != os.path.normpath(path):
        os.remove(path)
    return new_path


# ─── Connections ─────────────────────────────────────────────────────────────

def load_connections():
    path = os.path.normpath(CONNECTIONS_CSV)
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if not {'host', 'port', 'model', 'last_used'}.issubset(df.columns):
            return []
        df['port'] = df['port'].astype(str)
        return df.sort_values('last_used', ascending=False).to_dict('records')
    except Exception as e:
        print(f"Error loading connections: {e}")
        return []


def save_connection(host, port, model):
    path = os.path.normpath(CONNECTIONS_CSV)
    connections = load_connections()
    port = str(port)
    existing = next(
        (c for c in connections if c['host'] == host and str(c['port']) == port and c['model'] == model),
        None
    )
    if existing:
        existing['last_used'] = datetime.now().isoformat()
    else:
        connections.insert(0, {'host': host, 'port': port, 'model': model,
                               'last_used': datetime.now().isoformat()})
    connections.sort(key=lambda x: x['last_used'], reverse=True)
    pd.DataFrame(connections).to_csv(path, index=False)
    return connections


def get_active_connection():
    connections = load_connections()
    return connections[0] if connections else dict(_DEFAULT_CONNECTION)


def get_llm_client():
    conn = get_active_connection()
    client = openai.OpenAI(base_url=f"http://{conn['host']}:{conn['port']}/v1", api_key='ollama')
    return client, conn['model']


# ─── Image backend (Stable Diffusion server) ─────────────────────────────────
#
# Mirrors the LLM connection layer above, but points at the standalone SD server
# (see sd_server/). ODT only makes plain HTTP calls — no torch/diffusers here.

def load_image_connections():
    path = os.path.normpath(IMAGE_CONNECTIONS_CSV)
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if not {'host', 'port', 'model', 'last_used'}.issubset(df.columns):
            return []
        df['port'] = df['port'].astype(str)
        df['model'] = df['model'].fillna('').astype(str)
        return df.sort_values('last_used', ascending=False).to_dict('records')
    except Exception as e:
        print(f"Error loading image connections: {e}")
        return []


def save_image_connection(host, port, model=''):
    path = os.path.normpath(IMAGE_CONNECTIONS_CSV)
    connections = load_image_connections()
    port = str(port)
    existing = next(
        (c for c in connections if c['host'] == host and str(c['port']) == port and str(c.get('model', '')) == model),
        None
    )
    if existing:
        existing['last_used'] = datetime.now().isoformat()
    else:
        connections.insert(0, {'host': host, 'port': port, 'model': model,
                               'last_used': datetime.now().isoformat()})
    connections.sort(key=lambda x: x['last_used'], reverse=True)
    pd.DataFrame(connections).to_csv(path, index=False)
    return connections


def get_active_image_connection():
    connections = load_image_connections()
    return connections[0] if connections else dict(_DEFAULT_IMAGE_CONNECTION)


def _sd_base_url(conn=None):
    conn = conn or get_active_image_connection()
    return f"http://{conn['host']}:{conn['port']}"


def _sd_request(path, payload=None, method='GET', timeout=10):
    """Call the SD server. GET when payload is None, else JSON POST.
    Returns parsed JSON (dict). Raises on transport/HTTP errors."""
    url = _sd_base_url() + path
    data = None
    headers = {}
    if payload is not None or method == 'POST':
        data = json.dumps(payload or {}).encode('utf-8')
        headers['Content-Type'] = 'application/json'
        method = 'POST'
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def get_image_capability(timeout=6):
    return _sd_request('/capability', timeout=timeout)


def get_image_models(timeout=10):
    return _sd_request('/models', timeout=timeout)


def start_image_download(model_id, hf_token=None, timeout=10):
    return _sd_request('/download', {'model_id': model_id, 'hf_token': hf_token or None},
                       method='POST', timeout=timeout)


def image_download_status(job_id, timeout=6):
    return _sd_request(f'/download/status?job_id={urllib.parse.quote(job_id)}', timeout=timeout)


def get_downloaded_image_models():
    """List of catalog/local models already downloaded — for the model dropdown.
    Returns [] if the SD server is unreachable."""
    try:
        data = get_image_models()
        return [m for m in data.get('models', []) if m.get('downloaded')]
    except Exception as e:
        print(f"Image models unavailable: {e}")
        return []


def _coerce_int(val, default):
    try:
        s = str(val).strip()
        if s == '' or s.lower() == 'none':
            return default
        return int(float(s))
    except (TypeError, ValueError):
        return default


def _coerce_float(val, default):
    try:
        s = str(val).strip()
        if s == '' or s.lower() == 'none':
            return default
        return float(s)
    except (TypeError, ValueError):
        return default


def call_image_generation(prompt, params):
    """Generate image(s) via the SD server for one rendered prompt.

    params keys (all optional except model): model, negative_prompt, width,
    height, steps, guidance, seed, num_images, hf_token.

    Returns (image_bytes_list, meta, usage_dict). usage keys mirror the LLM
    path (prompt_tokens/completion_tokens are 0 for images) so record_stat is
    reused unchanged. On failure image_bytes_list is empty and meta['error']
    is set.
    """
    conn  = get_active_image_connection()
    model = (params.get('model') or conn.get('model') or '').strip()

    request_count = cache.get(IMAGE_CACHE_KEYS["requests"], 0) + 1
    cache.set(IMAGE_CACHE_KEYS["requests"], request_count, None)

    payload = {
        'model_id':        model,
        'prompt':          prompt,
        'negative_prompt': params.get('negative_prompt', '') or '',
        'width':           _coerce_int(params.get('width'), 512),
        'height':          _coerce_int(params.get('height'), 512),
        'steps':           _coerce_int(params.get('steps'), 30),
        'guidance_scale':  _coerce_float(params.get('guidance'), 7.5),
        'seed':            _coerce_int(params.get('seed'), -1),
        'num_images':      max(1, _coerce_int(params.get('num_images'), 1)),
        'hf_token':        params.get('hf_token') or None,
    }

    start = time.time()
    usage = {
        'prompt_tokens': 0, 'completion_tokens': 0, 'elapsed_sec': 0,
        'host': conn['host'], 'port': conn['port'], 'model': model,
    }
    try:
        data = _sd_request('/generate', payload, method='POST', timeout=SD_TIMEOUT)
        elapsed = time.time() - start
        usage['elapsed_sec'] = elapsed
        cache.set(IMAGE_CACHE_KEYS["total_time"],
                  cache.get(IMAGE_CACHE_KEYS["total_time"], 0.0) + elapsed, None)
        images = [base64.b64decode(b) for b in data.get('images', [])]
        meta = {'seed_used': data.get('seed_used'),
                'server_elapsed': data.get('elapsed_sec')}
        return images, meta, usage
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode('utf-8')).get('detail', str(e))
        except Exception:
            detail = str(e)
        usage['elapsed_sec'] = time.time() - start
        print(f"Image generation HTTP error: {detail}")
        return [], {'error': detail}, usage
    except Exception as e:
        usage['elapsed_sec'] = time.time() - start
        print(f"Image generation error: {e}")
        return [], {'error': str(e)}, usage


def _generate_image_for_tag(definition, rendered_prompt, images_dir, images_rel,
                            row_index, out_col, session_key, project_id):
    """Run one image generation for a tag, save the PNG(s), record a stat.

    Returns (cell_value, explanation, image_url): cell_value is the
    MEDIA_ROOT-relative path of the first image written into the tagged CSV
    (or an "ERROR: …" string on failure); image_url is that path prefixed with
    MEDIA_URL for the live-log thumbnail ('' on failure).
    """
    try:
        params = json.loads(definition.get('ImageParams') or '{}')
        if not isinstance(params, dict):
            params = {}
    except (ValueError, TypeError):
        params = {}

    images, meta, usage = call_image_generation(rendered_prompt, params)
    record_stat(
        usage['host'], usage['port'], usage['model'],
        session_key, project_id,
        usage['prompt_tokens'], usage['completion_tokens'], usage['elapsed_sec'],
    )

    if not images:
        err = meta.get('error', 'unknown error')
        return f"ERROR: {err}", f"Image generation failed: {err}", ''

    safe_col = re.sub(r'[^A-Za-z0-9_-]+', '_', str(out_col)).strip('_') or 'img'
    saved_rel = []
    for n, img_bytes in enumerate(images):
        fname = f"row{row_index}_{safe_col}_{n}.png"
        with open(os.path.join(images_dir, fname), 'wb') as fh:
            fh.write(img_bytes)
        saved_rel.append(f"{images_rel}/{fname}")

    cell_value = saved_rel[0]
    image_url  = settings.MEDIA_URL + saved_rel[0]
    model = params.get('model') or usage.get('model', '')
    seed  = meta.get('seed_used')
    explanation = f"Generated with {model}" + (f" (seed {seed})" if seed is not None else "")
    return cell_value, explanation, image_url


# ─── Projects ────────────────────────────────────────────────────────────────

def load_projects():
    path = os.path.normpath(PROJECTS_CSV)
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if not {'project_id', 'name', 'status'}.issubset(df.columns):
            return []
        for col in ('total_rows', 'done_rows'):
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        for col in ('csv_path', 'config_path', 'session_key', 'status'):
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        # Project mode: 'text' (LLM tagging) or 'image' (Stable Diffusion).
        # Older registries have no column — default them to 'text'.
        if 'mode' in df.columns:
            df['mode'] = df['mode'].fillna('text').replace('', 'text').astype(str)
        else:
            df['mode'] = 'text'
        return df.sort_values('last_updated', ascending=False).to_dict('records')
    except Exception as e:
        print(f"Error loading projects: {e}")
        return []


def save_project(project_id, name, csv_path, config_path='', mode='text'):
    path = os.path.normpath(PROJECTS_CSV)
    with _projects_lock:
        projects = load_projects()
        now = datetime.now().isoformat()
        existing = next((p for p in projects if str(p['project_id']) == str(project_id)), None)
        if not existing:
            projects.insert(0, {
                'project_id': project_id,
                'name': name,
                'csv_path': csv_path,
                'config_path': config_path or '',
                'created_at': now,
                'last_updated': now,
                'status': 'idle',
                'total_rows': 0,
                'done_rows': 0,
                'session_key': '',
                'mode': mode or 'text',
            })
        else:
            existing['last_updated'] = now
            existing['config_path'] = config_path or existing.get('config_path', '')
            existing['mode'] = mode or existing.get('mode', 'text')
        pd.DataFrame(projects).to_csv(path, index=False)


def update_project(project_id, **kwargs):
    path = os.path.normpath(PROJECTS_CSV)
    with _projects_lock:
        projects = load_projects()
        for p in projects:
            if str(p['project_id']) == str(project_id):
                p.update(kwargs)
                p['last_updated'] = datetime.now().isoformat()
                break
        pd.DataFrame(projects).to_csv(path, index=False)


def delete_project(project_id):
    path = os.path.normpath(PROJECTS_CSV)
    with _projects_lock:
        projects = load_projects()
        projects = [p for p in projects if str(p['project_id']) != str(project_id)]
        pd.DataFrame(projects).to_csv(path, index=False)


# ─── Stats ───────────────────────────────────────────────────────────────────

def record_stat(host, port, model, session_key, project_id,
                prompt_tokens, completion_tokens, elapsed_sec):
    path = os.path.normpath(STATS_CSV)
    row = pd.DataFrame([{
        'timestamp':        datetime.now().isoformat(),
        'host':             host,
        'port':             str(port),
        'model':            model,
        'session_key':      session_key or '',
        'project_id':       project_id or '',
        'prompt_tokens':    int(prompt_tokens),
        'completion_tokens': int(completion_tokens),
        'elapsed_sec':      round(float(elapsed_sec), 3),
    }])
    with _stats_lock:
        write_header = not os.path.exists(path)
        row.to_csv(path, mode='a', header=write_header, index=False)


def get_host_stats():
    """Return per-(host, port, model) aggregated stats."""
    path = os.path.normpath(STATS_CSV)
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if df.empty:
            return []
        agg = df.groupby(['host', 'port', 'model'], as_index=False).agg(
            requests=('elapsed_sec', 'count'),
            total_sec=('elapsed_sec', 'sum'),
            prompt_tokens=('prompt_tokens', 'sum'),
            completion_tokens=('completion_tokens', 'sum'),
        )
        agg['total_sec'] = agg['total_sec'].round(1)
        return agg.sort_values('total_sec', ascending=False).to_dict('records')
    except Exception as e:
        print(f"Error loading stats: {e}")
        return []


# ─── LLM call ────────────────────────────────────────────────────────────────

def call_llm_tagging(system_prompt, user_prompt):
    """
    Returns (best_answer, explanation, usage_dict).
    usage_dict keys: prompt_tokens, completion_tokens, elapsed_sec, host, port, model.
    """
    conn = get_active_connection()
    try:
        request_count = cache.get(LLM_CACHE_KEYS["requests"], 0) + 1
        cache.set(LLM_CACHE_KEYS["requests"], request_count, None)

        start_time = time.time()
        client, model_name = get_llm_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]
        )
        elapsed_time = time.time() - start_time

        total_time = cache.get(LLM_CACHE_KEYS["total_time"], 0.0) + elapsed_time
        cache.set(LLM_CACHE_KEYS["total_time"], total_time, None)

        message = response.choices[0].message.content.strip()
        if "Best Answer:" in message and "Explanation:" in message:
            parts = message.split("Explanation:")
            best_answer = parts[0].replace("Best Answer:", "").strip()
            explanation = parts[1].strip()
        else:
            best_answer = message.strip()
            explanation = "No explanation provided."

        usage = getattr(response, 'usage', None)
        return best_answer, explanation, {
            'prompt_tokens':     getattr(usage, 'prompt_tokens',     0) if usage else 0,
            'completion_tokens': getattr(usage, 'completion_tokens', 0) if usage else 0,
            'elapsed_sec':       elapsed_time,
            'host':              conn['host'],
            'port':              conn['port'],
            'model':             conn['model'],
        }

    except Exception as e:
        print(f"LLM API error: {e}")
        return "ERROR", "LLM call failed.", {
            'prompt_tokens': 0, 'completion_tokens': 0, 'elapsed_sec': 0,
            'host': conn['host'], 'port': conn['port'], 'model': conn['model'],
        }


# ─── Config file ─────────────────────────────────────────────────────────────

def load_config_file(config_path):
    if not config_path or not os.path.exists(config_path):
        return []
    try:
        df = read_csv_safe(config_path)
        if 'OutputColumn' not in df.columns or 'PromptTemplate' not in df.columns:
            return []
        records = df.to_dict('records')
        str_fields = ('ConditionField', 'ConditionOp', 'ConditionValue',
                      'DefaultValue', 'SendContext', 'InputColumns', 'ImageParams')
        for r in records:
            r.setdefault('ConditionField', '')
            r.setdefault('ConditionOp',    '==')
            r.setdefault('ConditionValue', '')
            r.setdefault('DefaultValue',   '')
            r.setdefault('SendContext',    '')
            r.setdefault('InputColumns',   '')
            r.setdefault('ImageParams',    '')
            for k in str_fields:
                val = r[k]
                if not isinstance(val, str):
                    # pandas reads '0'/'1' as int; NaN as float — convert properly
                    try:
                        r[k] = '' if pd.isna(val) else str(val)
                    except (TypeError, ValueError):
                        r[k] = str(val)
        return records
    except Exception as e:
        print(f"Error reading config file: {e}")
        return []


def save_config_file(config_path, config_data):
    pd.DataFrame(config_data).to_csv(config_path, index=False)


# ─── Condition evaluation ─────────────────────────────────────────────────────

def evaluate_condition(definition, full_context):
    field = definition.get('ConditionField', '').strip()
    if not field:
        return True
    actual = str(full_context.get(field, '')).strip()
    op     = definition.get('ConditionOp', '==').strip()
    cval   = definition.get('ConditionValue', '').strip()
    if op == '==':           return actual.lower() == cval.lower()
    if op == '!=':           return actual.lower() != cval.lower()
    if op == 'contains':     return cval.lower() in actual.lower()
    if op == 'not_contains': return cval.lower() not in actual.lower()
    if op == 'is_empty':     return actual == ''
    if op == 'is_not_empty': return actual != ''
    return True


# ─── Core tagger ─────────────────────────────────────────────────────────────

def row_by_row_tagger(session_key, csv_path, config_path, input_columns,
                      output_definitions, project_id=None, mode='text'):
    try:
        df = read_csv_safe(csv_path)
        total_rows = len(df)
        PROGRESS_STATUS[session_key] = {
            "done":        0,
            "total":       total_rows,
            "status":      "running",
            "tagged_file": "",
            "logs_file":   "",
            "start_time":  time.time(),
            "last_update": time.time(),
            "live_logs":   [],
        }

        base, ext = os.path.splitext(csv_path)
        logs_path   = base + "_logs.csv"
        tagged_path = base + "_tagged.csv"

        # For image mode, generated PNGs go in a sibling folder under media/;
        # cells store the path relative to MEDIA_ROOT so results/logs can render them.
        images_dir = base + "_images"
        images_rel = os.path.basename(images_dir)
        if mode == 'image':
            os.makedirs(images_dir, exist_ok=True)

        cache.set(f"tagged_file_{session_key}", tagged_path, timeout=86400)
        cache.set(f"logs_file_{session_key}",   logs_path,   timeout=86400)
        PROGRESS_STATUS[session_key]["tagged_file"] = tagged_path
        PROGRESS_STATUS[session_key]["logs_file"]   = logs_path

        pd.DataFrame(columns=["row_index", "column", "prompt", "best_answer", "explanation"]
                     ).to_csv(logs_path, index=False)

        for definition in output_definitions:
            out_col = definition['OutputColumn']
            if out_col not in df.columns:
                df[out_col] = ""
        df.to_csv(tagged_path, index=False)

        context_cols = [c for c in input_columns if c in df.columns] if input_columns else list(df.columns)

        output_col_names = [d['OutputColumn'] for d in output_definitions]
        system_prompt = (
            f"You are an AI-powered CSV Tagger.\n"
            f"You receive one row at a time from a dataset with {total_rows} rows.\n"
            f"Input fields per row: {', '.join(context_cols)}.\n"
            f"Output fields being generated in order: {', '.join(output_col_names)}.\n"
            f"Some rows include previously generated fields — treat them as facts.\n"
            f"Answer the user-defined task precisely.\n"
            f"Always format your response as:\n"
            f"Best Answer: <your answer>\n"
            f"Explanation: <brief reason>"
        )

        if project_id:
            update_project(project_id, status='running', total_rows=total_rows, session_key=session_key)

        for i in range(total_rows):
            row             = df.loc[i]
            row_context     = {c: row[c] for c in context_cols}
            all_row_context = {c: row[c] for c in df.columns}  # every CSV column
            generated       = {}
            generated_detail = {}

            for definition in output_definitions:
                out_col         = definition['OutputColumn']
                prompt_template = definition['PromptTemplate']

                # full_context: globally selected cols + AI-generated cols (for conditions + default prompt)
                full_context = {**row_context, **generated}
                # all_context: every CSV col + AI-generated cols (for per-tag overrides)
                all_context  = {**all_row_context, **generated}

                # Per-tag column filter: draws from all_context so any CSV column is reachable
                tag_input_str = definition.get('InputColumns', '').strip()
                if tag_input_str:
                    tag_cols = {c.strip() for c in tag_input_str.split(',') if c.strip()}
                    display_context = {k: v for k, v in all_context.items() if k in tag_cols}
                    if not display_context:
                        display_context = full_context  # fallback if nothing matches
                else:
                    display_context = full_context

                rendered_prompt = prompt_template
                for col, val in display_context.items():
                    rendered_prompt = rendered_prompt.replace(f'{{{col}}}', str(val))

                user_prompt = (
                    f"Row {i+1}/{total_rows}:\n"
                    + "\n".join(f"  {k}: {v}" for k, v in display_context.items())
                    + f"\n\nTask: {rendered_prompt}\n\n"
                    f"Best Answer: <your answer>\n"
                    f"Explanation: <brief reason>"
                )

                send_context = definition.get('SendContext', '').strip() == '1'
                cond_field   = definition.get('ConditionField', '').strip()
                if send_context and cond_field and cond_field in generated_detail:
                    d = generated_detail[cond_field]
                    user_prompt += (
                        f"\n\n--- Context from '{cond_field}' (condition column) ---"
                        f"\nPrompt used: {d['prompt']}"
                        f"\nAnswer: {d['best_answer']}"
                        f"\nExplanation: {d['explanation']}"
                        f"\n---"
                    )

                image_url = ''
                if evaluate_condition(definition, all_context):
                    if mode == 'image':
                        best_answer, explanation, image_url = _generate_image_for_tag(
                            definition, rendered_prompt, images_dir, images_rel,
                            i, out_col, session_key, project_id,
                        )
                    else:
                        best_answer, explanation, usage = call_llm_tagging(system_prompt, user_prompt)
                        record_stat(
                            usage['host'], usage['port'], usage['model'],
                            session_key, project_id,
                            usage['prompt_tokens'], usage['completion_tokens'], usage['elapsed_sec'],
                        )
                else:
                    best_answer = definition.get('DefaultValue', '').strip() or 'N/A'
                    explanation = (
                        f"Condition not met — "
                        f"{definition.get('ConditionField','')} "
                        f"{definition.get('ConditionOp','')} "
                        f"'{definition.get('ConditionValue','')}' was false. "
                        f"Default value used."
                    )

                df.at[i, out_col] = best_answer
                generated[out_col] = best_answer
                generated_detail[out_col] = {
                    'prompt':      rendered_prompt,
                    'best_answer': best_answer,
                    'explanation': explanation,
                }

                log_entry = {
                    "row_index":   i,
                    "column":      out_col,
                    "prompt":      rendered_prompt,
                    "best_answer": best_answer,
                    "explanation": explanation,
                }
                pd.DataFrame([log_entry]).to_csv(logs_path, mode='a', header=False, index=False)

                # live_logs carry an extra image_url for the frontend; the CSV
                # log above keeps its fixed 5-column schema.
                live_entry = dict(log_entry)
                live_entry["image_url"] = image_url
                live = PROGRESS_STATUS[session_key]["live_logs"]
                live.append(live_entry)
                if len(live) > 100:
                    live.pop(0)

                # Pause check (after each tag, not just each row)
                was_paused = False
                while PAUSE_FLAGS.get(session_key, False):
                    if not was_paused:
                        PROGRESS_STATUS[session_key]['status'] = 'paused'
                        if project_id:
                            update_project(project_id, status='paused', done_rows=i)
                        was_paused = True
                    time.sleep(0.5)
                if was_paused:
                    if project_id:
                        update_project(project_id, status='running')

            df.to_csv(tagged_path, index=False)
            PROGRESS_STATUS[session_key]["done"]        = i + 1
            PROGRESS_STATUS[session_key]["status"]      = f"Processing row {i + 1}/{total_rows}"
            PROGRESS_STATUS[session_key]["last_update"] = time.time()

        df.to_csv(tagged_path, index=False)
        PROGRESS_STATUS[session_key]["status"]      = "finished"
        PROGRESS_STATUS[session_key]["done"]        = total_rows
        PROGRESS_STATUS[session_key]["last_update"] = time.time()

        if project_id:
            update_project(project_id, status='finished', done_rows=total_rows, total_rows=total_rows)

        print(f"DEBUG: Tagging completed -> {tagged_path}, {logs_path}")

    except Exception as e:
        PROGRESS_STATUS[session_key]["status"]      = f"error: {str(e)}"
        PROGRESS_STATUS[session_key]["last_update"] = time.time()
        print(f"Tagging Error: {e}")
        if project_id:
            update_project(project_id, status='error')
        try:
            if 'df' in locals() and 'tagged_path' in locals():
                df.to_csv(tagged_path, index=False)
        except Exception as save_error:
            print(f"ERROR: Failed to save partial progress: {save_error}")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_session_files(session_key, csv_path=None):
    tagged_file = cache.get(f"tagged_file_{session_key}")
    logs_file   = cache.get(f"logs_file_{session_key}")
    if (not tagged_file or not logs_file) and csv_path:
        base, ext   = os.path.splitext(csv_path)
        tagged_file = base + "_tagged.csv"
        logs_file   = base + "_logs.csv"
    return tagged_file, logs_file


def cleanup_abandoned_sessions(force_cleanup_hours=24):
    current_time     = time.time()
    cleanup_threshold = force_cleanup_hours * 3600
    abandoned = [
        k for k, data in PROGRESS_STATUS.items()
        if (current_time - data.get("start_time", current_time)) > cleanup_threshold
        and data.get("status", "") in ("finished", "Completed")
        or data.get("status", "").startswith("error")
    ]
    for k in abandoned:
        del PROGRESS_STATUS[k]
