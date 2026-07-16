import threading
import openai
import pandas as pd
import os
import re
import shutil
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
CANCEL_FLAGS = {}    # session_key -> bool  (True = stop ASAP, e.g. project was deleted)

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

STYLE_PRESETS_PATH = os.path.join(_base, 'style_presets.json')


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


def get_image_loras(timeout=10):
    return _sd_request('/loras', timeout=timeout)


def get_downloaded_image_loras():
    """List of LoRAs already downloaded — for the per-tag LoRA dropdown."""
    try:
        data = get_image_loras()
        return [l for l in data.get('loras', []) if l.get('downloaded')]
    except Exception as e:
        print(f"Image LoRAs unavailable: {e}")
        return []


def start_image_lora_download(lora_id, hf_token=None, timeout=10):
    return _sd_request('/download', {'model_id': lora_id, 'hf_token': hf_token or None, 'kind': 'lora'},
                       method='POST', timeout=timeout)


def get_image_schedulers(timeout=6):
    """Curated scheduler/sampler options exposed by the SD server — for the
    per-tag scheduler dropdown. Falls back to just 'default' if unreachable."""
    try:
        data = _sd_request('/schedulers', timeout=timeout)
        return data.get('schedulers', [])
    except Exception as e:
        print(f"Image schedulers unavailable: {e}")
        return [{'key': 'default', 'label': 'Default (as shipped)'}]


def get_image_server_health(timeout=4):
    """Live SD server state — which model is actually loaded in memory right
    now. This can lag or differ from what a tag's ImageParams requests (e.g.
    right after switching the active model), so it's surfaced separately
    from the configured model on the tagging page. Returns {} if the server
    is unreachable rather than raising, since this backs a status display."""
    try:
        return _sd_request('/health', timeout=timeout)
    except Exception:
        return {}


def get_image_server_status(timeout=4):
    """Live activity feed from the SD server — current state (loading
    weights / loading LoRA / generating step N/M / idle) plus recent log
    lines, so a stuck or slow /generate call isn't a silent black box on
    the tagging page. Returns {} if the server is unreachable."""
    try:
        return _sd_request('/status', timeout=timeout)
    except Exception:
        return {}


def get_image_server_metrics(timeout=4):
    """Live CPU/RAM/GPU load on the machine running the SD server, for the
    tagging page's system-metrics panel. Returns {} if unreachable — the
    caller renders that as '—' rather than erroring."""
    try:
        return _sd_request('/metrics', timeout=timeout)
    except Exception:
        return {}


def request_image_generation_cancel(timeout=4):
    """Best-effort: ask the SD server to interrupt whatever it's doing right
    now. See sd_server's models.request_cancel for what this can and can't
    stop (an in-flight denoising loop, yes; a from_pretrained() weight load
    already in progress, no — that has no cooperative interrupt point)."""
    try:
        _sd_request('/cancel', {}, method='POST', timeout=timeout)
    except Exception:
        pass


def _human_bytes(n):
    n = float(n)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f"{n:.0f} {unit}" if unit == 'B' else f"{n:.1f} {unit}"
        n /= 1024


def get_disk_usage(path=None):
    """Free/total space on the filesystem backing generated-image storage —
    a long image-mode run can fill a disk, so this is surfaced live on the
    tagging page. Returns None if the path can't be statted."""
    path = path or settings.MEDIA_ROOT
    try:
        total, used, free = shutil.disk_usage(path)
    except OSError:
        return None
    return {
        'free_bytes':   free,
        'total_bytes':  total,
        'free_human':   _human_bytes(free),
        'total_human':  _human_bytes(total),
        'percent_free': round(free / total * 100, 1) if total else 0,
    }


def summarize_image_run_settings(config_data, conn=None):
    """Distinct models + LoRAs declared across an image-mode run's tag
    definitions, for the tagging page's status panel — per-tag ImageParams
    can override the active connection's model, so what actually gets used
    isn't always obvious from the sidebar alone."""
    conn = conn or get_active_image_connection()
    models = set()
    loras = {}
    for d in config_data:
        try:
            params = json.loads(d.get('ImageParams') or '{}')
            if not isinstance(params, dict):
                params = {}
        except (ValueError, TypeError):
            params = {}
        model = (params.get('model') or conn.get('model') or '').strip()
        if model:
            models.add(model)
        for lora in _normalize_loras(params):
            if lora.get('id'):
                loras[lora['id']] = lora.get('scale', 1.0)
    return {
        'models': sorted(models) if models else ([conn['model']] if conn.get('model') else []),
        'loras':  [{'id': k, 'scale': v} for k, v in sorted(loras.items())],
    }


def load_style_presets():
    """Named negative-prompt/sampling presets for the image-mode tag editor.
    Edit style_presets.json to add more — no code change needed."""
    try:
        with open(STYLE_PRESETS_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading style presets: {e}")
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


ASPECT_RATIO_PRESETS = [
    {'key': 'square',         'label': 'Square 1:1',       'width': 512,  'height': 512},
    {'key': 'portrait',       'label': 'Portrait 2:3',     'width': 512,  'height': 768},
    {'key': 'landscape',      'label': 'Landscape 3:2',    'width': 768,  'height': 512},
    {'key': 'portrait_tall',  'label': 'Portrait 9:16',    'width': 576,  'height': 1024},
    {'key': 'landscape_wide', 'label': 'Landscape 16:9',   'width': 1024, 'height': 576},
    {'key': 'wide_xl',        'label': 'Wide 16:9 (XL)',   'width': 1344, 'height': 768},
]


def get_aspect_ratio_presets():
    return ASPECT_RATIO_PRESETS


def _normalize_loras(params):
    """Normalize a tag's LoRA config to a list of {'id','scale'} dicts.
    Supports both the new stacked `loras` list and the older singular
    `lora`/`lora_scale` fields (still present in configs saved before
    multi-LoRA support was added)."""
    loras = params.get('loras')
    if isinstance(loras, list) and loras:
        out = []
        for l in loras:
            if not isinstance(l, dict):
                continue
            lid = (l.get('id') or '').strip()
            if lid:
                out.append({'id': lid, 'scale': _coerce_float(l.get('scale'), 1.0)})
        if out:
            return out
    legacy_id = (params.get('lora') or '').strip()
    if legacy_id:
        return [{'id': legacy_id, 'scale': _coerce_float(params.get('lora_scale'), 1.0)}]
    return []


def call_image_generation(prompt, params):
    """Generate image(s) via the SD server for one rendered prompt.

    params keys (all optional except model): model, negative_prompt, width,
    height, steps, guidance, seed, num_images, hf_token, loras (list of
    {id, scale}), scheduler.

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
        'loras':           _normalize_loras(params),
        'scheduler':       (params.get('scheduler') or 'default').strip() or 'default',
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


def estimate_image_generation(prompt, params):
    """Run ONE real generation with the given params to measure actual
    per-image time on the active hardware/model (extrapolating to a full
    project is the caller's job — this just produces a true sample).

    Returns (elapsed_sec, sample_image_url, error). elapsed_sec/url are None
    on failure, with `error` set.
    """
    images, meta, usage = call_image_generation(prompt, params)
    if not images:
        return None, None, meta.get('error', 'unknown error')

    scratch_dir = os.path.join(settings.MEDIA_ROOT, '_estimates')
    os.makedirs(scratch_dir, exist_ok=True)
    fname = f"estimate_{int(time.time() * 1000)}.png"
    with open(os.path.join(scratch_dir, fname), 'wb') as fh:
        fh.write(images[0])
    return usage['elapsed_sec'], settings.MEDIA_URL + '_estimates/' + fname, None


def _safe_col_name(out_col):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(out_col)).strip('_') or 'img'


def _safe_filename_value(value, fallback):
    """Sanitize a cell value for use as (part of) a filename, falling back
    to `fallback` (typically 'row{N}') when the value is blank/NaN or
    sanitizes away to nothing."""
    if value is None:
        return fallback
    s = str(value).strip()
    if not s or s.lower() == 'nan':
        return fallback
    s = re.sub(r'[^A-Za-z0-9_-]+', '_', s).strip('_')
    return s[:80] or fallback


def _convert_image_bytes(png_bytes, ext):
    """The SD server always returns PNG bytes; convert to JPEG here if the
    project's chosen output format asks for it. JPEG has no alpha channel,
    so transparency is flattened onto white first."""
    if ext not in ('jpg', 'jpeg'):
        return png_bytes
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(png_bytes))
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGBA')
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=92)
    return buf.getvalue()


def _image_base_name(row_data, naming_column, row_index):
    """User-facing base filename for a row's generated image(s) — the
    naming column's value when configured and present, else 'row{N}'."""
    if naming_column and row_data and naming_column in row_data:
        return _safe_filename_value(row_data.get(naming_column), f"row{row_index}")
    return f"row{row_index}"


def images_dir_for_tagged_path(tagged_path):
    """Derive the sibling `<name>_images/` folder + its MEDIA_ROOT-relative
    path from a `<name>_tagged.csv` path."""
    base = tagged_path[:-len('_tagged.csv')] if tagged_path.endswith('_tagged.csv') else os.path.splitext(tagged_path)[0]
    images_dir = base + '_images'
    images_rel = os.path.relpath(images_dir, settings.MEDIA_ROOT).replace(os.sep, '/')
    return images_dir, images_rel


# ─── Image manifest (attempt tracking + candidate lookup) ───────────────────
# A JSON sidecar next to the tagged CSV recording every image ever generated
# for a given (row, output column), independent of what the file is actually
# named on disk. This lets filenames stay clean (e.g. "SKU123.jpg" instead of
# "row0_image_a0_0.png") while still supporting retries/candidates: the
# manifest — not filename parsing — is the source of truth for attempt
# numbers and the Results-page grid picker.

_manifest_lock = threading.Lock()


def _manifest_path_for_tagged(tagged_path):
    base = tagged_path[:-len('_tagged.csv')] if tagged_path.endswith('_tagged.csv') else os.path.splitext(tagged_path)[0]
    return base + '_image_manifest.json'


def _load_manifest(tagged_path):
    try:
        with open(_manifest_path_for_tagged(tagged_path)) as f:
            return json.load(f)
    except Exception:
        return {}


def _next_attempt_index(tagged_path, row_index, out_col):
    """Each generation (initial run or retry) for a row/tag gets its own
    'attempt' number so retries add new candidate images instead of
    overwriting earlier ones — that's what makes the grid picker and
    regenerate-history work."""
    entries = _load_manifest(tagged_path).get(f"{row_index}:{out_col}", [])
    return (max((e['attempt'] for e in entries), default=-1) + 1)


def _save_generated_images(images_dir, images_rel, tagged_path, row_index, out_col,
                           attempt, images, base_name, ext):
    """Write generated image bytes to disk with a human-readable filename and
    record each one in the manifest. `base_name` should already be sanitized
    (see _image_base_name); a per-file existence check disambiguates the rare
    case where two rows sanitize to the same base name, so nothing on disk is
    ever silently overwritten."""
    saved_rel = []
    with _manifest_lock:
        manifest = _load_manifest(tagged_path)
        key = f"{row_index}:{out_col}"
        entries = manifest.setdefault(key, [])
        for n, img_bytes in enumerate(images):
            suffix = (f"_a{attempt}" if attempt > 0 else "") + (f"_{n}" if n > 0 else "")
            fname = f"{base_name}{suffix}.{ext}"
            path = os.path.join(images_dir, fname)
            if os.path.exists(path):
                fname = f"{base_name}_row{row_index}{suffix}.{ext}"
                path = os.path.join(images_dir, fname)
            with open(path, 'wb') as fh:
                fh.write(_convert_image_bytes(img_bytes, ext))
            rel = f"{images_rel}/{fname}"
            saved_rel.append(rel)
            entries.append({'rel_path': rel, 'attempt': attempt, 'n': n})
        with open(_manifest_path_for_tagged(tagged_path), 'w') as f:
            json.dump(manifest, f)
    return saved_rel


def list_image_candidates(tagged_path, row_index, out_col):
    """Every previously generated candidate image for one row/tag — the
    initial generation plus any retries — for the Results-page grid picker.
    Newest attempt first."""
    entries = _load_manifest(tagged_path).get(f"{row_index}:{out_col}", [])
    candidates = [{
        'rel_path': e['rel_path'],
        'url':      settings.MEDIA_URL + e['rel_path'],
        'attempt':  e['attempt'],
        'n':        e.get('n', 0),
    } for e in entries]
    candidates.sort(key=lambda c: (-c['attempt'], c['n']))
    return candidates


def select_image_candidate(tagged_path, row_index, out_col, rel_path):
    """Point a tagged-CSV image cell at a different already-generated candidate."""
    df = pd.read_csv(tagged_path)
    if row_index < 0 or row_index >= len(df) or out_col not in df.columns:
        raise ValueError("Invalid row index or column.")
    df.at[row_index, out_col] = rel_path
    df.to_csv(tagged_path, index=False)


# ─── Review state (approve/reject) ──────────────────────────────────────────
# A lightweight sidecar JSON next to the tagged CSV — keeps the approve/reject
# mark out of the CSV itself so it doesn't interfere with downstream consumers
# of the tagged data.

_review_lock = threading.Lock()


def _review_path_for_tagged(tagged_path):
    base = tagged_path[:-len('_tagged.csv')] if tagged_path.endswith('_tagged.csv') else os.path.splitext(tagged_path)[0]
    return base + '_review.json'


def load_review_state(tagged_path):
    """{'<row_index>:<column>': 'approved'|'rejected'}"""
    try:
        with open(_review_path_for_tagged(tagged_path)) as f:
            return json.load(f)
    except Exception:
        return {}


def set_review_state(tagged_path, row_index, column, status):
    """status: 'approved' | 'rejected' | '' (clears the mark)."""
    key = f"{row_index}:{column}"
    with _review_lock:
        state = load_review_state(tagged_path)
        if status:
            state[key] = status
        else:
            state.pop(key, None)
        with open(_review_path_for_tagged(tagged_path), 'w') as f:
            json.dump(state, f)
    return state


# ─── Seed history (locked-seed retry) ───────────────────────────────────────
# {'<rel_path>': seed} sidecar next to the tagged CSV. Every generated image
# is recorded here, so a later "retry with the same seed" can look up the
# seed behind whichever candidate is currently shown in a cell (including one
# picked via the grid, not just the latest attempt) and reuse it — letting
# you tweak the prompt/negative-prompt/LoRA and see the effect on the same
# composition instead of a fresh random one.

_seeds_lock = threading.Lock()


def _seeds_path_for_tagged(tagged_path):
    base = tagged_path[:-len('_tagged.csv')] if tagged_path.endswith('_tagged.csv') else os.path.splitext(tagged_path)[0]
    return base + '_seeds.json'


def load_seed_state(tagged_path):
    try:
        with open(_seeds_path_for_tagged(tagged_path)) as f:
            return json.load(f)
    except Exception:
        return {}


def record_seeds(tagged_path, rel_paths, seed):
    if seed is None or not rel_paths:
        return
    with _seeds_lock:
        state = load_seed_state(tagged_path)
        for rel in rel_paths:
            state[rel] = seed
        with open(_seeds_path_for_tagged(tagged_path), 'w') as f:
            json.dump(state, f)


def get_seed_for_path(tagged_path, rel_path):
    if not rel_path:
        return None
    return load_seed_state(tagged_path).get(str(rel_path))


# ─── Bulk retry ──────────────────────────────────────────────────────────────
# Mirrors the PROGRESS_STATUS pattern used by row_by_row_tagger: a background
# daemon thread writes into a module-level dict keyed by job id, polled over
# HTTP from the Results page.

BULK_RETRY_STATUS = {}  # job_key -> dict(status, done, total, fixed, failed, message)
_bulk_retry_lock = threading.Lock()


def _run_bulk_retry(job_key, groups, lock_seed=False):
    """Shared retry loop for both 'retry all failed' (one project, error
    cells only) and the gallery's 'retry selected' (any cells, possibly
    spanning several projects) — one at a time, since the SD server
    serializes generation on its end anyway so there's no benefit to
    parallelizing here.

    groups: [{'tagged_path', 'config_data', 'images_dir', 'images_rel',
    'session_key', 'project_id', 'targets': [(row_index, col), ...]}, ...]
    """
    total = sum(len(g['targets']) for g in groups)
    with _bulk_retry_lock:
        BULK_RETRY_STATUS[job_key] = {
            'status': 'running', 'done': 0, 'total': total,
            'fixed': 0, 'failed': 0, 'message': '',
        }

    for g in groups:
        for row_index, col in g['targets']:
            try:
                regenerate_image_cell(
                    g['tagged_path'], g['config_data'], row_index, col,
                    g['images_dir'], g['images_rel'],
                    session_key=g.get('session_key'), project_id=g.get('project_id'),
                    lock_seed=lock_seed,
                )
                with _bulk_retry_lock:
                    BULK_RETRY_STATUS[job_key]['fixed'] += 1
            except Exception as e:
                with _bulk_retry_lock:
                    BULK_RETRY_STATUS[job_key]['failed'] += 1
                    BULK_RETRY_STATUS[job_key]['message'] = str(e)
            with _bulk_retry_lock:
                BULK_RETRY_STATUS[job_key]['done'] += 1

    with _bulk_retry_lock:
        BULK_RETRY_STATUS[job_key]['status'] = 'finished'


def bulk_retry_errors(job_key, tagged_path, config_data, images_dir, images_rel,
                      session_key=None, project_id=None):
    """Find every image cell currently holding an 'ERROR: ...' value and
    retry it (fresh seed, same settings)."""
    image_cols = {d['OutputColumn'] for d in config_data if (d.get('ImageParams') or '').strip()}
    df = pd.read_csv(tagged_path)
    targets = [
        (int(row_index), col)
        for row_index, row in df.iterrows()
        for col in image_cols
        if col in df.columns and str(row[col]).startswith('ERROR:')
    ]
    _run_bulk_retry(job_key, [{
        'tagged_path': tagged_path, 'config_data': config_data,
        'images_dir': images_dir, 'images_rel': images_rel,
        'session_key': session_key, 'project_id': project_id,
        'targets': targets,
    }])


def bulk_retry_selected(job_key, groups, lock_seed=False):
    """Retry an explicit list of (row_index, column) cells, grouped by
    project — the Gallery's multi-select retry (as opposed to
    bulk_retry_errors, which scans one project for ERROR cells)."""
    _run_bulk_retry(job_key, groups, lock_seed=lock_seed)


# ─── Gallery (cross-project + per-project image browsing) ──────────────────
# Unlike the Results page, the Gallery has no single Django session to lean
# on — the cross-project view spans every image-mode project at once, and a
# per-project link from Home shouldn't disturb whatever session/project the
# user currently has open. Everything here is resolved straight from a
# `projects.csv` row instead.

def tagged_path_for_project(project):
    csv_path = project.get('csv_path', '') or ''
    if not csv_path:
        return ''
    base, _ext = os.path.splitext(csv_path)
    return base + '_tagged.csv'


def build_gallery_items(project):
    """Every generated-image cell for one project, flattened into gallery
    cards. Returns (items, tagged_path, images_dir, images_rel); the latter
    three are '' / None / '' when the project has no tagged output yet."""
    tagged_path = tagged_path_for_project(project)
    if not tagged_path or not os.path.exists(tagged_path):
        return [], '', None, ''

    config_data = load_config_file(project.get('config_path', ''))
    image_cols = [d['OutputColumn'] for d in config_data if (d.get('ImageParams') or '').strip()]
    if not image_cols:
        return [], tagged_path, None, ''

    images_dir, images_rel = images_dir_for_tagged_path(tagged_path)
    df = read_csv_safe(tagged_path)
    review_state = load_review_state(tagged_path)
    naming_column = project.get('image_naming_column', '') or ''
    image_exts = ('.png', '.jpg', '.jpeg', '.webp')

    items = []
    for row_index, row in df.iterrows():
        for col in image_cols:
            if col not in df.columns:
                continue
            cell = row[col]
            sval = '' if (isinstance(cell, float) and pd.isna(cell)) else str(cell)
            if not sval:
                continue
            is_err = sval.startswith('ERROR:')
            is_img = sval.lower().endswith(image_exts)
            if not (is_err or is_img):
                continue

            label = ''
            if naming_column and naming_column in df.columns:
                nv = row[naming_column]
                label = '' if (isinstance(nv, float) and pd.isna(nv)) else str(nv)
            if not label:
                label = f"row{row_index}"

            items.append({
                'project_id':      project['project_id'],
                'project_name':    project.get('name', ''),
                'row_index':       int(row_index),
                'column':          col,
                'value':           sval,
                'is_image':        is_img,
                'is_error':        is_err,
                'url':             (settings.MEDIA_URL + sval) if is_img else '',
                'candidates_json': json.dumps(list_image_candidates(tagged_path, row_index, col)),
                'review':          review_state.get(f"{row_index}:{col}", '') if is_img else '',
                'label':           label,
            })
    return items, tagged_path, images_dir, images_rel


def build_gallery_zip(entries, include_csv=True):
    """entries: [(project, rel_paths_or_None), ...] — rel_paths_or_None is a
    list of MEDIA_ROOT-relative image paths to zip, or None for every file
    under that project's `<name>_images/` folder. Images are namespaced per
    project when more than one project is included. Returns raw zip bytes."""
    import io
    import zipfile

    buf = io.BytesIO()
    multi = len(entries) > 1
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for project, rel_paths in entries:
            tagged_path = tagged_path_for_project(project)
            if not tagged_path:
                continue
            images_dir, images_rel = images_dir_for_tagged_path(tagged_path)
            prefix = f"{_safe_col_name(project.get('name') or project['project_id'])}/" if multi else ''

            if rel_paths is None:
                if os.path.isdir(images_dir):
                    for fname in sorted(os.listdir(images_dir)):
                        fpath = os.path.join(images_dir, fname)
                        if os.path.isfile(fpath):
                            zf.write(fpath, f"{prefix}images/{fname}")
            else:
                media_root = os.path.normpath(settings.MEDIA_ROOT)
                for rel in rel_paths:
                    rel = str(rel).replace('\\', '/').lstrip('/')
                    # Only ever pull files out of *this* project's own images
                    # folder — rel paths arrive from the client (POST body),
                    # so this stops a crafted path from reaching outside it.
                    if not (rel == images_rel or rel.startswith(images_rel + '/')):
                        continue
                    fpath = os.path.normpath(os.path.join(settings.MEDIA_ROOT, rel))
                    if not fpath.startswith(media_root) or not os.path.isfile(fpath):
                        continue
                    zf.write(fpath, f"{prefix}images/{os.path.basename(rel)}")

            if include_csv and os.path.isfile(tagged_path):
                zf.write(tagged_path, f"{prefix}tagged.csv")

    return buf.getvalue()


# ─── Compare models ──────────────────────────────────────────────────────────

def compare_models_generate(prompt, params, model_ids):
    """Generate one image per model for the same prompt/params — powers the
    Compare Models page. Images are written to a scratch media folder (not
    tied to any project) since this is exploratory, not part of a tagging run.

    Returns a list of {model, image_url, elapsed_sec, seed_used, error}.
    """
    scratch_dir = os.path.join(settings.MEDIA_ROOT, '_compare')
    os.makedirs(scratch_dir, exist_ok=True)

    results = []
    for model_id in model_ids:
        images, meta, usage = call_image_generation(prompt, {**params, 'model': model_id})
        if not images:
            results.append({'model': model_id, 'image_url': None, 'elapsed_sec': None,
                            'seed_used': None, 'error': meta.get('error', 'unknown error')})
            continue
        fname = f"compare_{int(time.time() * 1000)}_{_safe_col_name(model_id)}.png"
        with open(os.path.join(scratch_dir, fname), 'wb') as fh:
            fh.write(images[0])
        results.append({
            'model':      model_id,
            'image_url':  settings.MEDIA_URL + '_compare/' + fname,
            'elapsed_sec': round(usage['elapsed_sec'], 2),
            'seed_used':  meta.get('seed_used'),
            'error':      None,
        })
    return results


def render_tag_prompt(definition, full_context, all_context):
    """Render one tag's PromptTemplate given the row's context.

    full_context: globally-selected columns (+ generated so far).
    all_context: every column (+ generated so far) — per-tag InputColumns can
    reach any of these even if not globally selected.

    Shared by the main tagging loop, single-row retry, and the cost/time
    estimator so all three render a prompt identically.
    """
    prompt_template = definition['PromptTemplate']
    tag_input_str = definition.get('InputColumns', '').strip()
    if tag_input_str:
        tag_cols = {c.strip() for c in tag_input_str.split(',') if c.strip()}
        display_context = {k: v for k, v in all_context.items() if k in tag_cols}
        if not display_context:
            display_context = full_context
    else:
        display_context = full_context

    rendered_prompt = prompt_template
    for col, val in display_context.items():
        rendered_prompt = rendered_prompt.replace(f'{{{col}}}', str(val))
    return rendered_prompt, display_context


def regenerate_image_cell(tagged_path, config_data, row_index, out_col, images_dir, images_rel,
                          session_key=None, project_id=None, param_overrides=None, lock_seed=False):
    """Re-run image generation for one row/tag ('Retry') using the row's
    current (already-tagged) values as context.

    By default uses a fresh random seed. If lock_seed=True (and no explicit
    seed override), reuses the seed recorded for whichever candidate is
    currently shown in this cell — so you can tweak the prompt/negative
    prompt/LoRA/etc. in Define Columns and see the effect on the same
    composition instead of a new random one. Returns (new_relative_paths, seed_used).
    """
    definition = next((d for d in config_data if d['OutputColumn'] == out_col), None)
    if not definition:
        raise ValueError(f"No such output column: {out_col}")

    df = pd.read_csv(tagged_path)
    if row_index < 0 or row_index >= len(df):
        raise ValueError(f"Row {row_index} out of range.")
    row_context = {c: df.loc[row_index, c] for c in df.columns}  # already holds final generated state
    rendered_prompt, _ = render_tag_prompt(definition, row_context, row_context)

    try:
        params = json.loads(definition.get('ImageParams') or '{}')
        if not isinstance(params, dict):
            params = {}
    except (ValueError, TypeError):
        params = {}
    params = {**params, **(param_overrides or {})}
    if param_overrides and 'seed' in param_overrides:
        pass  # explicit caller override wins
    elif lock_seed:
        locked_seed = get_seed_for_path(tagged_path, row_context.get(out_col))
        params['seed'] = locked_seed if locked_seed is not None else -1
    else:
        params['seed'] = -1  # fresh randomness on a plain retry

    images, meta, usage = call_image_generation(rendered_prompt, params)
    record_stat(
        usage['host'], usage['port'], usage['model'],
        session_key, project_id,
        usage['prompt_tokens'], usage['completion_tokens'], usage['elapsed_sec'],
    )
    if not images:
        raise RuntimeError(meta.get('error', 'unknown error'))

    naming_column, image_format = _project_image_settings(project_id)
    base_name = _image_base_name(row_context, naming_column, row_index)
    attempt   = _next_attempt_index(tagged_path, row_index, out_col)
    saved_rel = _save_generated_images(images_dir, images_rel, tagged_path, row_index, out_col,
                                       attempt, images, base_name, image_format)
    record_seeds(tagged_path, saved_rel, meta.get('seed_used'))

    df.at[row_index, out_col] = saved_rel[0]
    df.to_csv(tagged_path, index=False)
    return saved_rel, meta.get('seed_used')


def _generate_image_for_tag(definition, rendered_prompt, images_dir, images_rel,
                            row_index, out_col, session_key, project_id, tagged_path,
                            row_data=None, naming_column='', image_format='png'):
    """Run one image generation for a tag, save the image(s), record a stat.

    Returns (cell_value, explanation, image_url, all_relative_paths, gen_meta):
    cell_value is the MEDIA_ROOT-relative path of the first image written
    into the tagged CSV (or an "ERROR: …" string on failure); image_url is
    that path prefixed with MEDIA_URL for the live-log thumbnail ('' on
    failure); all_relative_paths lists every image from this call (for
    num_images > 1 / the grid picker); gen_meta carries the generation
    parameters/timing for the live-log detail line ('error' key set on
    failure instead).
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

    model = params.get('model') or usage.get('model', '')

    if not images:
        err = meta.get('error', 'unknown error')
        gen_meta = {'model': model, 'elapsed_sec': usage.get('elapsed_sec'), 'error': err}
        return f"ERROR: {err}", f"Image generation failed: {err}", '', [], gen_meta

    base_name = _image_base_name(row_data, naming_column, row_index)
    attempt   = _next_attempt_index(tagged_path, row_index, out_col)
    saved_rel = _save_generated_images(images_dir, images_rel, tagged_path, row_index, out_col,
                                       attempt, images, base_name, image_format)
    record_seeds(tagged_path, saved_rel, meta.get('seed_used'))

    cell_value = saved_rel[0]
    image_url  = settings.MEDIA_URL + saved_rel[0]
    seed  = meta.get('seed_used')
    explanation = f"Generated with {model}" + (f" (seed {seed})" if seed is not None else "")
    gen_meta = {
        'model':        model,
        'loras':        _normalize_loras(params),
        'seed':         seed,
        'width':        _coerce_int(params.get('width'), 512),
        'height':       _coerce_int(params.get('height'), 512),
        'steps':        _coerce_int(params.get('steps'), 30),
        'guidance':     _coerce_float(params.get('guidance'), 7.5),
        'scheduler':    (params.get('scheduler') or 'default').strip() or 'default',
        'num_images':   len(saved_rel),
        'attempt':      attempt,
        'elapsed_sec':  usage.get('elapsed_sec'),
    }
    return cell_value, explanation, image_url, saved_rel, gen_meta


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
        # Per-project image settings — older registries predate these columns.
        if 'image_naming_column' in df.columns:
            df['image_naming_column'] = df['image_naming_column'].fillna('').astype(str)
        else:
            df['image_naming_column'] = ''
        if 'image_format' in df.columns:
            df['image_format'] = df['image_format'].fillna('png').replace('', 'png').astype(str)
        else:
            df['image_format'] = 'png'
        return df.sort_values('last_updated', ascending=False).to_dict('records')
    except Exception as e:
        print(f"Error loading projects: {e}")
        return []


def get_project(project_id):
    return next((p for p in load_projects() if str(p['project_id']) == str(project_id)), None)


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
                'image_naming_column': '',
                'image_format': 'png',
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
        target = next((p for p in projects if str(p['project_id']) == str(project_id)), None)
        projects = [p for p in projects if str(p['project_id']) != str(project_id)]
        pd.DataFrame(projects).to_csv(path, index=False)

    # A running/paused tagging thread for this project may still be alive
    # (e.g. the user paused it, then deleted the project without stopping
    # it first). Signal it to stop and unblock the pause wait-loop so it
    # exits on its own instead of looping forever against files we're
    # about to delete.
    session_key = target.get('session_key') if target else None
    if session_key and session_key in PROGRESS_STATUS:
        CANCEL_FLAGS[session_key] = True
        PAUSE_FLAGS[session_key] = False

    # Projects created after the per-project-folder change store all their
    # files under media/<project_id>/ — safe to remove outright. Older
    # projects still point at flat media/<name>... paths shared by naming
    # convention only, so leave those files on disk (metadata-only delete).
    if target and target.get('csv_path'):
        project_dir = os.path.dirname(os.path.abspath(target['csv_path']))
        if os.path.basename(project_dir) == str(project_id):
            shutil.rmtree(project_dir, ignore_errors=True)


def _project_image_settings(project_id):
    """(naming_column, format) for a project's generated-image filenames.
    Falls back to row-index naming / PNG when unset or the project is
    missing (e.g. exploratory calls with no project_id)."""
    if not project_id:
        return '', 'png'
    proj = get_project(project_id)
    if not proj:
        return '', 'png'
    fmt = (proj.get('image_format') or 'png').lower()
    if fmt not in ('png', 'jpg', 'jpeg'):
        fmt = 'png'
    return proj.get('image_naming_column', '') or '', fmt


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
            "project_id":  project_id,
        }

        base, ext = os.path.splitext(csv_path)
        logs_path   = base + "_logs.csv"
        tagged_path = base + "_tagged.csv"

        # For image mode, generated images go in a sibling folder under media/;
        # cells store the path relative to MEDIA_ROOT (not just the folder's
        # basename) so results/logs still render correctly now that uploads
        # live under per-project subfolders rather than flat in media/.
        images_dir = base + "_images"
        images_rel = os.path.relpath(images_dir, settings.MEDIA_ROOT).replace(os.sep, '/')
        naming_column, image_format = _project_image_settings(project_id)
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
            if CANCEL_FLAGS.get(session_key, False):
                break
            row             = df.loc[i]
            row_context     = {c: row[c] for c in context_cols}
            all_row_context = {c: row[c] for c in df.columns}  # every CSV column
            generated       = {}
            generated_detail = {}

            for definition in output_definitions:
                out_col = definition['OutputColumn']

                # full_context: globally selected cols + AI-generated cols (for conditions + default prompt)
                full_context = {**row_context, **generated}
                # all_context: every CSV col + AI-generated cols (for per-tag overrides)
                all_context  = {**all_row_context, **generated}

                # Per-tag column filter: draws from all_context so any CSV column is reachable
                rendered_prompt, display_context = render_tag_prompt(definition, full_context, all_context)

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
                image_urls = []
                image_meta = None
                if evaluate_condition(definition, all_context):
                    if mode == 'image':
                        best_answer, explanation, image_url, all_paths, image_meta = _generate_image_for_tag(
                            definition, rendered_prompt, images_dir, images_rel,
                            i, out_col, session_key, project_id, tagged_path,
                            row_data=all_context, naming_column=naming_column, image_format=image_format,
                        )
                        image_urls = [settings.MEDIA_URL + p for p in all_paths]
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

                # live_logs carry extra image_url/image_urls for the frontend
                # (image_urls has every candidate when num_images > 1, for the
                # grid thumbnail strip); the CSV log above keeps its fixed
                # 5-column schema.
                live_entry = dict(log_entry)
                live_entry["image_url"] = image_url
                live_entry["image_urls"] = image_urls
                live_entry["image_meta"] = image_meta
                live = PROGRESS_STATUS[session_key]["live_logs"]
                live.append(live_entry)
                if len(live) > 100:
                    live.pop(0)

                # Pause check (after each tag, not just each row). Also bails
                # out immediately if the project was deleted out from under
                # this (possibly paused) thread — CANCEL_FLAGS is what wakes
                # it up in that case, since PAUSE_FLAGS alone would leave it
                # sleeping forever.
                was_paused = False
                while PAUSE_FLAGS.get(session_key, False) and not CANCEL_FLAGS.get(session_key, False):
                    if not was_paused:
                        PROGRESS_STATUS[session_key]['status'] = 'paused'
                        if project_id:
                            update_project(project_id, status='paused', done_rows=i)
                        was_paused = True
                    time.sleep(0.5)
                if was_paused and not CANCEL_FLAGS.get(session_key, False):
                    if project_id:
                        update_project(project_id, status='running')

                if CANCEL_FLAGS.get(session_key, False):
                    break

            if CANCEL_FLAGS.get(session_key, False):
                break

            df.to_csv(tagged_path, index=False)
            PROGRESS_STATUS[session_key]["done"]        = i + 1
            PROGRESS_STATUS[session_key]["status"]      = f"Processing row {i + 1}/{total_rows}"
            PROGRESS_STATUS[session_key]["last_update"] = time.time()

        if CANCEL_FLAGS.pop(session_key, False):
            # Either the project was deleted (its files are gone — don't
            # touch disk again) or the user hit Stop (project still exists,
            # but every completed row up to here was already flushed to
            # tagged_path by the per-row save above, so there's nothing left
            # to write). Either way, just record where it stopped.
            PAUSE_FLAGS.pop(session_key, None)
            PROGRESS_STATUS[session_key]["status"]      = "cancelled"
            PROGRESS_STATUS[session_key]["last_update"] = time.time()
            if project_id:
                update_project(project_id, status='cancelled', done_rows=PROGRESS_STATUS[session_key]["done"])
            return

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
