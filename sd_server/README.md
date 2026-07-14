# ODT Stable Diffusion Server

A small HTTP server that runs Stable Diffusion models with HuggingFace
`diffusers`, so Athena ODT can generate images per CSV row the same way it
talks to Ollama for text tagging. ODT only ever makes HTTP calls — none of the
heavy GPU dependencies (torch/diffusers) are installed into the Django app.

## Run it

Install the deps **where the GPU is** (your Mac for local, or a LAN GPU box for
remote):

```bash
cd sd_server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CUDA users: install the matching torch build first, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

python app.py --port 7860            # or: uvicorn app:app --host 0.0.0.0 --port 7860
```

Then in ODT open **Image Backend** in the sidebar, point it at this server's
host/port (defaults to `localhost:7860`), and download a model.

- **Local:** run the server on the same machine as ODT and use `localhost`.
- **Remote:** run it on the GPU box, bind `--host 0.0.0.0`, and set ODT's Image
  Backend host to that machine's LAN IP.

## HTTP API

| Method & path | Body / query | Returns |
|---|---|---|
| `GET /health` | — | `{status, loaded_model}` |
| `GET /capability` | — | `{backend, device_name, vram_total_mb, vram_free_mb, ram_mb, torch, diffusers, warning}` |
| `GET /models` | — | `{capability, models:[{id,label,min_vram_mb,gated,downloaded,runnable,default_*}]}` |
| `POST /download` | `{model_id, hf_token?}` | `{job_id}` |
| `GET /download/status` | `?job_id=` | `{state: queued\|downloading\|complete\|error, message}` |
| `POST /generate` | `{model_id, prompt, negative_prompt?, width, height, steps, guidance_scale, seed, num_images, hf_token?}` | `{images:[base64 PNG…], seed_used, elapsed_sec}` |

## Models

`catalog.json` is a curated list (SD 1.5, SDXL, SDXL-Turbo, SD 3.5, FLUX.1)
tagged with approximate VRAM requirements and whether the repo is **gated**.
`/models` marks each entry `runnable` by comparing its `min_vram_mb` against the
detected capability, and `downloaded` if it's already in the local HF cache.
Gated repos (SD 3.5, FLUX.1-dev) need you to accept their licence on Hugging
Face and pass an `hf_token`. Edit `catalog.json` to add more models — any
diffusers text-to-image repo works.

## Notes

- One pipeline is held in memory at a time; switching models frees the previous
  one. Generation is serialised by a lock (single-GPU assumption), which matches
  ODT's sequential row-by-row loop.
- On Apple Silicon the `mps` backend is used and unified memory is reported as
  the VRAM proxy. On a host with no GPU it falls back to CPU (works, but slow).
- Downloaded weights live in the standard Hugging Face cache
  (`~/.cache/huggingface`), shared with any other diffusers/transformers tools.
