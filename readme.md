# OpenDataTagger

OpenDataTagger (Athena ODT) is an **AI-powered CSV tagging tool** built with Django. It lets users upload CSV files, configure tagging prompts, and process data row-by-row using a locally hosted LLM (via [Ollama](https://ollama.com)). Projects can also run in **image mode**, generating a Stable Diffusion image per row instead of a text tag. The application provides real-time progress updates, detailed logs, and downloadable results — all wrapped in a responsive UI powered by Tailwind CSS.

## Features

- **CSV Upload & Configuration:**
  Upload a CSV file along with an optional configuration file that pre-defines tagging prompts.

- **Custom Column Definition:**
  Select input columns and define output columns with custom AI prompt templates, per-tag conditions, and context chaining.

- **Background Tagging Process:**
  The tagging process runs in a background thread, allowing you to monitor progress in real time.

- **Real-Time Progress & Logging:**
  Get live updates and view detailed logs of AI responses (including the best answer and explanations) for each processed row.

- **Downloadable Results:**
  Once tagging is complete, preview and download both the tagged CSV file and the detailed logs.

- **Local LLM Integration:**
  Uses a locally hosted LLM via Ollama's OpenAI-compatible API. Host/port/model are managed from the **Connection Editor** page and remembered in `connections.csv`.

- **Image Generation Mode:**
  Run a project in **image** mode instead of text tagging — each output column generates a Stable Diffusion image per row via a companion [`sd_server`](sd_server/README.md) process. Manage the connection, browse/download models and LoRAs, and compare models side-by-side from the **Image Backend** page.

- **Responsive UI:**
  Clean and responsive interface built with Tailwind CSS and featuring a sidebar for easy navigation.

## Prerequisites

- **Python 3.8+**
- **Django 5.0** (or later)
- **Pandas**
- **OpenAI Python Library**
- A local [Ollama](https://ollama.com) server (or any OpenAI-compatible endpoint) for text tagging — no fixed host/port is required, it's configured from the app's Connection Editor page.
- *(Optional, only for Image Generation Mode)* the companion `sd_server` process — see [Starting the Servers](#starting-the-servers) below.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/ictashik-opendatatagger.git
   cd ictashik-opendatatagger
   ```

2. **Create a virtual environment & activate it:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r AthensMT/requirements.txt
   ```

   (The repo-root `requirements.txt` is stale — ignore it.)

4. **Apply migrations:**

   ```bash
   python AthensMT/manage.py migrate
   ```

5. **Start the servers** — see [Starting the Servers](#starting-the-servers) below.

6. **Access the application:**
   Open your browser and navigate to `http://localhost:8000/ODT/`.

## Starting the Servers

### Django app

Native:

```bash
python AthensMT/manage.py runserver
```

Docker:

```bash
docker compose -f docker-compose.app.yml up --build
```

### Stable Diffusion image server (optional, for Image Generation Mode)

Only needed if you plan to run a project in **image** mode. Install its dependencies where the GPU lives (your Mac, or a LAN GPU box) — the Django app never installs `torch`/`diffusers`. See [`sd_server/README.md`](sd_server/README.md) for CUDA-specific torch builds.

Native:

```bash
cd sd_server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py --port 7860
```

Docker:

```bash
docker compose -f docker-compose.server.yml up --build
```

> Apple Silicon note: Docker Desktop's Linux VM can't pass through the `mps` GPU, so the containerized server falls back to CPU (slow). Run it natively on Mac instead.

### Both together (Docker)

```bash
docker compose up --build
```

Starts the app and `sd_server` on one Docker network. In the app's **Image Backend** page, set the host to `sd_server` and the port to `7860`.

## Usage

1. **Upload CSV File:**
   - Navigate to the Upload CSV page.
   - Upload your main CSV file and, optionally, a configuration CSV file.
   - Choose the project mode — **Text** (LLM tagging) or **Image** (Stable Diffusion generation).
   - Click Upload & Continue.

2. **Define Columns & Prompts:**
   - On the Define Input & Output Columns page, select the input columns from your CSV.
   - Define one or more output columns and specify corresponding prompt templates.
   - Click Save & Continue.

3. **Tagging / Generation Process:**
   - The process starts in the background.
   - Monitor real-time progress and view detailed logs on the Tagging page.

4. **View & Download Results:**
   - Once processing is complete, navigate to the Results page.
   - Preview the tagged CSV (or generated images) and download the output files.

### Image Generation Mode

1. Start the [SD server](#stable-diffusion-image-server-optional-for-image-generation-mode).
2. Open **Image Backend** in the sidebar and point it at the server's host/port (defaults to `localhost:7860`).
3. Browse the model catalog, download a model (and optionally LoRAs), and set one active.
4. Use **Compare Models** to generate the same prompt across several models side-by-side before committing to one for the full run.
5. Upload a CSV in **Image** mode and define output columns as image-generation prompts, same as text tagging.

## Project Structure

```
OpenDataTagger/
├── docker-compose.yml           # app + sd_server together
├── docker-compose.app.yml       # Django app only
├── docker-compose.server.yml    # sd_server only
├── sd_server/                   # Stable Diffusion HTTP server (see its own README)
│   ├── Dockerfile
│   ├── app.py
│   ├── capability.py
│   ├── downloader.py
│   ├── models.py
│   ├── catalog.json
│   └── lora_catalog.json
└── AthensMT/
    ├── Dockerfile
    ├── requirements.txt
    ├── manage.py
    ├── db.sqlite3                # Django sessions only — no domain models
    ├── projects.csv               # project registry
    ├── connections.csv            # Ollama connection history
    ├── image_connections.csv      # SD server connection history
    ├── stats.csv                  # per-call token/latency log
    ├── media/                     # uploads, tagged output, generated images
    ├── AthensMT/
    │   ├── __init__.py
    │   ├── asgi.py
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    └── tagger_app/
        ├── __init__.py
        ├── admin.py
        ├── apps.py
        ├── forms.py
        ├── models.py
        ├── tests.py
        ├── urls.py
        ├── utils.py
        ├── views.py
        ├── migrations/
        └── templates/
            ├── base.html
            ├── home.html
            ├── upload.html
            ├── define_columns.html
            ├── tagging.html
            ├── results.html
            ├── connection.html
            └── image_backend.html
```

## Configuration

- **Django Settings:**
  All configuration lives in `AthensMT/AthensMT/settings.py`. Update `ALLOWED_HOSTS`, `DEBUG`, `SD_SERVER_DEFAULT`, and other settings as necessary for your deployment environment.

- **Media Files:**
  Uploaded CSVs and generated outputs (tagged CSV, logs, and generated images) are stored under `AthensMT/media/`.

- **LLM Integration:**
  Text tagging talks to Ollama via `tagger_app/utils.py`, using the most-recently-used entry in `connections.csv`. Manage this from the Connection Editor page — there's no env var to set.

- **Image Backend Integration:**
  Image generation talks to `sd_server` via `tagger_app/utils.py`, using the most-recently-used entry in `image_connections.csv` (falls back to `SD_SERVER_DEFAULT` in `settings.py`). Manage this from the Image Backend page.

- **Caching:**
  Django's `LocMemCache` tracks real-time LLM/image-generation usage statistics and tagging progress. It's in-memory only and resets on server restart.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License

This project is open source. Include your license information here if applicable.

## Acknowledgments

- Django
- Tailwind CSS
- OpenAI Python library
- HuggingFace `diffusers`
- Ollama
- Special thanks to all contributors and the open source community.
