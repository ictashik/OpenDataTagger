# Django app (Athena ODT). Talks to Ollama and the SD server (see ../sd_server)
# over plain HTTP — no ML dependencies are installed in this image.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# db.sqlite3 (sessions only) is created on first run if not already present
# on the mounted volume; projects.csv/connections.csv/stats.csv/media/ are
# read/written in place per CLAUDE.md's flat-file architecture.
CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py runserver 0.0.0.0:8000"]
