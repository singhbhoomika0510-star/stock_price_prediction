#!/usr/bin/env bash
PORT="${PORT:-10000}"
echo "Starting application on port $PORT..."

if command -v gunicorn >/dev/null 2>&1; then
    echo "Starting via system gunicorn..."
    exec gunicorn --bind "0.0.0.0:$PORT" --workers 1 --threads 4 --timeout 120 app:app
elif [ -f ".venv/bin/gunicorn" ]; then
    echo "Starting via .venv gunicorn..."
    exec .venv/bin/gunicorn --bind "0.0.0.0:$PORT" --workers 1 --threads 4 --timeout 120 app:app
elif [ -f "/opt/render/project/src/.venv/bin/gunicorn" ]; then
    echo "Starting via /opt/render gunicorn..."
    exec /opt/render/project/src/.venv/bin/gunicorn --bind "0.0.0.0:$PORT" --workers 1 --threads 4 --timeout 120 app:app
else
    echo "Starting via python -m gunicorn..."
    exec python -m gunicorn --bind "0.0.0.0:$PORT" --workers 1 --threads 4 --timeout 120 app:app || exec python app.py
fi
