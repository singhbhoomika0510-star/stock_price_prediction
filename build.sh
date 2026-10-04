#!/usr/bin/env bash
set -e
echo "Starting build process..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo "Build completed successfully!"
