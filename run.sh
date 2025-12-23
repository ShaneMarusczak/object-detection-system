#!/bin/bash
set -e

source ~/traffic-analysis/venv/bin/activate

echo "=== Validating ==="
python -m object_detection --validate

echo "=== Planning ==="
python -m object_detection --plan

echo "=== Dry Run ==="
python -m object_detection --dry-run

echo "=== Running ==="
python -m object_detection $1
