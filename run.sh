#!/bin/bash
set -e

source ~/traffic-analysis/venv/bin/activate

echo "=== Validating ==="
python -m object_detection --validate
read -p "Press Enter to continue to planning..."

echo "=== Planning ==="
python -m object_detection --plan
read -p "Press Enter to continue to dry run..."

echo "=== Dry Run ==="
python -m object_detection --dry-run
read -p "Press Enter to start running..."

echo "=== Running ==="
python -m object_detection $1
