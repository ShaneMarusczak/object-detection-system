#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

source ~/traffic-analysis/venv/bin/activate

echo -e "${CYAN}Object Detection System${NC}"
echo ""

# Prompt for config file
read -p "Config file [config.yaml]: " CONFIG_INPUT
CONFIG_FILE="${CONFIG_INPUT:-config.yaml}"
CONFIG_ARGS="-c $CONFIG_FILE"

echo ""
echo -e "${GREEN}=== Validating ===${NC}"
python -m object_detection --validate $CONFIG_ARGS
read -p "Press Enter to continue..."

echo ""
echo -e "${GREEN}=== Planning ===${NC}"
python -m object_detection --plan $CONFIG_ARGS
read -p "Press Enter to continue..."

echo ""
echo -e "${GREEN}=== Dry Run ===${NC}"
python -m object_detection --dry-run $CONFIG_ARGS
echo ""

# Prompt for run options
read -p "Duration in hours [from config]: " DURATION
read -p "Quiet mode? (y/N): " QUIET_INPUT

RUN_ARGS=""
if [ -n "$DURATION" ]; then
    RUN_ARGS="$DURATION"
fi
if [[ "$QUIET_INPUT" =~ ^[Yy]$ ]]; then
    RUN_ARGS="$RUN_ARGS -q"
fi

read -p "Press Enter to start running..."

echo ""
echo -e "${GREEN}=== Running ===${NC}"
python -m object_detection $CONFIG_ARGS $RUN_ARGS
