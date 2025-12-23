#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Parse flags
YES_MODE=false
while getopts "y" opt; do
    case $opt in
        y) YES_MODE=true ;;
    esac
done
shift $((OPTIND-1))

source ~/traffic-analysis/venv/bin/activate

echo -e "${CYAN}Object Detection System${NC}"
echo ""

# Entry point choice (skip in -y mode)
if [ "$YES_MODE" = false ]; then
    echo "What would you like to do?"
    echo "  1. Run with existing config"
    echo "  2. Build new config"
    read -p "Choice [1]: " ENTRY_CHOICE

    if [ "$ENTRY_CHOICE" = "2" ]; then
        python -m object_detection --build-config
        exit 0
    fi
fi

# Prompt for config file (or use default in -y mode)
if [ "$YES_MODE" = true ]; then
    CONFIG_FILE="config.yaml"
    echo -e "${YELLOW}Using default config: $CONFIG_FILE${NC}"
else
    read -p "Config file [config.yaml]: " CONFIG_INPUT
    CONFIG_FILE="${CONFIG_INPUT:-config.yaml}"
fi
CONFIG_ARGS="-c $CONFIG_FILE"

echo ""
echo -e "${GREEN}=== Validating ===${NC}"
python -m object_detection --validate $CONFIG_ARGS
if [ "$YES_MODE" = false ]; then
    read -p "Press Enter to continue..."
fi

echo ""
echo -e "${GREEN}=== Planning ===${NC}"
python -m object_detection --plan $CONFIG_ARGS
if [ "$YES_MODE" = false ]; then
    read -p "Press Enter to continue..."
fi

echo ""
echo -e "${GREEN}=== Dry Run ===${NC}"
python -m object_detection --dry-run $CONFIG_ARGS
echo ""

# Prompt for run options (or use defaults in -y mode)
if [ "$YES_MODE" = true ]; then
    RUN_ARGS=""
    echo -e "${YELLOW}Using duration from config${NC}"
else
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
fi

echo ""
echo -e "${GREEN}=== Running ===${NC}"
python -m object_detection $CONFIG_ARGS $RUN_ARGS
