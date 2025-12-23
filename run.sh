#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

# Arguments
SKIP_PROMPTS=false
CONFIG_ARGS=""
RUN_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            SKIP_PROMPTS=true
            shift
            ;;
        -c|--config)
            CONFIG_ARGS="-c $2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./run.sh [OPTIONS] [DURATION] [-q]"
            echo ""
            echo "Options:"
            echo "  -y, --yes           Skip confirmation prompts"
            echo "  -c, --config FILE   Config file (all phases)"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Run options (passed to final phase only):"
            echo "  DURATION            Duration in hours (e.g., 1, 0.5)"
            echo "  -q, --quiet         Quiet mode"
            echo ""
            echo "Examples:"
            echo "  ./run.sh                           # Default config, interactive"
            echo "  ./run.sh -c traffic.yaml           # Specific config"
            echo "  ./run.sh -c traffic.yaml 1         # 1 hour run"
            echo "  ./run.sh -y -c traffic.yaml 0.5 -q # No prompts, 30min, quiet"
            exit 0
            ;;
        *)
            # Everything else goes to run phase only
            RUN_ARGS="$RUN_ARGS $1"
            shift
            ;;
    esac
done

prompt() {
    if [ "$SKIP_PROMPTS" = false ]; then
        read -p "$1"
    fi
}

source ~/traffic-analysis/venv/bin/activate

echo -e "${GREEN}=== Validating ===${NC}"
python -m object_detection --validate $CONFIG_ARGS
prompt "Press Enter to continue to planning..."

echo -e "${GREEN}=== Planning ===${NC}"
python -m object_detection --plan $CONFIG_ARGS
prompt "Press Enter to continue to dry run..."

echo -e "${GREEN}=== Dry Run ===${NC}"
python -m object_detection --dry-run $CONFIG_ARGS
prompt "Press Enter to start running..."

echo -e "${GREEN}=== Running ===${NC}"
python -m object_detection $CONFIG_ARGS $RUN_ARGS
