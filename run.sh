#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_PROMPTS=false
CONFIG=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            SKIP_PROMPTS=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./run.sh [OPTIONS] [CONFIG_FILE] [-- EXTRA_ARGS]"
            echo ""
            echo "Options:"
            echo "  -y, --yes    Skip confirmation prompts"
            echo "  -h, --help   Show this help"
            echo ""
            echo "Examples:"
            echo "  ./run.sh                           # Use default config"
            echo "  ./run.sh configs/traffic.yaml      # Use specific config"
            echo "  ./run.sh -y configs/traffic.yaml   # Skip prompts"
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS="$@"
            break
            ;;
        *)
            CONFIG="$1"
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

echo -e "${CYAN}Config: ${NC}${CONFIG:-default}"
echo ""

echo -e "${GREEN}=== Validating ===${NC}"
python -m object_detection --validate $CONFIG
prompt "Press Enter to continue to planning..."

echo -e "${GREEN}=== Planning ===${NC}"
python -m object_detection --plan $CONFIG
prompt "Press Enter to continue to dry run..."

echo -e "${GREEN}=== Dry Run ===${NC}"
python -m object_detection --dry-run $CONFIG
prompt "Press Enter to start running..."

echo -e "${GREEN}=== Running ===${NC}"
python -m object_detection $CONFIG $EXTRA_ARGS
