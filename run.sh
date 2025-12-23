#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_PROMPTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            SKIP_PROMPTS=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./run.sh [OPTIONS] [RUNTIME_ARGS]"
            echo ""
            echo "Options:"
            echo "  -y, --yes    Skip confirmation prompts"
            echo "  -h, --help   Show this help"
            echo ""
            echo "Examples:"
            echo "  ./run.sh              # Interactive with prompts"
            echo "  ./run.sh -y           # Skip prompts"
            echo "  ./run.sh --duration 1h   # Pass runtime args"
            exit 0
            ;;
        *)
            break
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
python -m object_detection --validate
prompt "Press Enter to continue to planning..."

echo -e "${GREEN}=== Planning ===${NC}"
python -m object_detection --plan
prompt "Press Enter to continue to dry run..."

echo -e "${GREEN}=== Dry Run ===${NC}"
python -m object_detection --dry-run
prompt "Press Enter to start running..."

echo -e "${GREEN}=== Running ===${NC}"
python -m object_detection $@
