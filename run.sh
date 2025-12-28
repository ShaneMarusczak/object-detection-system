#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
GRAY='\033[0;90m'
NC='\033[0m'

# Parse flags
# -y = auto mode (skip prompts, use defaults)
# -s = skip menu (go straight to run, for use after builder)
# -b = background mode (run in tmux, detach immediately)
YES_MODE=false
SKIP_MENU=false
BACKGROUND_MODE=false
while getopts "ysb" opt; do
    case $opt in
        y) YES_MODE=true ;;
        s) SKIP_MENU=true ;;
        b) BACKGROUND_MODE=true; YES_MODE=true ;;
    esac
done
shift $((OPTIND-1))

# Activate virtual environment
# Use .run.local for machine-specific venv (e.g., Jetson with custom torch/numpy)
if [ -f ".run.local" ]; then
    source .run.local
elif [ -n "$VIRTUAL_ENV" ]; then
    : # Already in a venv
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo -e "${YELLOW}Warning: No virtual environment found${NC}"
    echo "Create .run.local with: source /path/to/your/venv/bin/activate"
fi

echo -e "${CYAN}Object Detection System${NC}"
echo ""

# Show current config before menu
if [ -f "config.yaml" ]; then
    CURRENT_CONFIG=$(grep -E "^use:" "config.yaml" 2>/dev/null | sed 's/use:[[:space:]]*//' || true)
    if [ -n "$CURRENT_CONFIG" ]; then
        echo -e "Current config: ${YELLOW}$CURRENT_CONFIG${NC}"
    else
        echo -e "Current config: ${YELLOW}config.yaml (inline)${NC}"
    fi
    echo ""
fi

# Entry point choice (skip if -s or -y flag)
if [ "$SKIP_MENU" = false ] && [ "$YES_MODE" = false ]; then
    echo "What would you like to do?"
    echo "  1. Run"
    echo "  2. Run in background (tmux)"
    echo "  3. Pick a config"
    echo "  4. Build new config"
    echo "  5. Edit a config"
    read -p "Choice [1]: " ENTRY_CHOICE

    if [ "$ENTRY_CHOICE" = "2" ]; then
        BACKGROUND_MODE=true
    elif [ "$ENTRY_CHOICE" = "4" ]; then
        python -m object_detection --build-config
        exit 0
    elif [ "$ENTRY_CHOICE" = "5" ]; then
        # List configs in configs/ folder for editing
        echo ""
        echo "Select config to edit:"
        configs=(configs/*.yaml configs/*.yml)
        valid_configs=()
        for cfg in "${configs[@]}"; do
            [ -f "$cfg" ] && valid_configs+=("$cfg")
        done

        if [ ${#valid_configs[@]} -eq 0 ]; then
            echo -e "  ${YELLOW}No configs found in configs/${NC}"
            exit 1
        fi

        i=1
        for cfg in "${valid_configs[@]}"; do
            echo "  $i. $cfg"
            ((i++))
        done

        read -p "Choice [1]: " CONFIG_CHOICE
        CONFIG_CHOICE=${CONFIG_CHOICE:-1}
        idx=$((CONFIG_CHOICE - 1))

        if [ $idx -lt 0 ] || [ $idx -ge ${#valid_configs[@]} ]; then
            echo -e "${YELLOW}Invalid choice${NC}"
            exit 1
        fi

        SELECTED_CONFIG="${valid_configs[$idx]}"
        echo ""
        python -c "from object_detection.config import run_editor; run_editor('$SELECTED_CONFIG')"
        exit 0
    elif [ "$ENTRY_CHOICE" = "3" ]; then
        # List configs in configs/ folder
        echo ""
        echo "Available configs:"
        configs=(configs/*.yaml configs/*.yml)
        # Filter out non-existent globs
        valid_configs=()
        for cfg in "${configs[@]}"; do
            [ -f "$cfg" ] && valid_configs+=("$cfg")
        done

        if [ ${#valid_configs[@]} -eq 0 ]; then
            echo -e "  ${YELLOW}No configs found in configs/${NC}"
            echo "  Run option 4 to build one"
            exit 1
        fi

        i=1
        for cfg in "${valid_configs[@]}"; do
            echo "  $i. $cfg"
            ((i++))
        done

        read -p "Choice [1]: " CONFIG_CHOICE
        CONFIG_CHOICE=${CONFIG_CHOICE:-1}
        idx=$((CONFIG_CHOICE - 1))

        if [ $idx -lt 0 ] || [ $idx -ge ${#valid_configs[@]} ]; then
            echo -e "${YELLOW}Invalid choice${NC}"
            exit 1
        fi

        SELECTED_CONFIG="${valid_configs[$idx]}"
        echo ""
        echo -e "${GREEN}Selected:${NC} $SELECTED_CONFIG"

        # Update config.yaml pointer
        cat > config.yaml << EOF
# Active configuration pointer
use: $SELECTED_CONFIG
EOF
        echo -e "${GRAY}Updated config.yaml to point to $SELECTED_CONFIG${NC}"
    fi
    # Choice 1 (or Enter) falls through - use default config
fi

# Use default config
CONFIG_FILE="config.yaml"
CONFIG_ARGS="-c $CONFIG_FILE"

# Show which config is being used (follow use: pointer if present)
if [ -f "$CONFIG_FILE" ]; then
    USE_POINTER=$(grep -E "^use:" "$CONFIG_FILE" 2>/dev/null | sed 's/use:[[:space:]]*//' || true)
    if [ -n "$USE_POINTER" ]; then
        echo -e "${YELLOW}Config: $CONFIG_FILE -> $USE_POINTER${NC}"
    else
        echo -e "${YELLOW}Config: $CONFIG_FILE${NC}"
    fi
fi

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
    echo -e "${GRAY}Using duration from config${NC}"
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

if [ "$BACKGROUND_MODE" = true ]; then
    # Run in tmux session, detached
    SESSION_NAME="detector"

    # Kill existing session if any
    tmux kill-session -t $SESSION_NAME 2>/dev/null || true

    # Start new detached session
    tmux new-session -d -s $SESSION_NAME "python -m object_detection $CONFIG_ARGS $RUN_ARGS"

    echo -e "${GREEN}Started in background (tmux session: $SESSION_NAME)${NC}"
    echo ""
    echo "Commands:"
    echo "  tmux attach -t $SESSION_NAME   # view output"
    echo "  tmux kill-session -t $SESSION_NAME   # stop"
    echo ""
else
    python -m object_detection $CONFIG_ARGS $RUN_ARGS
fi
