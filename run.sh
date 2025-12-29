#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
GRAY='\033[0;90m'
RED='\033[0;31m'
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

# ============================================================================
# Helper functions
# ============================================================================

list_configs() {
    # Returns array of config files in valid_configs
    configs=(configs/*.yaml configs/*.yml)
    valid_configs=()
    for cfg in "${configs[@]}"; do
        [ -f "$cfg" ] && valid_configs+=("$cfg")
    done
}

select_config() {
    # Lists configs and lets user pick one
    # Sets SELECTED_CONFIG variable
    list_configs

    if [ ${#valid_configs[@]} -eq 0 ]; then
        echo -e "  ${YELLOW}No configs found in configs/${NC}"
        echo "  Run option 4 to build one"
        return 1
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
        return 1
    fi

    SELECTED_CONFIG="${valid_configs[$idx]}"
    return 0
}

check_tmux() {
    # Returns 0 if tmux is available, 1 otherwise
    command -v tmux &> /dev/null
}

check_running_session() {
    # Returns 0 if detector session exists
    tmux has-session -t detector 2>/dev/null
}

prompt_run_mode() {
    # Ask foreground vs background
    # Sets BACKGROUND_MODE
    if check_tmux; then
        echo ""
        echo "Run mode:"
        echo "  1. Foreground (normal)"
        echo "  2. Background (tmux - survives SSH disconnect)"
        read -p "Choice [1]: " RUN_MODE_CHOICE
        if [ "$RUN_MODE_CHOICE" = "2" ]; then
            BACKGROUND_MODE=true
        fi
    fi
}

# ============================================================================
# Activate virtual environment
# ============================================================================

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

# ============================================================================
# Header and status
# ============================================================================

echo -e "${CYAN}Object Detection System${NC}"
echo ""

# Show current config
if [ -f "config.yaml" ]; then
    CURRENT_CONFIG=$(grep -E "^use:" "config.yaml" 2>/dev/null | sed 's/use:[[:space:]]*//' || true)
    if [ -n "$CURRENT_CONFIG" ]; then
        echo -e "Current config: ${YELLOW}$CURRENT_CONFIG${NC}"
    else
        echo -e "Current config: ${YELLOW}config.yaml (inline)${NC}"
    fi
fi

# Check if detector is already running in tmux
if check_tmux && check_running_session; then
    echo -e "Status: ${GREEN}Detector running in background${NC}"
fi
echo ""

# ============================================================================
# Main menu
# ============================================================================

if [ "$SKIP_MENU" = false ] && [ "$YES_MODE" = false ]; then
    echo "What would you like to do?"
    echo "  1. Run"

    # Show tmux options based on availability and state
    if check_tmux; then
        if check_running_session; then
            echo "  2. Attach to running detector"
            echo "  3. Stop running detector"
            MENU_OFFSET=2
        else
            echo "  2. Run in background (tmux)"
            MENU_OFFSET=1
        fi
    else
        echo -e "  ${GRAY}2. Run in background (install tmux first)${NC}"
        MENU_OFFSET=1
    fi

    # Remaining options (numbers shift based on tmux state)
    PICK_CONFIG_OPT=$((2 + MENU_OFFSET))
    BUILD_CONFIG_OPT=$((3 + MENU_OFFSET))
    EDIT_CONFIG_OPT=$((4 + MENU_OFFSET))

    echo "  $PICK_CONFIG_OPT. Pick a config"
    echo "  $BUILD_CONFIG_OPT. Build new config"
    echo "  $EDIT_CONFIG_OPT. Edit a config"

    read -p "Choice [1]: " ENTRY_CHOICE
    ENTRY_CHOICE=${ENTRY_CHOICE:-1}

    # Handle choice based on tmux state
    if [ "$ENTRY_CHOICE" = "1" ]; then
        # Run - will prompt for mode later
        :
    elif [ "$ENTRY_CHOICE" = "2" ]; then
        if check_tmux; then
            if check_running_session; then
                # Attach to running session
                echo ""
                echo -e "${GREEN}Attaching to detector session...${NC}"
                echo -e "${GRAY}(Ctrl+B, then D to detach)${NC}"
                tmux attach -t detector
                exit 0
            else
                # Background mode
                BACKGROUND_MODE=true
            fi
        else
            echo -e "${RED}tmux not installed. Run: sudo apt install tmux${NC}"
            exit 1
        fi
    elif [ "$ENTRY_CHOICE" = "3" ] && check_tmux && check_running_session; then
        # Stop running detector
        echo ""
        read -p "Stop the running detector? (y/N): " CONFIRM_STOP
        if [[ "$CONFIRM_STOP" =~ ^[Yy]$ ]]; then
            tmux kill-session -t detector
            echo -e "${GREEN}Detector stopped${NC}"
        fi
        exit 0
    elif [ "$ENTRY_CHOICE" = "$PICK_CONFIG_OPT" ]; then
        # Pick a config
        echo ""
        echo "Available configs:"
        if select_config; then
            echo ""
            echo -e "${GREEN}Selected:${NC} $SELECTED_CONFIG"

            # Update config.yaml pointer
            cat > config.yaml << EOF
# Active configuration pointer
use: $SELECTED_CONFIG
EOF
            echo -e "${GRAY}Updated config.yaml to point to $SELECTED_CONFIG${NC}"

            # Ask about run mode before continuing
            prompt_run_mode
        else
            exit 1
        fi
    elif [ "$ENTRY_CHOICE" = "$BUILD_CONFIG_OPT" ]; then
        python -m object_detection --build-config
        exit 0
    elif [ "$ENTRY_CHOICE" = "$EDIT_CONFIG_OPT" ]; then
        echo ""
        echo "Select config to edit:"
        if select_config; then
            echo ""
            python -c "from object_detection.config import run_editor; run_editor('$SELECTED_CONFIG')"
        fi
        exit 0
    else
        echo -e "${YELLOW}Invalid choice${NC}"
        exit 1
    fi
fi

# ============================================================================
# Pre-run checks (Terraform-style)
# ============================================================================

CONFIG_FILE="config.yaml"
CONFIG_ARGS="-c $CONFIG_FILE"

# Show which config is being used
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

# ============================================================================
# Run options
# ============================================================================

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

    # If not already set to background, ask about run mode
    if [ "$BACKGROUND_MODE" = false ]; then
        prompt_run_mode
    fi

    read -p "Press Enter to start running..."
fi

# ============================================================================
# Run
# ============================================================================

echo ""
echo -e "${GREEN}=== Running ===${NC}"

if [ "$BACKGROUND_MODE" = true ]; then
    if ! check_tmux; then
        echo -e "${RED}Error: tmux not installed${NC}"
        echo "Install with: sudo apt install tmux"
        exit 1
    fi

    SESSION_NAME="detector"

    # Kill existing session if any
    tmux kill-session -t $SESSION_NAME 2>/dev/null || true

    # Start new detached session
    tmux new-session -d -s $SESSION_NAME "python -m object_detection $CONFIG_ARGS $RUN_ARGS"

    echo -e "${GREEN}Started in background (tmux session: $SESSION_NAME)${NC}"
    echo ""
    echo "Commands:"
    echo "  tmux attach -t $SESSION_NAME        # view output"
    echo "  tmux kill-session -t $SESSION_NAME  # stop"
    echo ""
    echo "Or run ./run.sh again to attach/stop from menu"
    echo ""
else
    python -m object_detection $CONFIG_ARGS $RUN_ARGS
fi
