#!/bin/bash
# Example: pandoc-based PDF report triggered by command action
#
# Test this script standalone:
#   ./scripts/example_report.sh
#
# In production, command_runner.py sets these env vars automatically.

set -e

# ============================================================================
# Simulate event data (command_runner.py sets these in production)
# ============================================================================

: "${EVENT_TYPE:=LINE_CROSS}"
: "${EVENT_NAME:=traffic_passing}"
: "${TIMESTAMP:=$(date -Iseconds)}"
: "${TIMESTAMP_RELATIVE:=127.5}"
: "${OBJECT_CLASS:=car}"
: "${CONFIDENCE:=0.87}"
: "${TRACK_ID:=42}"
: "${BBOX:=150,200,450,380}"
: "${LINE_NAME:=signal}"
: "${DIRECTION:=left_to_right}"
: "${FRAME_PATH:=}"

# ============================================================================
# Generate report
# ============================================================================

REPORT_DIR="${REPORT_DIR:-reports}"
mkdir -p "$REPORT_DIR"

REPORT_NAME="event_$(date +%Y%m%d_%H%M%S).pdf"
REPORT_PATH="$REPORT_DIR/$REPORT_NAME"
TEMP_MD=$(mktemp /tmp/report_XXXXXX.md)

# Build markdown
cat > "$TEMP_MD" << EOF
---
title: Detection Event
date: $TIMESTAMP
---

# $EVENT_NAME

**$OBJECT_CLASS** crossed **$LINE_NAME** ($DIRECTION)

| Field | Value |
|-------|-------|
| Confidence | ${CONFIDENCE} |
| Track ID | ${TRACK_ID} |
| Time | ${TIMESTAMP_RELATIVE}s into session |

EOF

# Include frame if available
if [ -n "$FRAME_PATH" ] && [ -f "$FRAME_PATH" ]; then
    echo "![Captured frame]($FRAME_PATH){ width=100% }" >> "$TEMP_MD"
fi

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "pandoc not installed - showing markdown instead:"
    echo "----------------------------------------"
    cat "$TEMP_MD"
    echo "----------------------------------------"
    echo ""
    echo "Install with: sudo apt install pandoc texlive-latex-base"
    rm "$TEMP_MD"
    exit 0
fi

# Convert to PDF
pandoc "$TEMP_MD" -o "$REPORT_PATH" --pdf-engine=pdflatex -V geometry:margin=1in 2>/dev/null || {
    echo "PDF generation failed (missing texlive?) - saving as HTML instead"
    REPORT_PATH="${REPORT_PATH%.pdf}.html"
    pandoc "$TEMP_MD" -o "$REPORT_PATH" --standalone
}

rm "$TEMP_MD"
echo "Report: $REPORT_PATH"
