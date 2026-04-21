#!/usr/bin/env bash
# =============================================================================
# Neural Search — Master Control Script
# Usage:
#   ./run.sh setup        — create venv, install deps, download NLTK data
#   ./run.sh ingest       — ingest documents from data/documents/
#   ./run.sh ingest --reset  — wipe indexes and reingest
#   ./run.sh api          — start FastAPI server
#   ./run.sh ui           — start Streamlit dashboard
#   ./run.sh start        — start API + UI together (background + foreground)
#   ./run.sh eval         — run retrieval evaluation
#   ./run.sh verify       — verify BM25 and Qdrant index sync
#   ./run.sh stop         — stop background API process
#   ./run.sh clean        — wipe all indexes and snapshots
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
SRC_DIR="$PROJECT_ROOT/src"
ENV_FILE="$PROJECT_ROOT/.env"
PID_FILE="$PROJECT_ROOT/.api.pid"
API_HOST="127.0.0.1"
API_PORT="8000"
API_LOG="$PROJECT_ROOT/logs/api.log"
DOCUMENTS_DIR="$PROJECT_ROOT/data/documents"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()     { echo -e "${CYAN}[neural-search]${RESET} $*"; }
success() { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[!]${RESET} $*"; }
error()   { echo -e "${RED}[✗]${RESET} $*"; exit 1; }

# ── Helpers ───────────────────────────────────────────────────────────────────
check_env() {
    if [[ ! -f "$ENV_FILE" ]]; then
        warn ".env not found — copying from .env.example"
        cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
        warn "Fill in GROQ_API_KEY in .env before running ingest or api"
    fi
}

activate_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        error "Virtual environment not found — run './run.sh setup' first"
    fi
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    export PYTHONPATH="$SRC_DIR:$PYTHONPATH"
}

ensure_dirs() {
    mkdir -p "$PROJECT_ROOT/logs" \
             "$PROJECT_ROOT/data/documents" \
             "$PROJECT_ROOT/data/qdrant" \
             "$PROJECT_ROOT/data/bm25_index" \
             "$PROJECT_ROOT/data/snapshots"
}

# ── Commands ──────────────────────────────────────────────────────────────────
cmd_setup() {
    log "Setting up Neural Search environment..."
    ensure_dirs
    check_env

    # Create venv if missing
    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"

    # Install deps
    log "Installing dependencies..."
    if command -v uv &>/dev/null; then
        uv pip install -e "$PROJECT_ROOT[dev]"
    else
        pip install --upgrade pip -q
        pip install -e "$PROJECT_ROOT[dev]" -q
    fi

    # NLTK data
    log "Downloading NLTK stopwords..."
    python3 -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True)"

    success "Setup complete — activate with: source .venv/bin/activate"
    echo -e "\n${BOLD}Next steps:${RESET}"
    echo "  1. Add your GROQ_API_KEY to .env"
    echo "  2. Drop PDF/DOCX files into data/documents/"
    echo "  3. ./run.sh ingest"
    echo "  4. ./run.sh start"
}

cmd_ingest() {
    activate_venv
    check_env
    log "Starting document ingestion from: $DOCUMENTS_DIR"
    python3 "$PROJECT_ROOT/scripts/ingest_documents.py" \
        --input-dir "$DOCUMENTS_DIR" "$@"
    success "Ingestion complete"
}

cmd_api() {
    activate_venv
    check_env
    log "Starting FastAPI server at http://$API_HOST:$API_PORT"
    log "Swagger docs: http://$API_HOST:$API_PORT/docs"
    uvicorn neural_search.api.main:app \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --reload \
        --log-level info
}

cmd_ui() {
    activate_venv
    log "Starting Streamlit dashboard..."
    cd "$PROJECT_ROOT/ui"
    streamlit run app.py \
        --server.port 8501 \
        --server.address localhost \
        --browser.gatherUsageStats false
}

cmd_start() {
    activate_venv
    check_env
    ensure_dirs

    # Check if API already running
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        warn "API already running (PID $(cat "$PID_FILE")) — skipping"
    else
        log "Starting FastAPI in background..."
        PYTHONPATH="$SRC_DIR" uvicorn neural_search.api.main:app \
            --host "$API_HOST" \
            --port "$API_PORT" \
            --log-level info \
            >> "$API_LOG" 2>&1 &
        echo $! > "$PID_FILE"
        success "API started (PID $(cat "$PID_FILE")) — logs: logs/api.log"

        # Wait for API to be ready
        log "Waiting for API to be ready..."
        for i in {1..15}; do
            if curl -sf "http://$API_HOST:$API_PORT/health" &>/dev/null; then
                success "API is up at http://$API_HOST:$API_PORT"
                break
            fi
            sleep 1
        done
    fi

    # Start Streamlit in foreground
    log "Starting Streamlit dashboard (Ctrl+C to stop)..."
    cd "$PROJECT_ROOT/ui"
    PYTHONPATH="$SRC_DIR" streamlit run app.py \
        --server.port 8501 \
        --server.address localhost \
        --browser.gatherUsageStats false

    # On exit, stop API
    cmd_stop
}

cmd_eval() {
    activate_venv
    check_env
    log "Running retrieval evaluation..."
    python3 "$PROJECT_ROOT/scripts/run_eval.py" "$@"
}

cmd_verify() {
    activate_venv
    log "Verifying index sync..."
    python3 "$PROJECT_ROOT/scripts/verify_index.py"
}

cmd_stop() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            success "API stopped (PID $PID)"
        else
            warn "API process not running"
        fi
        rm -f "$PID_FILE"
    else
        warn "No API PID file found — nothing to stop"
    fi
}

cmd_clean() {
    warn "This will wipe all indexes and snapshots. Continue? [y/N]"
    read -r confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        rm -rf "$PROJECT_ROOT/data/qdrant/"*
        rm -rf "$PROJECT_ROOT/data/bm25_index/"*
        rm -rf "$PROJECT_ROOT/data/snapshots/"*
        success "All indexes and snapshots cleared"
    else
        log "Aborted"
    fi
}

# ── Entrypoint ────────────────────────────────────────────────────────────────
COMMAND="${1:-help}"
shift || true   # remaining args passed through to subcommands

case "$COMMAND" in
    setup)   cmd_setup ;;
    ingest)  cmd_ingest "$@" ;;
    api)     cmd_api ;;
    ui)      cmd_ui ;;
    start)   cmd_start ;;
    eval)    cmd_eval "$@" ;;
    verify)  cmd_verify ;;
    stop)    cmd_stop ;;
    clean)   cmd_clean ;;
    help|*)
        echo -e "\n${BOLD}Neural Search — run.sh${RESET}"
        echo -e "${CYAN}Usage: ./run.sh <command> [options]${RESET}\n"
        echo "  setup          Create venv, install deps, download NLTK data"
        echo "  ingest         Ingest documents from data/documents/"
        echo "  ingest --reset Wipe indexes and reingest"
        echo "  api            Start FastAPI server (foreground)"
        echo "  ui             Start Streamlit dashboard (foreground)"
        echo "  start          Start API (background) + Streamlit (foreground)"
        echo "  eval           Run retrieval evaluation (P@K, MRR, nDCG)"
        echo "  eval --k 5     Evaluate at k=5"
        echo "  verify         Check BM25 and Qdrant index sync"
        echo "  stop           Stop background API process"
        echo "  clean          Wipe all indexes and snapshots"
        echo ""
        ;;
esac
