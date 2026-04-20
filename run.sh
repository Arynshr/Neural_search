#!/usr/bin/env bash
# =============================================================================
# Neural Search — Project Runner
# Usage:
#   ./run.sh setup      — create venv, install deps, init dirs, download NLTK
#   ./run.sh api        — start FastAPI server
#   ./run.sh ui         — start Streamlit UI
#   ./run.sh ingest     — ingest documents from data/documents/
#   ./run.sh ingest --reset  — wipe indexes and reingest
#   ./run.sh eval       — run evaluation against evaluation/queries.json
#   ./run.sh verify     — verify BM25 and Qdrant indexes are in sync
#   ./run.sh start      — start API + UI together (background API)
#   ./run.sh stop       — stop background API process
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
ENV_FILE="$PROJECT_ROOT/.env"
PID_FILE="$PROJECT_ROOT/.api.pid"
LOG_DIR="$PROJECT_ROOT/logs"
DATA_DIR="$PROJECT_ROOT/data"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()     { echo -e "${CYAN}[neural-search]${NC} $*"; }
success() { echo -e "${GREEN}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
error()   { echo -e "${RED}[✗]${NC} $*"; exit 1; }

# ── Guards ────────────────────────────────────────────────────────────────────
require_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        error "Virtual environment not found. Run: ./run.sh setup"
    fi
    source "$VENV_DIR/bin/activate"
}

require_env() {
    if [[ ! -f "$ENV_FILE" ]]; then
        error ".env file not found. Copy .env.example to .env and fill in GROQ_API_KEY"
    fi
    # Fail fast if GROQ_API_KEY is missing or placeholder
    source "$ENV_FILE"
    if [[ -z "${GROQ_API_KEY:-}" || "$GROQ_API_KEY" == "your_groq_api_key_here" ]]; then
        error "GROQ_API_KEY is not set in .env"
    fi
}

# ── Commands ──────────────────────────────────────────────────────────────────
cmd_setup() {
    log "Setting up Neural Search..."

    # Python version check
    python_bin=$(command -v python3 || command -v python || error "Python not found")
    py_version=$($python_bin --version 2>&1 | awk '{print $2}')
    log "Python: $py_version at $python_bin"

    # Create .env if missing
    if [[ ! -f "$ENV_FILE" ]]; then
        cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
        warn ".env created from .env.example — fill in GROQ_API_KEY before running"
    fi

    # Create virtual environment
    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating virtual environment..."
        $python_bin -m venv "$VENV_DIR"
        success "Virtual environment created at .venv"
    else
        warn "Virtual environment already exists — skipping creation"
    fi

    source "$VENV_DIR/bin/activate"

    # Install deps
    log "Installing dependencies..."
    pip install --quiet --upgrade pip
    if command -v uv &> /dev/null; then
        uv pip install -e ".[dev]"
    else
        pip install -e ".[dev]"
    fi
    success "Dependencies installed"

    # NLTK data
    log "Downloading NLTK stopwords..."
    python3 -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True)"
    success "NLTK data downloaded"

    # Create data dirs
    log "Creating data directories..."
    mkdir -p \
        "$DATA_DIR/documents" \
        "$DATA_DIR/qdrant" \
        "$DATA_DIR/bm25_index" \
        "$DATA_DIR/snapshots" \
        "$LOG_DIR" \
        "$PROJECT_ROOT/evaluation"
    success "Data directories created"

    success "Setup complete. Next steps:"
    echo "  1. Edit .env and set your GROQ_API_KEY"
    echo "  2. Drop PDF/DOCX files into data/documents/"
    echo "  3. ./run.sh ingest"
    echo "  4. ./run.sh start"
}

cmd_api() {
    require_venv
    require_env
    log "Starting FastAPI server on http://localhost:8000 ..."
    log "Docs available at http://localhost:8000/docs"
    cd "$PROJECT_ROOT"
    uvicorn neural_search.api.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --log-level info
}

cmd_ui() {
    require_venv
    require_env
    log "Starting Streamlit UI on http://localhost:8501 ..."
    cd "$PROJECT_ROOT/ui"
    streamlit run app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --browser.gatherUsageStats false
}

cmd_ingest() {
    require_venv
    require_env
    RESET_FLAG=""
    for arg in "$@"; do
        [[ "$arg" == "--reset" ]] && RESET_FLAG="--reset"
    done
    log "Running document ingestion from $DATA_DIR/documents/ $RESET_FLAG"
    cd "$PROJECT_ROOT"
    python3 scripts/ingest_documents.py --input-dir "$DATA_DIR/documents" $RESET_FLAG
    success "Ingestion complete"
}

cmd_eval() {
    require_venv
    require_env
    if [[ ! -f "$PROJECT_ROOT/evaluation/queries.json" ]]; then
        error "evaluation/queries.json not found — add your labeled queries first"
    fi
    log "Running evaluation..."
    cd "$PROJECT_ROOT"
    python3 scripts/run_eval.py "$@"
}

cmd_verify() {
    require_venv
    require_env
    log "Verifying index integrity..."
    cd "$PROJECT_ROOT"
    python3 scripts/verify_index.py
}

cmd_start() {
    require_venv
    require_env

    # Start API in background
    log "Starting API in background..."
    mkdir -p "$LOG_DIR"
    cd "$PROJECT_ROOT"
    nohup uvicorn neural_search.api.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --log-level info \
        > "$LOG_DIR/api.log" 2>&1 &
    echo $! > "$PID_FILE"
    success "API started (PID $(cat $PID_FILE)) — logs at logs/api.log"

    # Wait for API to be ready
    log "Waiting for API to be ready..."
    for i in {1..15}; do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            success "API is ready"
            break
        fi
        sleep 1
        [[ $i -eq 15 ]] && error "API did not start in time — check logs/api.log"
    done

    # Start UI in foreground
    log "Starting Streamlit UI on http://localhost:8501 ..."
    cd "$PROJECT_ROOT/ui"
    streamlit run app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --browser.gatherUsageStats false
}

cmd_stop() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            rm "$PID_FILE"
            success "API process $PID stopped"
        else
            warn "Process $PID not running — cleaning up PID file"
            rm "$PID_FILE"
        fi
    else
        warn "No API PID file found — nothing to stop"
    fi
}

# ── Entrypoint ────────────────────────────────────────────────────────────────
COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    setup)   cmd_setup "$@" ;;
    api)     cmd_api "$@" ;;
    ui)      cmd_ui "$@" ;;
    ingest)  cmd_ingest "$@" ;;
    eval)    cmd_eval "$@" ;;
    verify)  cmd_verify "$@" ;;
    start)   cmd_start "$@" ;;
    stop)    cmd_stop "$@" ;;
    help|*)
        echo ""
        echo "  Neural Search — Runner Script"
        echo ""
        echo "  Usage: ./run.sh <command> [options]"
        echo ""
        echo "  Commands:"
        echo "    setup          Create venv, install deps, init dirs, download NLTK data"
        echo "    ingest         Ingest documents from data/documents/"
        echo "    ingest --reset Wipe indexes and reingest from scratch"
        echo "    api            Start FastAPI server (foreground)"
        echo "    ui             Start Streamlit UI (foreground)"
        echo "    start          Start API (background) + UI (foreground)"
        echo "    stop           Stop background API process"
        echo "    verify         Check BM25 and Qdrant index are in sync"
        echo "    eval           Run evaluation against evaluation/queries.json"
        echo ""
        ;;
esac
