"""
eval_bm25_retrieval.py — BM25 retrieval evaluation with Textual TUI.

Metrics:
    - MRR@10      Mean Reciprocal Rank at cutoff 10
    - Recall@100  Recall at cutoff 100
    - Latency     p50 / p95 / p99 / mean in ms

Data expectations:
    eval_queries.jsonl  — one JSON object per line: {"id": "...", "text": "..."}
    qrels.json          — Dict[qid, Dict[pid, relevance_int]]

Usage:
    uv run evals/lexical_search/eval_bm25_retrieval.py
    uv run evals/lexical_search/eval_bm25_retrieval.py --index data/indices/bm25.pkl
    uv run evals/lexical_search/eval_bm25_retrieval.py --limit 200 --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Dependency guards
# ---------------------------------------------------------------------------
try:
    import jsonlines
except ImportError as e:
    raise ImportError("jsonlines required: uv add jsonlines") from e

try:
    from textual.app import App, ComposeResult
    from textual.widgets import DataTable, Footer, Header, Label, ProgressBar, Static
    from textual.containers import Container, Vertical, Horizontal
    from textual import work
except ImportError as e:
    raise ImportError("textual required: uv add textual") from e

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]
    logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Path bootstrap — works whether run directly or via uv
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src"))

from neural_search.lexical_search.bm25_index import BM25Index  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INDEX   = BASE / "data" / "indices" / "bm25.pkl"
DEFAULT_QUERIES = BASE / "data" / "processed" / "eval_queries.jsonl"
DEFAULT_QRELS   = BASE / "data" / "processed" / "qrels.json"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_queries(path: Path, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    queries: List[Tuple[str, str]] = []
    with jsonlines.open(path) as reader:
        for i, record in enumerate(reader):
            if limit and i >= limit:
                break
            qid  = str(record.get("id") or record.get("_id") or i)
            text = str(record.get("text") or record.get("query") or "").strip()
            if text:
                queries.append((qid, text))
    return queries


def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    with open(path) as f:
        raw = json.load(f)
    # Normalise all keys to str
    return {
        str(qid): {str(pid): int(rel) for pid, rel in docs.items()}
        for qid, docs in raw.items()
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mrr_at_k(
    run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> float:
    scores: List[float] = []
    for qid, pid_scores in run.items():
        relevant = {pid for pid, rel in qrels.get(qid, {}).items() if rel > 0}
        ranked   = sorted(pid_scores, key=pid_scores.__getitem__, reverse=True)[:k]
        rr = 0.0
        for rank, pid in enumerate(ranked, 1):
            if pid in relevant:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return sum(scores) / len(scores) if scores else 0.0


def recall_at_k(
    run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 100,
) -> float:
    scores: List[float] = []
    for qid, pid_scores in run.items():
        relevant = {pid for pid, rel in qrels.get(qid, {}).items() if rel > 0}
        if not relevant:
            continue
        ranked  = sorted(pid_scores, key=pid_scores.__getitem__, reverse=True)[:k]
        hits    = sum(1 for pid in ranked if pid in relevant)
        scores.append(hits / len(relevant))
    return sum(scores) / len(scores) if scores else 0.0


def latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    s = sorted(latencies_ms)
    n = len(s)

    def pct(p: float) -> float:
        idx = min(int(p / 100 * n), n - 1)
        return s[idx]

    return {
        "mean": sum(s) / n,
        "std":  (sum((x - sum(s) / n) ** 2 for x in s) / n) ** 0.5,
        "min":  s[0],
        "p50":  pct(50),
        "p95":  pct(95),
        "p99":  pct(99),
        "max":  s[-1],
    }


# ---------------------------------------------------------------------------
# Core eval runner (no TUI dependency)
# ---------------------------------------------------------------------------

def run_eval(
    index_path: Path,
    queries_path: Path,
    qrels_path: Path,
    limit: Optional[int],
    top_k_mrr: int = 10,
    top_k_recall: int = 100,
    progress_cb=None,
) -> dict:
    """
    Run full evaluation. Returns a results dict.
    progress_cb(done, total) called after each query if provided.
    """
    index   = BM25Index.load(index_path)
    queries = load_queries(queries_path, limit=limit)
    qrels   = load_qrels(qrels_path)

    run_mrr:    Dict[str, Dict[str, float]] = {}
    run_recall: Dict[str, Dict[str, float]] = {}
    latencies:  List[float] = []

    total = len(queries)
    for i, (qid, text) in enumerate(queries):
        # MRR pass — top 10
        pids_mrr, scores_mrr, lat_ms = index.search(text, top_k=top_k_mrr)
        run_mrr[qid] = {pid: score for pid, score in zip(pids_mrr, scores_mrr)}
        latencies.append(lat_ms)

        # Recall pass — top 100 (additional search, latency not double-counted)
        if top_k_recall != top_k_mrr:
            pids_rec, scores_rec, _ = index.search(text, top_k=top_k_recall)
            run_recall[qid] = {pid: score for pid, score in zip(pids_rec, scores_rec)}
        else:
            run_recall[qid] = run_mrr[qid]

        if progress_cb:
            progress_cb(i + 1, total)

    mrr    = mrr_at_k(run_mrr, qrels, k=top_k_mrr)
    recall = recall_at_k(run_recall, qrels, k=top_k_recall)
    lat    = latency_stats(latencies)

    return {
        "corpus_size":    index.corpus_size,
        "queries_evaluated": total,
        "top_k_mrr":      top_k_mrr,
        "top_k_recall":   top_k_recall,
        "mrr":            round(mrr, 4),
        "recall":         round(recall, 4),
        "latency":        {k: round(v, 2) for k, v in lat.items()},
    }


# ---------------------------------------------------------------------------
# Textual TUI
# ---------------------------------------------------------------------------

TITLE = "BM25 Retrieval Evaluation"

CSS = """
Screen {
    background: #0d1117;
}

#header-bar {
    height: 3;
    background: #161b22;
    border-bottom: solid #30363d;
    padding: 0 2;
    align: left middle;
}

#title-label {
    color: #58a6ff;
    text-style: bold;
    width: 1fr;
}

#status-label {
    color: #8b949e;
    text-align: right;
}

#progress-container {
    height: 5;
    background: #161b22;
    border: solid #30363d;
    margin: 1 2;
    padding: 1 2;
}

#progress-label {
    color: #8b949e;
    margin-bottom: 1;
}

ProgressBar {
    width: 1fr;
}

ProgressBar > .bar--bar {
    color: #238636;
}

ProgressBar > .bar--complete {
    color: #2ea043;
}

#metrics-container {
    margin: 0 2;
    height: auto;
}

#metrics-title {
    color: #58a6ff;
    text-style: bold;
    margin-bottom: 1;
    padding-left: 1;
}

DataTable {
    height: auto;
    background: #161b22;
    border: solid #30363d;
}

DataTable > .datatable--header {
    background: #21262d;
    color: #58a6ff;
    text-style: bold;
}

DataTable > .datatable--cursor {
    background: #1f6feb33;
}

DataTable > .datatable--odd-row {
    background: #161b22;
}

DataTable > .datatable--even-row {
    background: #0d1117;
}

#retrieval-table {
    margin-bottom: 1;
}

#latency-table {
    margin-bottom: 1;
}

#footer-bar {
    height: 3;
    background: #161b22;
    border-top: solid #30363d;
    padding: 0 2;
    align: left middle;
    color: #8b949e;
}

.pass { color: #2ea043; text-style: bold; }
.fail { color: #f85149; text-style: bold; }
.metric-value { color: #e6edf3; }
.metric-label { color: #8b949e; }
"""


class EvalApp(App):
    CSS = CSS
    TITLE = TITLE

    def __init__(
        self,
        index_path: Path,
        queries_path: Path,
        qrels_path: Path,
        limit: Optional[int],
        out_path: Optional[Path],
    ):
        super().__init__()
        self.index_path   = index_path
        self.queries_path = queries_path
        self.qrels_path   = qrels_path
        self.limit        = limit
        self.out_path     = out_path
        self._results: Optional[dict] = None
        self._total_queries = 0

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Label(f"◆  {TITLE}", id="title-label"),
            Label("Initialising …", id="status-label"),
            id="header-bar",
        )
        with Vertical(id="progress-container"):
            yield Label("Running evaluation …", id="progress-label")
            yield ProgressBar(total=100, show_eta=False, id="progress-bar")
        with Vertical(id="metrics-container"):
            yield Label("RETRIEVAL METRICS", id="metrics-title")
            yield DataTable(id="retrieval-table", show_cursor=False)
            yield Label("LATENCY  (ms)", id="latency-title")
            yield DataTable(id="latency-table", show_cursor=False)
        yield Label(
            "q  quit    s  save results",
            id="footer-bar",
        )

    def on_mount(self) -> None:
        # Retrieval table columns
        rt = self.query_one("#retrieval-table", DataTable)
        rt.add_columns("Metric", "Value", "SLA", "Status")

        # Latency table columns
        lt = self.query_one("#latency-table", DataTable)
        lt.add_columns("Stat", "Value (ms)")

        self._start_eval()

    @work(thread=True)
    def _start_eval(self) -> None:
        def progress_cb(done: int, total: int) -> None:
            self._total_queries = total
            pct = int(done / total * 100)
            self.call_from_thread(
                self.query_one("#progress-bar", ProgressBar).update,
                progress=pct,
            )
            self.call_from_thread(
                self.query_one("#progress-label", Label).update,
                f"Evaluating … {done}/{total} queries",
            )
            self.call_from_thread(
                self.query_one("#status-label", Label).update,
                f"{done}/{total}",
            )

        try:
            results = run_eval(
                index_path=self.index_path,
                queries_path=self.queries_path,
                qrels_path=self.qrels_path,
                limit=self.limit,
                progress_cb=progress_cb,
            )
            self._results = results
            self.call_from_thread(self._render_results, results)
        except Exception as exc:
            self.call_from_thread(
                self.query_one("#status-label", Label).update,
                f"[red]Error: {exc}[/red]",
            )

    def _render_results(self, r: dict) -> None:
        self.query_one("#progress-label", Label).update(
            f"Complete — {r['queries_evaluated']:,} queries evaluated  |  "
            f"Corpus: {r['corpus_size']:,} passages"
        )
        self.query_one("#status-label", Label).update("Done ✓")

        # Retrieval table
        rt = self.query_one("#retrieval-table", DataTable)
        mrr_val    = r["mrr"]
        recall_val = r["recall"]
        p50        = r["latency"]["p50"]
        p95        = r["latency"]["p95"]

        mrr_pass    = mrr_val >= 0.18
        recall_pass = recall_val >= 0.80
        p50_pass    = p50 <= 50.0
        p95_pass    = p95 <= 200.0

        def badge(ok: bool) -> str:
            return "[@click]✅ PASS[/]" if ok else "[@click]❌ FAIL[/]"

        rt.add_row(
            f"MRR@{r['top_k_mrr']}",
            f"{mrr_val:.4f}",
            "≥ 0.18",
            "✅ PASS" if mrr_pass else "❌ FAIL",
        )
        rt.add_row(
            f"Recall@{r['top_k_recall']}",
            f"{recall_val:.4f}",
            "≥ 0.80",
            "✅ PASS" if recall_pass else "❌ FAIL",
        )
        rt.add_row(
            "p50 latency",
            f"{p50:.1f} ms",
            "< 50 ms",
            "✅ PASS" if p50_pass else "❌ FAIL",
        )
        rt.add_row(
            "p95 latency",
            f"{p95:.1f} ms",
            "< 200 ms",
            "✅ PASS" if p95_pass else "❌ FAIL",
        )

        # Latency table
        lt  = self.query_one("#latency-table", DataTable)
        lat = r["latency"]
        for stat, val in [
            ("Mean",  lat["mean"]),
            ("Std",   lat["std"]),
            ("Min",   lat["min"]),
            ("p50",   lat["p50"]),
            ("p95",   lat["p95"]),
            ("p99",   lat["p99"]),
            ("Max",   lat["max"]),
        ]:
            lt.add_row(stat, f"{val:.1f}")

        # Optionally persist
        if self.out_path:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_path, "w") as f:
                json.dump(r, f, indent=2)
            self.query_one("#footer-bar", Label).update(
                f"q  quit    Results saved → {self.out_path}"
            )

    def on_key(self, event) -> None:
        if event.key == "q":
            self.exit()
        if event.key == "s" and self._results and not self.out_path:
            path = BASE / "evals" / "results" / "bm25_eval.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(self._results, f, indent=2)
            self.query_one("#footer-bar", Label).update(
                f"q  quit    Saved → {path}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate BM25 retrieval quality.")
    p.add_argument("--index",   type=Path, default=DEFAULT_INDEX)
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    p.add_argument("--qrels",   type=Path, default=DEFAULT_QRELS)
    p.add_argument("--limit",   type=int,  default=None,
                   help="Cap queries (smoke test). Omit for full eval.")
    p.add_argument("--out",     type=Path, default=None,
                   help="Optional path to save JSON results.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for label, path in [
        ("index",   args.index),
        ("queries", args.queries),
        ("qrels",   args.qrels),
    ]:
        if not path.exists():
            print(f"[error] {label} not found: {path}")
            sys.exit(1)

    app = EvalApp(
        index_path=args.index,
        queries_path=args.queries,
        qrels_path=args.qrels,
        limit=args.limit,
        out_path=args.out,
    )
    app.run()
