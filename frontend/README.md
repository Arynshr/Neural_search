# Neural Search v3 — Frontend (Landing Page)

Standalone marketing/landing page for the retrieval system. Fully decoupled
from `src/neural_search` (API/retrieval core) and `src/ui` (Streamlit
internal tool) — this folder only renders a public-facing static page and
talks to the API over HTTP, nothing more.

## Why separate from core
- Core (`src/neural_search`) is a Python retrieval service. This is a static
  site — different language, different lifecycle, different deploy target.
- Streamlit (`src/ui`) is the internal ops/debug tool for collections and
  search testing. This is the public page — no coupling, no shared state.
- Independent deploy: this folder ships to Vercel/Netlify as a static
  build; core ships as a container/service. Changes to one never block
  the other.

## Stack
| Concern | Choice | Why |
|---|---|---|
| Framework | Astro | Zero JS by default, static output, ideal for a single content page |
| Styling | Tailwind CSS | Utility-first, fast to theme (dark-first, herdr.dev-inspired) |
| Font | JetBrains Mono / IBM Plex Mono | Terminal aesthetic for code/metric blocks |
| Data | Static JSON snapshot of `evaluation/results/phase3.json` | No live API call needed for the metrics table on the landing page |
| Deployment | Vercel or Netlify | Zero-config static hosting, PR previews |

## Folder Structure
```
frontend/
  src/
    pages/        # route-level .astro files (index.astro = the landing page)
    components/   # Hero, MetricsTable, FeatureCard, TerminalMock, ThemeToggle
    styles/       # global.css, tailwind config, theme CSS variables
    lib/          # small helpers (e.g. formatting metrics, theme persistence)
  public/
    assets/       # images, favicons, og-image
  astro.config.mjs
  tailwind.config.mjs
  package.json
```

## Page Sections (index.astro)
1. **Hero** — headline + animated terminal mockup showing a mock query →
   ranked results, evoking herdr.dev's live-pane style
2. **How it works** — three cards: Sparse (BM25), Dense (Qdrant), Hybrid
   (RRF + Rerank)
3. **Benchmarked results** — table sourced from `phase3.json`
   (P@5, Recall@5, MRR, nDCG@5 per method/query-type)
4. **CTA** — links to docs / GitHub / live demo

## Theming
CSS custom properties define a small palette (bg, fg, accent, muted).
Default: dark. A toggle persists preference (in-memory / cookie — no
`localStorage` if rendered inside sandboxed previews).

## Data Flow
- Build time: metrics table reads a checked-in JSON snapshot (no runtime
  dependency on the FastAPI service).
- No live search demo on this page — that stays in the internal Streamlit
  UI. Keeps the public page static, cache-friendly, and free of backend
  coupling.

## Deployment
1. `frontend/` is its own deployable unit — separate Vercel/Netlify project
   rooted at this folder.
2. Build command: `astro build` → output in `frontend/dist/`.
3. No environment variables required (no API calls at runtime).

## Explicitly Out of Scope
- No live retrieval calls from this page (avoids exposing the API publicly
  and avoids coupling release cycles)
- No auth, no collections UI — that remains in `src/ui` (Streamlit)
- No shared code/imports with `src/neural_search` or `src/ui`
