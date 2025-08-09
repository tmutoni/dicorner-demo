
# DiCorner Demo — Next.js + FastAPI (Deployable)

## Run (Docker)
```bash
export NEXT_PUBLIC_POSTHOG_KEY=phx_hzqJJkcF8MlRsiFiIVlKdcEK3vyDaXTq0Jh5hHV9IFeFJY8
docker compose up --build
```
- Frontend: http://localhost:3000
- Backend:  http://localhost:8000

## Pages
- `/dashboard` — AUROC, AUPRC, ECE, overrides, time‑to‑value
- `/model-evals` — GRACE highlights + slice explorer
- `/transparency` — Transparency Card + Override (records via API)
- `/what-if` — Threshold slider → projected lift

## Backend API
- `POST /ingest/event` — append event & retrain
- `POST /eval/run` — recompute metrics
- `GET /eval/metrics[?slice=persona:Budget]` — metrics
- `POST /override` — record override
- `GET /overrides` — list overrides
- `GET /nba` — next‑best‑actions
- `GET /ttv` — seconds from ingest to first insight

## Notes
- Synthetic data seeded at `backend/data/seed_events.csv`
- Logistic Regression churn stub; re‑trains on ingest
- Metrics: AUROC, AUPRC, Brier, ECE + persona/campaign/channel slices

## Strategic Layer (YC decks → product)
- North‑star KPIs: Δ trial→paid, Δ M2 retention, ↓ analyst time
- Trust: transparency card, override, calibration, slice parity ≤ 2pp
- Responsible AI: DPIA, canary→rollback, model registry, drift runbook


## New in v1.1
- `/ethics` page (ECE, Brier, parity heatmap, autorater checks, partner cards)
- API: `GET /ethics/status`, `GET /partners`
