# FPL Agent on Databricks — project context

## Goal
Build an agentic Fantasy Premier League (FPL) assistant as an **FDE (Forward Deployed Engineer) portfolio project**.
It recommends squad, transfers, captain and chips each gameweek. The owner (Ashish) doesn't follow matches, so it must
handle player availability/injury news automatically. Recommendations only — transfers are made manually in the PL app
(no automated login/transfers).

Owner background: AWS AI Practitioner + Databricks Gen AI certified. Primary stack: **Databricks**, AWS (Bedrock) optional.

## Differentiators (the portfolio story)
1. **News → structured signal**: LLM extracts `{injury_type, expected_return_gw, p_available}` from FPL player news;
   backtest shows it predicts minutes better than the official `chance_of_playing_next_round`. Headline metric.
2. **LLM calls tools, never does the maths**: agent calls an ILP optimiser for picks, then explains / answers what-ifs, citing data.
3. **Evaluation & observability**: backtested points vs baseline, golden-set agent evals, custom scorers, cost/latency per query.

Competitors for reference: FPL Review (projections + MILP solver), Fantasy Football Hub (AI team rating, transfers, expert content).
Neither publishes rigorous evals or grounded "why" explanations — that's the gap.

## Environment & constraints
- **Databricks Free Edition** (personal account, not the employer's partner workspace — keep IP/portfolio personal).
- Serverless only; **outbound internet restricted to trusted domains** → FPL API likely blocked.
  Day-1 check: `requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")` from a notebook.
  If blocked: GitHub Actions cron fetches the API and uploads JSON to a UC Volume via Databricks CLI.
  **Result (2026-11-22): FPL API is reachable from a Free Edition serverless notebook** → ingest directly with a
  scheduled Databricks Job (no GitHub Actions fetcher needed; keep it only as a fallback).
- Free Edition limits: 1 Vector Search endpoint (1 unit), up to 3 Apps (auto-stop after 24h), no GPU serving,
  no provisioned throughput, no Agent Bricks (use code-based Mosaic AI Agent Framework), daily compute quota, non-commercial.
- Everything as code with **Databricks Asset Bundles**; public GitHub repo.

## Data sources
Base URL `https://fantasy.premierleague.com/api/` (public, no auth, be gentle — cache/throttle):
- `bootstrap-static/` players, teams, gameweeks, prices, form, `status`, `news`, `chance_of_playing_next_round`
- `fixtures/` fixtures + FDR
- `element-summary/{player_id}/` per-player GW history + upcoming fixtures
- `entry/{team_id}/event/{gw}/picks/` owner's squad
Historical seasons: GitHub `vaastav/Fantasy-Premier-League`.
**Store every snapshot with a timestamp** — news/price/injury history only exists from the day ingestion starts.

## 2026/27 rules to respect
£100m budget; 2 GK / 5 DEF / 5 MID / 3 FWD; max 3 per club; extra transfers cost −4.
Two sets of chips (Wildcard, Free Hit, Triple Captain, Bench Boost); first set must be used before the GW19 deadline.
Assistant Manager chip removed. Defensive contribution points still exist; BPS rebalanced this season.

## Architecture (Unity Catalog: catalog `fpl`, schemas raw/bronze/silver/gold/ml)
1. Ingest: scheduled Databricks Job (every 6h) fetches the API → Volume `fpl.raw.landing` → Lakeflow declarative pipeline + Auto Loader → bronze.
2. Silver: players, teams, fixtures, player_gw_history, news_snapshots. Gold: per-player-per-GW feature table.
3. News signals: `ai_query()` / FM API → `silver.news_signals`; backtest vs official field, log to MLflow.
4. xPts model: baseline weighted-form formula → LightGBM per position; MLflow tracking, UC model registry; score next 5 GWs → `gold.xpts`.
5. Optimiser: PuLP ILP (budget, positions, club cap, free transfers/hits, chip flags) → `gold.recommendations`.
6. Agent: Mosaic AI Agent Framework; tools `get_my_squad`, `get_xpts`, `run_optimizer`, `search_news` (Vector Search).
   LLM via Databricks Foundation Model APIs (optional: Bedrock as external model).
7. Eval: MLflow GenAI evaluation — golden set of 30–50 questions; scorers for correctness, groundedness, advice-matches-optimiser,
   correct tool use, no invented injury claims. MLflow Tracing on.
8. App: Databricks App (Streamlit) — team ID input, squad view, recommendations, chat, "why?".
9. Orchestration: one Job (file-arrival / deadline-aware): transform → signals → score → optimise → notify.
10. Packaging: README + architecture diagram, eval report, 3-min demo, write-up; onboard mini-league friends as "customers".

## Costs
Free Edition, FPL API, GitHub Actions (public repo): free. Optional Bedrock: pay-per-token, small — set an AWS budget alert.

## Next step
Phase 0–1: repo skeleton + Asset Bundle, Databricks ingestion job (API → Volume), bronze ingestion pipeline.
