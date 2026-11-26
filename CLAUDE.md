# CLAUDE.md — FPL Agent on Databricks

Project goals, architecture and constraints: see [Introduction.md](Introduction.md). This file tracks **current state**.
The owner is a beginner with tech setup: explain *why* each step is done, define jargon, give click-by-click steps.

## Environment facts (verified)
- Databricks **Free Edition**, serverless only. Workspace auth via CLI profile `DEFAULT` (never commit `~/.databrickscfg`).
- Outbound internet from serverless works for: `fantasy.premierleague.com`, `raw.githubusercontent.com`, PyPI,
  `allaboutfpl.com`, `www.fplwire.com`, `www.fantasyfootballpundit.com`. **No GitHub Actions** are used; all fetching runs as Databricks Job tasks.
- Free Edition cannot create catalogs via CLI/bundle → catalog `fpl` was created once with SQL `CREATE CATALOG fpl`.
- SQL warehouse for ad-hoc queries: "Serverless Starter Warehouse" (`f3ca550bf3bcd5c5`).
- Everything else is in the Asset Bundle (`databricks.yml`, `resources/*.yml`), single target `prod`.

## Commands
```powershell
databricks bundle validate; databricks bundle deploy
databricks bundle run fpl_ingest                 # scheduled every 6h (00:15/06:15/12:15/18:15 Europe/London)
databricks bundle run fpl_bronze                 # the "fpl-transform" pipeline only (resource key kept as fpl_bronze)
databricks bundle run fpl_backfill_history       # past seasons from vaastav (manual)
databricks bundle run fpl_tips_backfill          # tip articles from sitemaps (manual)
databricks bundle run fpl_xpts_eval              # score xPts baselines -> fpl.ml.xpts_eval (manual)
python -m unittest discover -s tests -v          # unit tests (pure Python, run locally)
```

## Jobs & pipeline
- `fpl-ingest` (scheduled): `fetch_fpl_api` → `fetch_element_summaries` (once/UTC day); `fetch_tips` (independent);
  → `transform` (run_if ALL_DONE) = pipeline `fpl-transform` (bronze + silver, serverless, triggered).
- `fpl-backfill-history`, `fpl-tips-backfill`: manual, each ends with the same `transform` pipeline task.
- Only one pipeline update can run at a time: don't trigger jobs concurrently.

## Tables (catalog `fpl`)
| Layer | Table | Grain / notes |
|---|---|---|
| raw | Volume `raw.landing` | FPL API + vaastav files, timestamped, `prevent_destroy` |
| raw | Volume `raw.tips_landing` | **PRIVATE** tip-article JSON (full text), `source=<name>/dt=<date>/` |
| bronze | `bootstrap_static`, `fixtures`, `element_summary` | raw JSON text columns + `_source_file`, `_snapshot_ts`, `_ingested_at` |
| bronze | `vaastav_merged_gw`, `vaastav_players`, `vaastav_teams` | past seasons 2020-21..2025-26, column `season` |
| bronze | `tip_articles_raw` | **PRIVATE**, one row per fetched article version; never expose `text` |
| silver | `player_snapshots` | (season, player_id, snapshot_ts); streaming; price £m, status, news, transfers_in_event, selected_by_percent |
| silver | `news_snapshots` | change log of status/news/chance_of_playing |
| silver | `players`, `teams`, `gameweeks`, `fixtures` | current state (latest snapshot) |
| silver | `player_gw_history` | (season, player_id, fixture_id); 2020-21 → now; price, minutes, points, xG, transfers, selected |

| gold | `player_gw_features` | (season, player_id, gameweek); training rows 2020-21 → last finished GW + `is_future` rows for next 5 GWs; features as of `feature_as_of_gw` (< gameweek, enforced by expect_or_fail); targets `target_points`, `target_minutes` |
| ml | `xpts_eval` | (model, season, split, subset, segment) metrics: mae, rmse, pearson, spearman, top1/top10 actual + oracle. Each model's job deletes its own rows then appends |

Keys: player ids are reused across seasons → always join on `(season, player_id)`; `player_code` is stable.
Season label from a timestamp: Jul–Jun, e.g. `2026-27`.

## Roadmap decision (2026-11-26)
Build the **xPts model first**, then resume Tipster Phases 2–4 as an enhancement to it (tip consensus as a model feature).
Tip collection (Phase 1) keeps running on the 6-hourly schedule meanwhile, so tip history keeps accumulating.

## xPts model progress
- Step 1 (2026-11-26) done: `gold.player_gw_features` in `src/pipelines/gold.py` (same pipeline). ~167k training rows, 3,370 prediction rows.
  Simple baselines on training rows: MAE ≈ 1.03–1.28 for "points_avg5", corr ≈ 0.46–0.50; predicting 0 gives MAE ≈ 1.10–1.41
  because most rows are non-playing squad players. **Step 2 must evaluate on relevant players too** (e.g. minutes_avg5 > 0 / starters),
  not only all rows.
- Not in features yet: availability (status / chance_of_playing) exists only from 2026-11-25 → apply as a current-season adjustment.
- Step 2 (2026-11-26) done: `src/ml/evaluate_baselines.py` (job `fpl-xpts-eval`) + `src/ml/metrics.py` (tested).
  Splits: train 2020-21..2024-25, valid 2025-26, test = current season. Segments (pre-GW info only): all /
  relevant (minutes_avg5 > 0) / regular (played60_rate5 >= 0.6). **Best baseline = weighted_form**
  (0.6·avg3 + 0.4·avg5, blended with prev-season pts/GW for the first 5 GWs, × n_fixtures).
  Bars for step 3 (relevant, valid 2025-26): MAE 1.976, RMSE 2.93, Spearman 0.44, top-10 actual 4.18 (oracle 12.9),
  captain (top-1) 4.59. Test 2026-27 relevant: MAE 1.91, Spearman 0.44.
- **vaastav `xP` is look-ahead** (recorded after the GW: corr with same-GW points 0.6 vs 0.2 with next GW; picks
  a ~11-pt captain every week). Never use it as a feature or baseline. FPL `ep_next` baseline comes only from our
  own pre-deadline snapshots (from GW12 2026-27 on); `fpl_xp` rows appear in xpts_eval once GW12 finishes.
- Ranking metrics for constant predictions (`zero`) are tie-break noise; ignore them.
- Step 3 (2026-11-26) done: `src/ml/train_xpts.py` (job `fpl-xpts-train`, lightgbm 4.5.0), model `lgbm_pos` in
  xpts_eval. Shared scoring in `src/ml/evaluation.py` (both notebooks use it). Protocol: tune 6-setting grid per
  position on valid by **RMSE** (xPts = mean, so RMSE not MAE), early stopping; retrain on train+valid with 1.1×
  trees; predict test once. Train-season rows in xpts_eval are in-sample (ignore).
  Relevant players: valid 2025-26 MAE 1.78 / RMSE 2.67 / Spearman 0.53 / captain 5.16 / top-10 5.01 (baseline
  1.98 / 2.93 / 0.44 / 4.59 / 4.18); test 2026-27 MAE 1.77 / RMSE 2.61 / Spearman 0.55 / captain 7.0 / top-10 5.07.
  Calibrated (mean pred ≈ mean actual). Top features: last_minutes, minutes_avg3 (playing time ≈ 55-70% of gain),
  then selected, price, n_fixtures, points_avg3. Valid 2025-26 under-predicts relevant by ~0.15: DC points were new
  that season (train-only model couldn't learn them; the final model includes 2025-26).
- Ideas not done yet: separate minutes model (P(plays 60)) × points-given-minutes; availability adjustment from
  status/chance_of_playing for the current season.
- Next: step 4/5 MLflow tracking + register `fpl.ml.xpts` in UC → step 6 score future rows into `gold.xpts` in the job.

## Optimiser requirements (owner, 2026-11-26)
The agent must give, each gameweek, **(1) transfers / chip advice starting from the owner's actual squad** and
**(2) the best lineup** (XI, captain, vice, bench order) for the resulting squad. Not a fresh "best team" each week.
Design: ILP (PuLP) over a 3–5 GW horizon using `gold.xpts`; inputs = current squad, selling prices, bank, free transfers,
chips left (two sets; first set expires at the GW19 deadline). Hits cost −4 per extra transfer. Only the first GW of the
plan is recommended; re-plan every week. Chips are evaluated by re-solving with each chip (WC/FH = squad unconstrained,
BB = bench counts, TC = captain ×3) and recommended only if the gain beats a keep-it threshold.
Squad data: public `entry/{team_id}/event/{gw}/picks/` + `entry/{team_id}/transfers/` (purchase prices → selling prices).

## Tipster Accuracy feature (spec pasted by owner on 2026-11-26)
Phased; **stop after each phase and wait for go-ahead**. Copyright: raw text only in private bronze; silver/gold/app
store tip facts + URL only. No look-ahead: a tip counts only if `published_at < deadline` of its gameweek.

### Phase 1: Sources & ingestion (bronze) — built 2026-11-26
- Sources approved after robots.txt + terms review: `allaboutfpl` (RSS + sitemap backfill), `fplwire` (RSS + sitemap),
  `ffpundit` (snapshot pages rewritten weekly; published_at = fetched_at; no backfill). Config: `config/tip_sources.yml`.
- Rejected: fantasyfootballscout + fantasyfootballcommunity (terms 6.6 forbid automated extraction), fantasyfootballfix
  (data reuse forbidden), sportskeeda/skysports (block AI crawlers), planetfpl (TLS error). Inactive/unclear: evening_standard, goal.
- Code: `src/tips/tiplib.py` (pure logic, tested in `tests/test_tiplib.py`), `src/tips/fetch_tips.py` (notebook, trafilatura 2.0.0),
  `scripts/check_robots.py`. Spec deviation (approved by owner): fetching runs in Databricks, not GitHub Actions.
- published_at priority: RSS pubDate > HTML article:published_time/JSON-LD (date-only → 23:59:59 UTC) > sitemap lastmod;
  snapshot pages use fetch time.

- Landing JSON fields: source, url, title, season, published_at, published_at_source, fetched_at, gw_guess, gw_in_title,
  flags (`stale_snapshot`, `previous_season`), discovery_method, extractor, text, content_hash.
- Result 2026-11-26: bronze `tip_articles_raw` = 169 rows (allaboutfpl 137, fplwire 29, ffpundit 3); 116 from 2026-27,
  53 older (2023-2025, found via sitemaps). This season, gw_guess agrees with the GW in the title for all but 1 (a stale ffpundit page).

### Open issues
- First 169 bronze rows were written before the season/flags fix: older-season rows carry `gw_guess=1`. **Silver must
  recompute gameweek from published_at + deadlines per season** (never trust bronze gw_guess) and drop `stale_snapshot` rows.
- 53 previous-season articles could be scored against vaastav history (would need past-season deadlines). Optional.
- One allaboutfpl URL has two versions in bronze (re-fetched before the URL de-dup fix); silver should keep the earliest per URL.
- Only 3 active sources → small leaderboard; owner may request permission from FFScout etc. to add more.
- `ffpundit` history exists only from 2026-11-26 and depends on fetching before each deadline (6-hourly schedule).
- FPL "most captained" per GW exists only in bronze `bootstrap_static.events`; Phase 3 needs a new table (don't modify `silver.gameweeks`).
- `ai_query()` availability on Free Edition not yet tested (Phase 2).
- Phase 4 depends on the xPts model, agent and app, which don't exist yet.
