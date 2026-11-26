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

Keys: player ids are reused across seasons → always join on `(season, player_id)`; `player_code` is stable.
Season label from a timestamp: Jul–Jun, e.g. `2026-27`.

## Roadmap decision (2026-11-26)
Build the **xPts model first**, then resume Tipster Phases 2–4 as an enhancement to it (tip consensus as a model feature).
Tip collection (Phase 1) keeps running on the 6-hourly schedule meanwhile, so tip history keeps accumulating.

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
