# FPL Agent on Databricks

An agentic Fantasy Premier League assistant built on Databricks. It recommends the squad, transfers, captain and chips
for each gameweek, with grounded explanations and rigorous evaluations. See [Introduction.md](Introduction.md) for the
full project context.

## Repo layout

```
databricks.yml            # Asset Bundle entry point: names the project and the workspace to deploy to
resources/                # What Databricks should create (YAML)
  catalog.yml             #   Unity Catalog schemas (raw/bronze/silver/gold/ml) + landing Volume
  ingest_job.yml          #   Scheduled job (every 6h): fetch FPL API (+ per-player history once a day) → transform
  backfill_job.yml        #   Manual job: load past seasons (vaastav) → transform
  bronze_pipeline.yml     #   Lakeflow Declarative Pipeline "fpl-transform" (bronze + silver)
src/
  ingest/fetch_fpl.py     # Notebook: downloads FPL API JSON → /Volumes/fpl/raw/landing/... (timestamped)
  ingest/fetch_element_summaries.py  # Notebook: per-player current-season history, once per day
  ingest/backfill_vaastav.py         # Notebook: past seasons from vaastav/Fantasy-Premier-League (CSV → JSON lines)
  pipelines/bronze.py     # Auto Loader: new landing files → fpl.bronze.bootstrap_static / fpl.bronze.fixtures
  pipelines/silver.py     # Parse + type bronze JSON → fpl.silver.* (players, teams, fixtures, news history, ...)
```

## Data flow so far

```
FPL API ──(fetch_fpl_api, every 6h)──► /Volumes/fpl/raw/landing/<endpoint>/dt=YYYY-MM-DD/*.json
        ──(transform: fpl-transform pipeline)
             bronze  fpl.bronze.bootstrap_static   1 row per snapshot (raw JSON text columns)
                     fpl.bronze.fixtures           1 row per fixture per snapshot
                     fpl.bronze.element_summary    1 row per player per daily snapshot
                     fpl.bronze.vaastav_*          past seasons: merged_gw, players, teams
             silver  fpl.silver.player_snapshots   1 row per player per snapshot (typed, full history)
                     fpl.silver.news_snapshots     1 row per change in status / news / chance of playing
                     fpl.silver.players            current state of each player (+ team, position)
                     fpl.silver.teams              current teams + strength ratings
                     fpl.silver.gameweeks          deadlines, current/next flags
                     fpl.silver.fixtures           fixture list with difficulty + scores
                     fpl.silver.player_gw_history  1 row per player per fixture, 2020-21 → now (~170k rows)
```

Keys: player ids are reused across seasons, so player tables are keyed by `(season, player_id)`; `player_code` is stable across seasons.

Example queries:

```sql
-- Who is injured or doubtful right now?
SELECT web_name, team_short_name, status, chance_of_playing_next_round, news
FROM fpl.silver.players WHERE status <> 'a' ORDER BY chance_of_playing_next_round;

-- How has one player's availability news evolved?
SELECT observed_at, status, chance_of_playing_next_round, news
FROM fpl.silver.news_snapshots WHERE web_name = 'Saka' ORDER BY observed_at;
```

## Common commands

```powershell
databricks bundle validate      # check the config for mistakes (changes nothing)
databricks bundle deploy        # create/update the schemas, volume and job in the workspace
databricks bundle run fpl_ingest  # run the ingestion job once, right now
```

## Data notes
- `player_gw_history`: past seasons come from vaastav, the current season from FPL's `element-summary`.
  xG is NULL before 2022-23 and defensive-contribution stats are NULL before 2025-26 (not tracked then).
  2024-25 Assistant Manager rows are excluded, and a few exact duplicate vaastav rows are dropped.

## One-off setup already done
- Historical backfill: `databricks bundle run fpl_backfill_history` (re-runnable; latest load per season wins).
- `fpl` catalog created via SQL (`CREATE CATALOG fpl`), because Free Edition's Default Storage
  doesn't allow creating catalogs from the CLI/bundle.
