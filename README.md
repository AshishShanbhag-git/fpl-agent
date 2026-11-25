# FPL Agent on Databricks

An agentic Fantasy Premier League assistant built on Databricks. It recommends the squad, transfers, captain and chips
for each gameweek, with grounded explanations and rigorous evaluations. See [Introduction.md](Introduction.md) for the
full project context.

## Repo layout

```
databricks.yml            # Asset Bundle entry point: names the project and the workspace to deploy to
resources/                # What Databricks should create (YAML)
  catalog.yml             #   Unity Catalog schemas (raw/bronze/silver/gold/ml) + landing Volume
  ingest_job.yml          #   Scheduled job (every 6h): fetch FPL API → load bronze
  bronze_pipeline.yml     #   Lakeflow Declarative Pipeline "fpl-transform" (bronze + silver)
src/
  ingest/fetch_fpl.py     # Notebook: downloads FPL API JSON → /Volumes/fpl/raw/landing/... (timestamped)
  pipelines/bronze.py     # Auto Loader: new landing files → fpl.bronze.bootstrap_static / fpl.bronze.fixtures
  pipelines/silver.py     # Parse + type bronze JSON → fpl.silver.* (players, teams, fixtures, news history, ...)
```

## Data flow so far

```
FPL API ──(fetch_fpl_api, every 6h)──► /Volumes/fpl/raw/landing/<endpoint>/dt=YYYY-MM-DD/*.json
        ──(transform: fpl-transform pipeline)
             bronze  fpl.bronze.bootstrap_static   1 row per snapshot (raw JSON text columns)
                     fpl.bronze.fixtures           1 row per fixture per snapshot
             silver  fpl.silver.player_snapshots   1 row per player per snapshot (typed, full history)
                     fpl.silver.news_snapshots     1 row per change in status / news / chance of playing
                     fpl.silver.players            current state of each player (+ team, position)
                     fpl.silver.teams              current teams + strength ratings
                     fpl.silver.gameweeks          deadlines, current/next flags
                     fpl.silver.fixtures           fixture list with difficulty + scores
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

## One-off setup already done
- `fpl` catalog created via SQL (`CREATE CATALOG fpl`), because Free Edition's Default Storage
  doesn't allow creating catalogs from the CLI/bundle.
