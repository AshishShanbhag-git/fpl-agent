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
  bronze_pipeline.yml     #   Lakeflow Declarative Pipeline definition for the bronze layer
src/
  ingest/fetch_fpl.py     # Notebook: downloads FPL API JSON → /Volumes/fpl/raw/landing/... (timestamped)
  pipelines/bronze.py     # Auto Loader: new landing files → fpl.bronze.bootstrap_static / fpl.bronze.fixtures
```

## Data flow so far

```
FPL API ──(fetch_fpl_api, every 6h)──► /Volumes/fpl/raw/landing/<endpoint>/dt=YYYY-MM-DD/*.json
        ──(load_bronze: Auto Loader)──► fpl.bronze.bootstrap_static  (1 row per snapshot)
                                        fpl.bronze.fixtures          (1 row per fixture per snapshot)
```

Example query: current injury news straight from bronze

```sql
SELECT p:web_name::string AS player, p:status::string AS status, p:news::string AS news
FROM (SELECT explode(cast(parse_json(elements) AS ARRAY<VARIANT>)) AS p
      FROM fpl.bronze.bootstrap_static
      WHERE _snapshot_ts = (SELECT max(_snapshot_ts) FROM fpl.bronze.bootstrap_static))
WHERE p:news::string <> '';
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
