# FPL Agent on Databricks

An agentic Fantasy Premier League assistant built on Databricks. It recommends the squad, transfers, captain and chips
for each gameweek, with grounded explanations and rigorous evaluations. See [Introduction.md](Introduction.md) for the
full project context.

## Repo layout

```
databricks.yml            # Asset Bundle entry point: names the project and the workspace to deploy to
resources/                # What Databricks should create (YAML)
  catalog.yml             #   Unity Catalog schemas (raw/bronze/silver/gold/ml) + landing Volume
  ingest_job.yml          #   Scheduled job that snapshots the FPL API every 6 hours
src/
  ingest/fetch_fpl.py     # Notebook: downloads FPL API JSON → /Volumes/fpl/raw/landing/... (timestamped)
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
