# Databricks notebook source
# MAGIC %md
# MAGIC # xPts step 6: score the next 5 gameweeks → `gold.xpts`
# MAGIC
# MAGIC Runs after the `transform` pipeline in the 6-hourly `fpl-ingest` job.
# MAGIC
# MAGIC **Cheap by design.** The prediction rows' features come from the last *finished* gameweek, so they change
# MAGIC roughly once a week. The first cell computes a fingerprint of (feature rows + champion model version) using
# MAGIC only Spark; if it matches the last scoring run, the notebook exits in seconds without installing LightGBM.
# MAGIC
# MAGIC Outputs:
# MAGIC * `gold.xpts`         - latest predictions, one row per (season, player_id, gameweek). The optimiser reads this.
# MAGIC * `gold.xpts_history` - every scoring run appended, so we can later check what we predicted *before* each
# MAGIC   deadline against what actually happened (live evaluation, no hindsight).

# COMMAND ----------

from databricks.sdk import WorkspaceClient

dbutils.widgets.text("catalog", "fpl")
dbutils.widgets.dropdown("force", "false", ["false", "true"])
CATALOG = dbutils.widgets.get("catalog")
FORCE = dbutils.widgets.get("force") == "true"
MODEL = f"{CATALOG}.ml.xpts"
XPTS, HISTORY = f"{CATALOG}.gold.xpts", f"{CATALOG}.gold.xpts_history"

# Champion version via the Databricks SDK (pre-installed; no MLflow needed for this check).
aliases = WorkspaceClient().registered_models.get(MODEL, include_aliases=True).aliases or []
champion = next((str(a.version_num) for a in aliases if a.alias_name == "champion"), None)
if champion is None:
    raise RuntimeError(f"{MODEL} has no 'champion' alias: run the fpl-xpts-train job first")

# Order-independent fingerprint of every prediction row (decimal sum: no overflow).
fp = spark.sql(f"""
  SELECT count(*) AS n, cast(sum(cast(xxhash64(to_json(struct(*))) AS DECIMAL(38, 0))) AS STRING) AS h
  FROM {CATALOG}.gold.player_gw_features WHERE is_future
""").first()
FINGERPRINT = f"{fp.n}:{fp.h}"

last = None
if spark.catalog.tableExists(XPTS):
    last = spark.sql(f"SELECT first(model_version) v, first(features_hash) h FROM {XPTS}").first()
if not FORCE and last is not None and last.v == champion and last.h == FINGERPRINT:
    dbutils.notebook.exit(f"skipped: inputs unchanged (model v{champion}, {fp.n} rows)")
print(f"scoring: model v{champion}, {fp.n} rows (last run: {None if last is None else (last.v, last.h)})")

# COMMAND ----------

# MAGIC %pip install lightgbm==4.5.0 mlflow-skinny[databricks]==2.22.0 --quiet

# COMMAND ----------

# %pip restarts Python, so re-read the settings and recompute the fingerprint.
from datetime import datetime, timezone

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient

CATALOG = dbutils.widgets.get("catalog")
MODEL = f"{CATALOG}.ml.xpts"
XPTS, HISTORY = f"{CATALOG}.gold.xpts", f"{CATALOG}.gold.xpts_history"
aliases = WorkspaceClient().registered_models.get(MODEL, include_aliases=True).aliases or []
champion = next(str(a.version_num) for a in aliases if a.alias_name == "champion")
fp = spark.sql(f"""
  SELECT count(*) AS n, cast(sum(cast(xxhash64(to_json(struct(*))) AS DECIMAL(38, 0))) AS STRING) AS h
  FROM {CATALOG}.gold.player_gw_features WHERE is_future
""").first()
FINGERPRINT = f"{fp.n}:{fp.h}"

mlflow.set_registry_uri("databricks-uc")
model = mlflow.pyfunc.load_model(f"models:/{MODEL}/{champion}")   # pinned version, same as the alias now
rows = spark.table(f"{CATALOG}.gold.player_gw_features").where("is_future").toPandas()
inputs = [c for c in model.metadata.get_input_schema().input_names() if c != "position"]
x = rows[["position"]].copy()
x[inputs] = rows[inputs].apply(pd.to_numeric, errors="coerce").astype("float64")   # int/DECIMAL -> double
rows["xpts"] = model.predict(x)["xpts"].to_numpy()
assert rows["xpts"].notna().all(), "some prediction rows got no xPts (unknown position?)"

# COMMAND ----------

# MAGIC %md ## Save latest + append history

# COMMAND ----------

scored = spark.createDataFrame(rows[["season", "player_id", "gameweek", "xpts"]])
scored.createOrReplaceTempView("scored")
out = spark.sql(f"""
  SELECT f.season, f.player_id, f.player_code, p.web_name, f.position, f.team_id, f.gameweek, f.n_fixtures,
         f.feature_as_of_gw, cast(s.xpts AS DOUBLE) AS xpts,
         '{champion}' AS model_version, '{FINGERPRINT}' AS features_hash, current_timestamp() AS scored_at
  FROM {CATALOG}.gold.player_gw_features f
  JOIN scored s USING (season, player_id, gameweek)
  LEFT JOIN {CATALOG}.silver.players p USING (season, player_id)
  WHERE f.is_future
""")
(out.write.mode("overwrite").option("overwriteSchema", "true")
    .option("comment", "Latest xPts predictions for the next 5 gameweeks (one row per season, player_id, "
                       "gameweek). Written by src/ml/score_xpts.py with models:/fpl.ml.xpts@champion.")
    .saveAsTable(XPTS))
out.write.mode("append").saveAsTable(HISTORY)

summary = spark.sql(f"""
  SELECT gameweek, count(*) AS players, round(avg(xpts), 2) AS avg_xpts, round(max(xpts), 2) AS max_xpts,
         max_by(web_name, xpts) AS top_player
  FROM {XPTS} GROUP BY gameweek ORDER BY gameweek
""")
display(summary)
dbutils.notebook.exit(f"scored {len(rows)} rows with model v{champion}: "
                      + "; ".join(f"GW{r.gameweek} top {r.top_player} {r.max_xpts}" for r in summary.collect()))
