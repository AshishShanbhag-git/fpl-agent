# Databricks notebook source
# MAGIC %md
# MAGIC # xPts step 2: baselines to beat
# MAGIC
# MAGIC Scores simple "expected points" rules on the training rows of `gold.player_gw_features` and writes the
# MAGIC results to `<catalog>.ml.xpts_eval`. Step 3 (LightGBM) appends its scores to the same table, so every
# MAGIC model is compared on exactly the same rows with exactly the same metrics.
# MAGIC
# MAGIC **Baselines**
# MAGIC * `zero`          - everyone scores 0 (shows how many rows are non-playing squad players)
# MAGIC * `points_avg5`   - average points over the player's last 5 gameweeks this season
# MAGIC * `weighted_form` - 60% last-3 + 40% last-5 average, blended with last season's points per gameweek
# MAGIC                     early in the season, times the number of fixtures (0 in a blank, 2 in a double)
# MAGIC * `fpl_xp`        - FPL's own expected points (`ep_next`) from our last snapshot before each deadline.
# MAGIC                     Exists only from 2026-11-25 (when our snapshots start), so it becomes comparable as
# MAGIC                     current-season gameweeks finish.
# MAGIC
# MAGIC **Why not vaastav's `xP` column for past seasons?** It is recorded *after* the gameweek: xP(g) correlates
# MAGIC 0.59-0.64 with the points of gameweek g itself but only 0.21-0.26 with gameweek g+1, and it scores a
# MAGIC near-perfect captain every week. Using it would be look-ahead, so it is not a fair baseline.
# MAGIC
# MAGIC **Splits** (by time, never random): train = 2020-21…2024-25, valid = 2025-26, test = current season.
# MAGIC **Segments**: `all` rows; `relevant` = played at least once in the last 5 GWs; `regular` = 60+ minutes in
# MAGIC at least 60% of the last 5 GWs. Segments use only pre-gameweek information (no look-ahead).
# MAGIC **Subsets**: `all_gws`; `fpl_xp_rows` = only rows where FPL's xP exists, for a fair comparison with it.

# COMMAND ----------

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

dbutils.widgets.text("catalog", "fpl")
dbutils.widgets.text("bundle_root", "")
CATALOG = dbutils.widgets.get("catalog")
ROOT = Path(dbutils.widgets.get("bundle_root") or Path.cwd().parents[1])
sys.path.insert(0, str(ROOT / "src" / "ml"))
import evaluation as E  # noqa: E402

EVAL_TABLE = f"{CATALOG}.ml.xpts_eval"
RUN_AT = datetime.now(timezone.utc)

# COMMAND ----------

# MAGIC %md ## Load training rows + FPL's own expected points

# COMMAND ----------

df = spark.sql(f"""
WITH
-- Current season: ep_next from the last snapshot taken before the deadline (ep_next = next gameweek).
dl AS (
  SELECT season, gameweek, deadline_time,
         lag(deadline_time) OVER (PARTITION BY season ORDER BY gameweek) AS prev_deadline
  FROM {CATALOG}.silver.gameweeks
),
cx AS (
  SELECT s.season, s.player_id, d.gameweek, s.ep_next AS fpl_xp
  FROM {CATALOG}.silver.player_snapshots s
  JOIN dl d ON s.season = d.season AND s.snapshot_ts < d.deadline_time
           AND (d.prev_deadline IS NULL OR s.snapshot_ts >= d.prev_deadline)
  QUALIFY row_number() OVER (PARTITION BY s.season, s.player_id, d.gameweek ORDER BY s.snapshot_ts DESC) = 1
)
SELECT f.season, f.player_id, f.gameweek, f.position, f.n_fixtures,
       f.points_avg3, f.points_avg5, f.minutes_avg5, f.played60_rate5, f.gws_season, f.prev_points,
       cx.fpl_xp,
       f.target_points
FROM {CATALOG}.gold.player_gw_features f
LEFT JOIN cx USING (season, player_id, gameweek)
WHERE NOT f.is_future
""").toPandas()

num = ["points_avg3", "points_avg5", "minutes_avg5", "played60_rate5", "gws_season", "prev_points", "fpl_xp",
       "n_fixtures", "target_points"]
df[num] = df[num].astype("float64")
current_season = df["season"].max()
df["split"] = df["season"].map(lambda s: E.split_of(s, current_season))
print(f"{len(df):,} training rows; current season {current_season}; "
      f"rows with FPL xP: {df['fpl_xp'].notna().sum():,}")
print(df[df["fpl_xp"].notna()].groupby(["season", "gameweek"]).size())

# COMMAND ----------

# MAGIC %md ## Baseline predictions

# COMMAND ----------

form = 0.6 * df["points_avg3"].fillna(0) + 0.4 * df["points_avg5"].fillna(0)
trust = df["gws_season"].fillna(0).clip(upper=5) / 5      # how much this season's form is trusted (0..1)
prior = df["prev_points"].fillna(0) / 38                    # last season's points per gameweek
BASELINES = {
    "zero": pd.Series(0.0, index=df.index),
    "points_avg5": df["points_avg5"].fillna(0),
    "weighted_form": (trust * form + (1 - trust) * prior) * df["n_fixtures"].fillna(1),
    "fpl_xp": df["fpl_xp"],                                 # NaN where FPL xP is missing
}

SUBSETS = {"all_gws": pd.Series(True, index=df.index), "fpl_xp_rows": df["fpl_xp"].notna()}

# COMMAND ----------

# MAGIC %md ## Score every (model, season, subset, segment) and save

# COMMAND ----------

res = E.score(df, BASELINES, SUBSETS, only_subset={"fpl_xp": "fpl_xp_rows"})   # xP only exists on its rows
print(f"{len(res)} result rows")
E.save(spark, EVAL_TABLE, res, RUN_AT)

# COMMAND ----------

# MAGIC %md ## Summary: validation + test seasons, relevant players

# COMMAND ----------

summary = (res[(res.segment == "relevant") & res.split.isin(["valid", "test"])]
           [["split", "season", "subset", "model", "n", "mae", "rmse", "spearman", "top1_actual", "top10_actual"]]
           .sort_values(["split", "subset", "mae"]).round(3))
display(summary)
dbutils.notebook.exit(summary.to_json(orient="records"))
