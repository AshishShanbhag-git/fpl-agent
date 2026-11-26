# Databricks notebook source
# MAGIC %md
# MAGIC # xPts step 3: LightGBM, one model per position
# MAGIC
# MAGIC Learns "expected points next gameweek" from `gold.player_gw_features` and scores it with exactly the same
# MAGIC code as the baselines (`evaluation.py`), appending to `<catalog>.ml.xpts_eval` as model `lgbm_pos`.
# MAGIC
# MAGIC **Why one model per position?** Points come from different things: keepers from saves and clean sheets,
# MAGIC defenders from clean sheets and defensive contributions, attackers from goals and assists.
# MAGIC
# MAGIC **Honest protocol (no peeking at the test season):**
# MAGIC 1. Train on 2020-21…2024-25; try a few settings; keep the one with the lowest RMSE on 2025-26 (valid).
# MAGIC    Early stopping on valid picks the number of trees.
# MAGIC 2. Retrain the chosen settings on train + valid, then predict the current season (test) once.
# MAGIC
# MAGIC **Why choose by RMSE, not MAE?** xPts is an *average* (expected) outcome. RMSE is minimised by predicting
# MAGIC the average; MAE is minimised by predicting the *typical* outcome, which for most players is ~0-2 points
# MAGIC and would under-rate haul potential. MAE is still reported for comparison.
# MAGIC
# MAGIC Train-season scores in `xpts_eval` are in-sample (the model saw those rows), so only valid/test count.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.5.0 --quiet

# COMMAND ----------

import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import pandas as pd

dbutils.widgets.text("catalog", "fpl")
dbutils.widgets.text("bundle_root", "")
CATALOG = dbutils.widgets.get("catalog")
ROOT = Path(dbutils.widgets.get("bundle_root") or Path.cwd().parents[1])
sys.path.insert(0, str(ROOT / "src" / "ml"))
import evaluation as E  # noqa: E402
import metrics as M  # noqa: E402

EVAL_TABLE = f"{CATALOG}.ml.xpts_eval"
RUN_AT = datetime.now(timezone.utc)
MODEL_NAME = "lgbm_pos"
SEED = 42

# COMMAND ----------

# MAGIC %md ## Load training rows

# COMMAND ----------

df = spark.table(f"{CATALOG}.gold.player_gw_features").where("NOT is_future").toPandas()

NOT_FEATURES = {"season", "player_id", "player_code", "is_future", "position", "team_id", "feature_as_of_gw",
                "target_points", "target_minutes", "dup_count"}
FEATURES = [c for c in df.columns if c not in NOT_FEATURES]
for c in FEATURES + ["target_points"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")   # DECIMAL/BIGINT -> float

current_season = df["season"].max()
df["split"] = df["season"].map(lambda s: E.split_of(s, current_season))
print(f"{len(df):,} rows, {len(FEATURES)} features; current season {current_season}")
print(df.groupby(["split", "position"]).size().unstack())

# COMMAND ----------

# MAGIC %md ## Tune on valid, retrain on train + valid, predict test

# COMMAND ----------

GRID = [dict(num_leaves=nl, min_child_samples=mcs) for nl, mcs in itertools.product([15, 31, 63], [50, 200])]
BASE = dict(objective="regression", learning_rate=0.03, n_estimators=3000, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1)

pred = pd.Series(float("nan"), index=df.index)
chosen, importance = {}, {}
for pos in ["GKP", "DEF", "MID", "FWD"]:
    d = df[df["position"] == pos]
    tr, va, te = d[d.split == "train"], d[d.split == "valid"], d[d.split == "test"]

    best = None
    for params in GRID:
        m = lgb.LGBMRegressor(**BASE, **params)
        m.fit(tr[FEATURES], tr["target_points"], eval_set=[(va[FEATURES], va["target_points"])],
              eval_metric="l2", callbacks=[lgb.early_stopping(100, verbose=False)])
        p = m.predict(va[FEATURES], num_iteration=m.best_iteration_)
        r = M.rmse(p.tolist(), va["target_points"].tolist())
        if best is None or r < best[0]:
            best = (r, params, m.best_iteration_, m)
    rmse_va, params, n_trees, m_tr = best
    chosen[pos] = {**params, "n_trees": int(n_trees), "valid_rmse": round(rmse_va, 4)}
    print(pos, chosen[pos])

    # Train + valid rows scored by the train-only model (valid = honest, train = in-sample).
    pred.loc[tr.index] = m_tr.predict(tr[FEATURES], num_iteration=n_trees)
    pred.loc[va.index] = m_tr.predict(va[FEATURES], num_iteration=n_trees)

    # Final model: same settings on train + valid, ~10% more trees for ~20% more data. Predict test once.
    trva = pd.concat([tr, va])
    m_final = lgb.LGBMRegressor(**{**BASE, **params, "n_estimators": int(n_trees * 1.1)})
    m_final.fit(trva[FEATURES], trva["target_points"])
    if len(te):
        pred.loc[te.index] = m_final.predict(te[FEATURES])

    gain = pd.Series(m_final.booster_.feature_importance("gain"), index=FEATURES)
    importance[pos] = (gain / gain.sum()).sort_values(ascending=False).head(10).round(3).to_dict()

# COMMAND ----------

# MAGIC %md ## Score with the shared evaluation and save

# COMMAND ----------

res = E.score(df, {MODEL_NAME: pred})
E.save(spark, EVAL_TABLE, res, RUN_AT)

comparison = spark.sql(f"""
  SELECT split, segment, model, sum(n) AS n,
         round(sum(mae * n) / sum(n), 3) AS mae, round(sqrt(sum(rmse * rmse * n) / sum(n)), 3) AS rmse,
         round(avg(spearman), 3) AS spearman, round(avg(top1_actual), 2) AS captain_pts,
         round(avg(top10_actual), 2) AS top10_pts
  FROM {EVAL_TABLE}
  WHERE subset = 'all_gws' AND split IN ('valid', 'test') AND model IN ('{MODEL_NAME}', 'weighted_form', 'points_avg5')
  GROUP BY ALL ORDER BY split DESC, segment, rmse
""")
display(comparison)
dbutils.notebook.exit(json.dumps({"chosen": chosen, "importance": importance,
                                  "comparison": [r.asDict() for r in comparison.collect()]}, default=str))
