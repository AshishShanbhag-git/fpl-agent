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
# MAGIC
# MAGIC **MLflow (step 5):** every run is logged to the bundle's experiment (settings, scores, feature importance,
# MAGIC one child run per tried setting). The final model is registered in Unity Catalog as `<catalog>.ml.xpts`
# MAGIC and gets the alias `champion` if its valid RMSE (relevant players) is at least as good as the current
# MAGIC champion's. Finally the registered version is reloaded and must reproduce this notebook's predictions.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.5.0 mlflow-skinny[databricks]==2.22.0 --quiet

# COMMAND ----------

import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

dbutils.widgets.text("catalog", "fpl")
dbutils.widgets.text("bundle_root", "")
dbutils.widgets.text("experiment_id", "")      # set by the job from the bundle
CATALOG = dbutils.widgets.get("catalog")
ROOT = Path(dbutils.widgets.get("bundle_root") or Path.cwd().parents[1])
sys.path.insert(0, str(ROOT / "src" / "ml"))
import evaluation as E  # noqa: E402
import metrics as M  # noqa: E402
from xpts_model import XptsModel  # noqa: E402

EVAL_TABLE = f"{CATALOG}.ml.xpts_eval"
RUN_AT = datetime.now(timezone.utc)
MODEL_NAME = "lgbm_pos"
SEED = 42
REGISTERED_MODEL = f"{CATALOG}.ml.xpts"
ALIAS = "champion"
GATE_METRIC = "valid_relevant_rmse"

mlflow.set_registry_uri("databricks-uc")
if dbutils.widgets.get("experiment_id"):
    mlflow.set_experiment(experiment_id=dbutils.widgets.get("experiment_id"))
print("mlflow", mlflow.__version__)

# COMMAND ----------

# MAGIC %md ## Load training rows

# COMMAND ----------

FEATURES_TABLE = f"{CATALOG}.gold.player_gw_features"
df = spark.table(FEATURES_TABLE).where("NOT is_future").toPandas()

NOT_FEATURES = {"season", "player_id", "player_code", "is_future", "position", "team_id", "feature_as_of_gw",
                "target_points", "target_minutes", "dup_count"}
FEATURES = [c for c in df.columns if c not in NOT_FEATURES]
for c in FEATURES + ["target_points"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")   # DECIMAL/BIGINT -> float

current_season = df["season"].max()
df["split"] = df["season"].map(lambda s: E.split_of(s, current_season))
# The feature table is a materialized view (no Delta version history), so record a data fingerprint instead.
FEATURES_VERSION = f"{len(df)} rows to {current_season} GW{int(df.loc[df.season == current_season, 'gameweek'].max())}"
print(f"{len(df):,} rows, {len(FEATURES)} features; current season {current_season}")
print(df.groupby(["split", "position"]).size().unstack())

# COMMAND ----------

# MAGIC %md ## Tune on valid, retrain on train + valid, predict test

# COMMAND ----------

GRID = [dict(num_leaves=nl, min_child_samples=mcs) for nl, mcs in itertools.product([15, 31, 63], [50, 200])]
BASE = dict(objective="regression", learning_rate=0.03, n_estimators=3000, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1)

run = mlflow.start_run(run_name=f"{MODEL_NAME} {RUN_AT:%Y-%m-%d %H:%M}")
mlflow.log_params({"features_table": FEATURES_TABLE, "features_table_version": FEATURES_VERSION,
                   "n_features": len(FEATURES), "train_seasons": "2020-21..2024-25", "valid_season": E.VALID_SEASON,
                   "test_season": current_season, "selection_metric": "valid rmse (per position)",
                   **{f"base_{k}": v for k, v in BASE.items()}})
mlflow.log_dict({"features": FEATURES, "grid": GRID}, "config.json")

pred = pd.Series(float("nan"), index=df.index)
chosen, importance, model_strings = {}, {}, {}
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
        with mlflow.start_run(run_name=f"{pos} leaves={params['num_leaves']} min_child={params['min_child_samples']}",
                              nested=True):
            mlflow.log_params({"position": pos, **params})
            mlflow.log_metrics({"valid_rmse": r, "best_iteration": m.best_iteration_})
        if best is None or r < best[0]:
            best = (r, params, m.best_iteration_, m)
    rmse_va, params, n_trees, m_tr = best
    chosen[pos] = {**params, "n_trees": int(n_trees), "valid_rmse": round(rmse_va, 4)}
    print(pos, chosen[pos])
    mlflow.log_params({f"{pos}_{k}": v for k, v in chosen[pos].items() if k != "valid_rmse"})

    # Train + valid rows scored by the train-only model (valid = honest, train = in-sample).
    pred.loc[tr.index] = m_tr.predict(tr[FEATURES], num_iteration=n_trees)
    pred.loc[va.index] = m_tr.predict(va[FEATURES], num_iteration=n_trees)

    # Final model: same settings on train + valid, ~10% more trees for ~20% more data. Predict test once.
    trva = pd.concat([tr, va])
    m_final = lgb.LGBMRegressor(**{**BASE, **params, "n_estimators": int(n_trees * 1.1)})
    m_final.fit(trva[FEATURES], trva["target_points"])
    if len(te):
        pred.loc[te.index] = m_final.predict(te[FEATURES])

    model_strings[pos] = m_final.booster_.model_to_string()
    gain = pd.Series(m_final.booster_.feature_importance("gain"), index=FEATURES)
    importance[pos] = (gain / gain.sum()).sort_values(ascending=False).head(10).round(3).to_dict()

# COMMAND ----------

# MAGIC %md ## Score with the shared evaluation and save

# COMMAND ----------

res = E.score(df, {MODEL_NAME: pred})
E.save(spark, EVAL_TABLE, res, RUN_AT)

# Headline metrics per split x segment, weighted over seasons (valid and test are single seasons anyway).
for (split, segment), g in res[res.split.isin(["valid", "test"])].groupby(["split", "segment"]):
    r = g.iloc[0]
    mlflow.log_metrics({f"{split}_{segment}_{k}": float(r[k]) for k in
                        ["mae", "rmse", "spearman", "top1_actual", "top10_actual"] if pd.notna(r[k])})
mlflow.log_dict(importance, "feature_importance.json")
mlflow.log_text(res.to_csv(index=False), "xpts_eval_rows.csv")

# COMMAND ----------

# MAGIC %md ## Register in Unity Catalog, promote if not worse, and prove the saved model reproduces predictions

# COMMAND ----------

model = XptsModel(model_strings, FEATURES)
example = df[["position"] + FEATURES].head(20)
model_info = mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=model,
    code_paths=[str(ROOT / "src" / "ml" / "xpts_model.py")],
    signature=infer_signature(example, model.predict(None, example)),
    input_example=example.head(5),
    pip_requirements=["lightgbm==4.5.0", "pandas", f"mlflow-skinny=={mlflow.__version__}"],
    registered_model_name=REGISTERED_MODEL,
)
version = model_info.registered_model_version
new_score = mlflow.get_run(run.info.run_id).data.metrics[GATE_METRIC]

client = MlflowClient(registry_uri="databricks-uc")
try:
    champ = client.get_model_version_by_alias(REGISTERED_MODEL, ALIAS)
    champ_score = client.get_run(champ.run_id).data.metrics.get(GATE_METRIC, float("inf"))
except Exception:
    champ, champ_score = None, float("inf")
promoted = new_score <= champ_score + 1e-9
if promoted:
    client.set_registered_model_alias(REGISTERED_MODEL, ALIAS, version)
client.set_model_version_tag(REGISTERED_MODEL, version, GATE_METRIC, f"{new_score:.4f}")
client.set_model_version_tag(REGISTERED_MODEL, version, "features_table_version", str(FEATURES_VERSION))
mlflow.set_tags({"registered_version": version, "promoted_to_champion": str(promoted)})
print(f"registered {REGISTERED_MODEL} v{version}; {GATE_METRIC}={new_score:.4f} vs champion "
      f"{'none' if champ is None else f'v{champ.version} {champ_score:.4f}'} -> promoted={promoted}")

# Reload from Unity Catalog and compare with the in-notebook predictions on the test season.
te = df[df.split == "test"]
reloaded = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}/{version}")
diff = (reloaded.predict(te[["position"] + FEATURES])["xpts"].to_numpy() - pred.loc[te.index].to_numpy())
max_diff = float(abs(diff).max())
mlflow.log_metric("reload_max_abs_diff", max_diff)
assert max_diff < 1e-6, f"reloaded model differs from notebook predictions by {max_diff}"
print(f"reload check OK on {len(te):,} test rows (max diff {max_diff:.2e})")
mlflow.end_run()

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
dbutils.notebook.exit(json.dumps({"version": version, "promoted": promoted, "chosen": chosen,
                                  "comparison": [r.asDict() for r in comparison.collect()]}, default=str))
