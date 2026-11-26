"""Shared evaluation for every xPts model (baselines, LightGBM, ...): same rows, same segments, same metrics.

Needs pandas (available on Databricks); the metric maths itself lives in metrics.py and is unit tested.
"""
from __future__ import annotations

import pandas as pd

import metrics as M

VALID_SEASON = "2025-26"
EVAL_COLUMNS = """
  run_at TIMESTAMP, model STRING, season STRING, split STRING, subset STRING, segment STRING,
  n BIGINT, mae DOUBLE, rmse DOUBLE, pearson DOUBLE, spearman DOUBLE, mean_pred DOUBLE, mean_actual DOUBLE,
  top1_actual DOUBLE, top10_actual DOUBLE, top1_oracle DOUBLE, top10_oracle DOUBLE
"""


def split_of(season: str, current_season: str) -> str:
    """Time-based split: train = older seasons, valid = 2025-26, test = the current season."""
    if season == current_season:
        return "test"
    return "valid" if season == VALID_SEASON else "train"


def segments(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Player groups defined only from pre-gameweek information (no look-ahead)."""
    return {
        "all": pd.Series(True, index=df.index),
        "relevant": df["minutes_avg5"].astype(float).fillna(0) > 0,       # played in the last 5 GWs
        "regular": df["played60_rate5"].astype(float).fillna(0) >= 0.6,   # 60+ mins in >= 60% of last 5
    }


def score(df: pd.DataFrame, preds: dict[str, pd.Series], subsets: dict[str, pd.Series] | None = None,
          only_subset: dict[str, str] | None = None) -> pd.DataFrame:
    """One result row per (model, season, subset, segment). df needs season, split, gameweek, target_points,
    minutes_avg5, played60_rate5. only_subset maps a model to the single subset it is scored on."""
    subsets = subsets or {"all_gws": pd.Series(True, index=df.index)}
    segs = segments(df)
    rows = []
    for model, pred in preds.items():
        for (season, split), g in df.groupby(["season", "split"]):
            for subset, sub_mask in subsets.items():
                if only_subset and model in only_subset and only_subset[model] != subset:
                    continue
                for segment, seg_mask in segs.items():
                    mask = sub_mask.loc[g.index] & seg_mask.loc[g.index] & pred.loc[g.index].notna()
                    sel = g[mask]
                    if sel.empty:
                        continue
                    p = pred.loc[sel.index].astype(float).tolist()
                    a = sel["target_points"].astype(float).tolist()
                    gw = sel["gameweek"].tolist()
                    rows.append({
                        "model": model, "season": season, "split": split, "subset": subset, "segment": segment,
                        **M.summarise(p, a),
                        # Ranking among the players of each gameweek: the optimiser's real question.
                        "top1_actual": M.top_k_actual(gw, p, a, 1),
                        "top10_actual": M.top_k_actual(gw, p, a, 10),
                        "top1_oracle": M.top_k_actual(gw, a, a, 1),     # best possible, for scale
                        "top10_oracle": M.top_k_actual(gw, a, a, 10),
                    })
    return pd.DataFrame(rows)


def save(spark, table: str, res: pd.DataFrame, run_at) -> None:
    """Replace this run's models' earlier rows in the eval table, keep every other model's rows."""
    res = res.copy()
    res.insert(0, "run_at", run_at)
    spark.sql(f"CREATE TABLE IF NOT EXISTS {table} ({EVAL_COLUMNS}) COMMENT 'xPts model evaluation: one row per "
              "model x season x subset x segment. Lower MAE/RMSE is better; higher correlation and top-k actual "
              "points is better. Written by src/ml/evaluate_baselines.py and src/ml/train_xpts.py.'")
    models = ", ".join(f"'{m}'" for m in res["model"].unique())
    spark.sql(f"DELETE FROM {table} WHERE model IN ({models})")
    spark.createDataFrame(res).write.mode("append").saveAsTable(table)
