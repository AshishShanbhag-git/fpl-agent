"""The xPts model as one MLflow object: four LightGBM boosters (one per position) behind a single predict().

Logged with `code_paths=[this file]`, so anything that loads `models:/fpl.ml.xpts@champion` gets this class.
Boosters are stored as LightGBM's own text format (not pickled objects), so they reload safely.
"""
from __future__ import annotations

import lightgbm as lgb
import mlflow.pyfunc
import pandas as pd

POSITIONS = ("GKP", "DEF", "MID", "FWD")


class XptsModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_strings: dict[str, str], features: list[str]):
        self.model_strings = model_strings      # position -> LightGBM model text
        self.features = list(features)
        self._boosters = None

    def __getstate__(self):                     # save only the text models, never live booster objects
        return {**self.__dict__, "_boosters": None}

    def _load(self):
        if self._boosters is None:
            self._boosters = {p: lgb.Booster(model_str=s) for p, s in self.model_strings.items()}
        return self._boosters

    def load_context(self, context):
        self._load()

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        boosters = self._load()
        x = model_input[self.features].apply(pd.to_numeric, errors="coerce").astype("float64")
        out = pd.Series(float("nan"), index=model_input.index)
        for pos, booster in boosters.items():
            mask = (model_input["position"] == pos).to_numpy()
            if mask.any():
                out[mask] = booster.predict(x[mask])
        return pd.DataFrame({"xpts": out.to_numpy()})
