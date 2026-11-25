# Databricks notebook source
# MAGIC %md
# MAGIC # Fetch FPL API → landing Volume
# MAGIC
# MAGIC Downloads the public FPL endpoints and saves each response **unchanged** as a timestamped JSON file:
# MAGIC
# MAGIC `/Volumes/<catalog>/raw/landing/<endpoint>/dt=YYYY-MM-DD/<endpoint>_YYYYMMDDTHHMMSSZ.json`
# MAGIC
# MAGIC Why keep raw files? FPL only shows *current* news, prices and injury status. Saving every snapshot is
# MAGIC the only way to build the history that the news-signal backtest needs. Cleaning happens later (bronze/silver).

# COMMAND ----------

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

dbutils.widgets.text("catalog", "fpl")
CATALOG = dbutils.widgets.get("catalog")

BASE_URL = "https://fantasy.premierleague.com/api"
LANDING = Path(f"/Volumes/{CATALOG}/raw/landing")

# Endpoints captured every run. Per-player history (element-summary) is added later,
# on a slower schedule, so we don't hit the API ~700 times every 6 hours.
ENDPOINTS = {
    "bootstrap_static": "bootstrap-static/",
    "fixtures": "fixtures/",
}

session = requests.Session()
session.headers.update({"User-Agent": "fpl-agent-portfolio/0.1 (personal, non-commercial)"})

# COMMAND ----------

def fetch_json(path: str, attempts: int = 3) -> bytes:
    """GET an endpoint, retrying politely on errors. Returns raw bytes (validated as JSON)."""
    for attempt in range(1, attempts + 1):
        try:
            resp = session.get(f"{BASE_URL}/{path}", timeout=30)
            resp.raise_for_status()
            json.loads(resp.content)  # fail fast if FPL returned an HTML error page
            return resp.content
        except Exception as exc:
            if attempt == attempts:
                raise
            wait = 10 * attempt
            print(f"  attempt {attempt} failed ({exc!r}); retrying in {wait}s")
            time.sleep(wait)


run_ts = datetime.now(timezone.utc)
stamp = run_ts.strftime("%Y%m%dT%H%M%SZ")
day = run_ts.strftime("%Y-%m-%d")

written = []
for name, path in ENDPOINTS.items():
    body = fetch_json(path)
    out_dir = LANDING / name / f"dt={day}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{name}_{stamp}.json"
    out_file.write_bytes(body)
    written.append((name, out_file, len(body)))
    print(f"saved {name}: {out_file} ({len(body):,} bytes)")
    time.sleep(2)  # be gentle with the API

# COMMAND ----------

# Quick sanity check so a broken download fails the job (and shows red in the Jobs UI).
bootstrap = json.loads((LANDING / "bootstrap_static" / f"dt={day}" / f"bootstrap_static_{stamp}.json").read_text())
n_players, n_teams = len(bootstrap["elements"]), len(bootstrap["teams"])
current_gw = next((e["id"] for e in bootstrap["events"] if e.get("is_current")), None)
assert n_players > 400 and n_teams == 20, f"unexpected payload: {n_players} players, {n_teams} teams"

summary = f"OK {stamp}: {n_players} players, {n_teams} teams, current GW={current_gw}, files={len(written)}"
print(summary)
dbutils.notebook.exit(summary)
