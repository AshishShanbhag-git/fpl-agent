# Databricks notebook source
# MAGIC %md
# MAGIC # Backfill past seasons from vaastav/Fantasy-Premier-League
# MAGIC
# MAGIC One-off (re-runnable) load of historical per-gameweek player data. The xPts model trains on it.
# MAGIC Source: https://github.com/vaastav/Fantasy-Premier-League (community dataset, `data/<season>/...`).
# MAGIC
# MAGIC For each season it downloads three CSVs and saves them to the landing Volume **as JSON lines**:
# MAGIC * `gws/merged_gw.csv`: one row per player per fixture (points, minutes, xG, price, transfers...)
# MAGIC * `players_raw.csv`: id → code (stable cross-season player id), web_name, position
# MAGIC * `teams.csv`: id → team name
# MAGIC
# MAGIC Why convert CSV → JSON lines? Column sets differ between seasons (e.g. xG only from 2022-23,
# MAGIC defensive contribution from 2025-26). JSON matches columns **by name**, so mixing seasons is safe;
# MAGIC CSV would match by position and silently shift columns.

# COMMAND ----------

import csv
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

dbutils.widgets.text("catalog", "fpl")
dbutils.widgets.text("seasons", "2020-21,2021-22,2022-23,2023-24,2024-25,2025-26")
CATALOG = dbutils.widgets.get("catalog")
SEASONS = [s.strip() for s in dbutils.widgets.get("seasons").split(",") if s.strip()]

BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
LANDING = Path(f"/Volumes/{CATALOG}/raw/landing")

# landing folder name -> file inside data/<season>/
FILES = {
    "vaastav_merged_gw": "gws/merged_gw.csv",
    "vaastav_players": "players_raw.csv",
    "vaastav_teams": "teams.csv",
}
# players_raw.csv has ~100 columns; keep only what we need to map ids.
KEEP_COLUMNS = {
    "vaastav_players": ["id", "code", "web_name", "first_name", "second_name", "element_type", "team"],
    "vaastav_teams": ["id", "code", "name", "short_name"],
}

session = requests.Session()
session.headers.update({"User-Agent": "fpl-agent-portfolio/0.1 (personal, non-commercial)"})

# COMMAND ----------

def csv_to_jsonl(text: str, keep: list[str] | None) -> tuple[str, int]:
    """Convert CSV text to JSON lines (one object per row), optionally keeping only some columns."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))  # strip a UTF-8 BOM if present
    lines = []
    for row in reader:
        if keep:
            row = {k: row.get(k) for k in keep}
        lines.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(lines) + "\n", len(lines)


stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
summary = []
for season in SEASONS:
    for name, rel_path in FILES.items():
        url = f"{BASE}/{season}/{rel_path}"
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        # vaastav files are mostly UTF-8, but a few older ones are Latin-1.
        try:
            text = resp.content.decode("utf-8")
        except UnicodeDecodeError:
            text = resp.content.decode("latin-1")
        body, n_rows = csv_to_jsonl(text, KEEP_COLUMNS.get(name))
        out_dir = LANDING / name / f"season={season}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{name}_{stamp}.json"
        out_file.write_text(body, encoding="utf-8")
        summary.append(f"{season} {name}: {n_rows:,} rows")
        print(summary[-1])
        time.sleep(1)  # be polite to GitHub

msg = f"OK {stamp}: " + "; ".join(summary)
dbutils.notebook.exit(msg)
