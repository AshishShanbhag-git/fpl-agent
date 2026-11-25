# Databricks notebook source
# MAGIC %md
# MAGIC # Fetch per-player history (element-summary) → landing Volume, once a day
# MAGIC
# MAGIC `element-summary/{id}/` gives one player's match-by-match record **for the current season**
# MAGIC (points, minutes, xG, bonus, price and transfers per fixture). That's one request per player (~700),
# MAGIC so this runs **at most once per UTC day**: the first job run of the day fetches, later runs skip.
# MAGIC
# MAGIC Output: one JSON-lines file per day,
# MAGIC `/Volumes/<catalog>/raw/landing/element_summary/dt=YYYY-MM-DD/element_summary_<ts>.json`,
# MAGIC one line per player: `{"player_id": ..., "history": [...], "fixtures": [...], "history_past": [...]}`.

# COMMAND ----------

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

dbutils.widgets.text("catalog", "fpl")
dbutils.widgets.text("force", "false")           # "true" = fetch even if today's file already exists
dbutils.widgets.text("delay_seconds", "0.5")     # pause between requests, to be gentle with the API
CATALOG = dbutils.widgets.get("catalog")
FORCE = dbutils.widgets.get("force").lower() == "true"
DELAY = float(dbutils.widgets.get("delay_seconds"))

BASE_URL = "https://fantasy.premierleague.com/api"
LANDING = Path(f"/Volumes/{CATALOG}/raw/landing")
OUT_ROOT = LANDING / "element_summary"
MAX_FAILURE_RATE = 0.05

run_ts = datetime.now(timezone.utc)
stamp, day = run_ts.strftime("%Y%m%dT%H%M%SZ"), run_ts.strftime("%Y-%m-%d")
out_dir = OUT_ROOT / f"dt={day}"

if out_dir.exists() and any(out_dir.glob("element_summary_*.json")) and not FORCE:
    dbutils.notebook.exit(f"SKIPPED: already fetched today ({day})")

# COMMAND ----------

# Player ids come from the newest bootstrap-static snapshot the previous task just saved,
# so we don't need an extra API call.
latest_bootstrap = max((LANDING / "bootstrap_static").glob("dt=*/bootstrap_static_*.json"), key=lambda p: p.name)
player_ids = sorted(e["id"] for e in json.loads(latest_bootstrap.read_text())["elements"])
print(f"{len(player_ids)} players from {latest_bootstrap.name}")

session = requests.Session()
session.headers.update({"User-Agent": "fpl-agent-portfolio/0.1 (personal, non-commercial)"})


def fetch(player_id: int, attempts: int = 3) -> dict:
    for attempt in range(1, attempts + 1):
        try:
            resp = session.get(f"{BASE_URL}/element-summary/{player_id}/", timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(5 * attempt)


lines, failures = [], []
for i, pid in enumerate(player_ids, 1):
    try:
        lines.append(json.dumps({"player_id": pid, **fetch(pid)}, ensure_ascii=False))
    except Exception as exc:
        failures.append((pid, repr(exc)))
    if i % 100 == 0:
        print(f"  {i}/{len(player_ids)} done, {len(failures)} failed")
    time.sleep(DELAY)

# COMMAND ----------

failure_rate = len(failures) / len(player_ids)
if failure_rate > MAX_FAILURE_RATE:
    raise RuntimeError(f"{len(failures)} of {len(player_ids)} players failed (first: {failures[:3]})")

# Write to a temp name first, then rename, so the pipeline never sees a half-written file.
out_dir.mkdir(parents=True, exist_ok=True)
tmp = out_dir / f"_tmp_{stamp}"
tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
final = out_dir / f"element_summary_{stamp}.json"
tmp.rename(final)

msg = f"OK {stamp}: {len(lines)} players saved, {len(failures)} failed -> {final}"
print(msg)
dbutils.notebook.exit(msg)
