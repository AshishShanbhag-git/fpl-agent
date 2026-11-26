# Databricks notebook source
# MAGIC %md
# MAGIC # Optimiser: this gameweek's transfers, chip and lineup for one FPL team
# MAGIC
# MAGIC 1. Reads the team's real squad, bank, free transfers, chips left and selling prices from FPL's public API.
# MAGIC 2. Reads xPts for the next gameweeks from `gold.xpts`, adjusted for injuries / suspensions (live FPL status).
# MAGIC 3. Solves an integer programme over the horizon (`src/optimiser/solver.py`), with and without each chip.
# MAGIC 4. Recommends only the **first** gameweek's moves (re-plan every week) and appends them to
# MAGIC    `gold.recommendations`.
# MAGIC
# MAGIC Run: `databricks bundle run fpl_optimise --params team_id=<your FPL team id>`.
# MAGIC The team id is a job parameter, not stored in the (public) repo.
# MAGIC
# MAGIC Limitation: FPL only shows a team's picks publicly after each deadline, so transfers you've already made for
# MAGIC the coming gameweek aren't visible. Run this before making transfers.

# COMMAND ----------

# MAGIC %pip install pulp==2.9.0 --quiet

# COMMAND ----------

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

dbutils.widgets.text("catalog", "fpl")
dbutils.widgets.text("bundle_root", "")
dbutils.widgets.text("team_id", "")
dbutils.widgets.text("free_transfers", "")      # blank = work it out from the team's history
dbutils.widgets.text("horizon", "5")
CATALOG = dbutils.widgets.get("catalog")
TEAM_ID = dbutils.widgets.get("team_id").strip()
FT_OVERRIDE = dbutils.widgets.get("free_transfers").strip()
HORIZON = int(dbutils.widgets.get("horizon"))
if not TEAM_ID.isdigit():
    raise ValueError("set the team_id parameter (the number in fantasy.premierleague.com/entry/<id>/event/...)")

ROOT = Path(dbutils.widgets.get("bundle_root") or Path.cwd().parents[1])
sys.path.insert(0, str(ROOT / "src" / "optimiser"))
import fpl_entry as E  # noqa: E402
import solver as S  # noqa: E402

API = "https://fantasy.premierleague.com/api"
session = requests.Session()
session.headers.update({"User-Agent": "fpl-agent-portfolio/0.1 (personal, non-commercial)"})


def get(path):
    r = session.get(f"{API}/{path}", timeout=30)
    r.raise_for_status()
    return r.json()


boot = get("bootstrap-static/")          # live prices + injury status (prices change daily ~01:30 UK)
history = get(f"entry/{TEAM_ID}/history/")
transfers = get(f"entry/{TEAM_ID}/transfers/")
picks = get(f"entry/{TEAM_ID}/event/{E.squad_event(history)}/picks/")

elements = {e["id"]: e for e in boot["elements"]}
now_cost = {p: e["now_cost"] for p, e in elements.items()}
start_price = {p: e["now_cost"] - e["cost_change_start"] for p, e in elements.items()}
state = E.starting_state(history, picks, transfers, now_cost, start_price)
if FT_OVERRIDE:
    state["free_transfers"] = int(FT_OVERRIDE)
next_gw = next(ev["id"] for ev in boot["events"] if ev["is_next"])
assert next_gw == state["next_gw"], f"team history ends at GW{state['next_gw'] - 1} but the next GW is {next_gw}"
GWS = list(range(next_gw, min(next_gw + HORIZON, 39)))
chips = E.chips_available(history, next_gw)
print(f"GW{next_gw}: bank £{state['bank'] / 10:.1f}m, {state['free_transfers']} free transfer(s), chips {chips}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## xPts with an availability adjustment
# MAGIC The model doesn't know about injuries (FPL status history only starts this season). So we scale its xPts by
# MAGIC FPL's live `chance_of_playing_next_round` (or by status if that's blank: doubtful 50%, injured/suspended 0%)
# MAGIC for the next gameweek, half-way back to normal the week after, and normal from the third week.
# MAGIC Players who have left the club (`u`) or are unavailable (`n`) stay at 0.

# COMMAND ----------

season = spark.sql(f"SELECT first(season) FROM {CATALOG}.gold.xpts").first()[0]
xp_rows = spark.sql(f"""
  SELECT player_id, gameweek, n_fixtures, xpts FROM {CATALOG}.gold.xpts
  WHERE season = '{season}' AND gameweek BETWEEN {GWS[0]} AND {GWS[-1]}
""").collect()
if not xp_rows:
    raise RuntimeError(f"no xPts for GW{GWS[0]}-{GWS[-1]} in gold.xpts: run fpl_ingest first")
xpts = {}
for r in xp_rows:
    xpts.setdefault(r.player_id, {})[r.gameweek] = 0.0 if r.n_fixtures == 0 else float(r.xpts)

STATUS_FACTOR = {"a": 1.0, "d": 0.5, "i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0}


def availability(e, i):
    cop = e["chance_of_playing_next_round"]
    f0 = cop / 100 if cop is not None else STATUS_FACTOR.get(e["status"], 1.0)
    if e["status"] in ("u", "n"):
        return f0
    return f0 + (1 - f0) * min(i, 2) / 2


POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
TEAMS = {t["id"]: t["short_name"] for t in boot["teams"]}
players = {}
for p, e in elements.items():
    if p not in xpts and p not in state["squad"]:
        continue
    players[p] = {"name": e["web_name"], "position": POS[e["element_type"]], "team": e["team"],
                  "price": e["now_cost"],
                  "xpts": {t: xpts.get(p, {}).get(t, 0.0) * availability(e, i) for i, t in enumerate(GWS)}}
pool = S.prune(players, state["squad"])
print(f"{len(players)} players with xPts; {len(pool)} in the solver's pool")

# COMMAND ----------

started = datetime.now(timezone.utc)
plan, chip_evals, base = S.plan_with_chips(pool, state["squad"], state["selling_price"], state["bank"],
                                     state["free_transfers"], GWS, chips, E.last_gw_for_chips(next_gw))
print(f"solved in {(datetime.now(timezone.utc) - started).total_seconds():.0f}s")

# Baseline for comparison: keep the squad, no transfers, best lineup.
keep = S.solve(pool, state["squad"], state["selling_price"], state["bank"], 0, GWS[:1])["gameweeks"][0]

# COMMAND ----------

# MAGIC %md ## Recommendation

# COMMAND ----------

g = plan["gameweeks"][0]
name = lambda p: f"{players[p]['name']} ({TEAMS[players[p]['team']]})"   # noqa: E731
by_pos = lambda ps: sorted(ps, key=lambda p: list(S.SQUAD).index(players[p]["position"]))   # noqa: E731


def transfer_lines(gw):
    if not gw["transfers_in"]:
        return [f"  No transfers: save the free transfer(s) ({state['free_transfers']} available)"]
    out = [f"  OUT {name(o)} £{state['selling_price'].get(o, players[o]['price']) / 10:.1f}m"
           f"  ->  IN {name(i)} £{players[i]['price'] / 10:.1f}m"
           for o, i in zip(by_pos(gw["transfers_out"]), by_pos(gw["transfers_in"]))]   # same count per position
    return out + [f"  hits: {gw['hits']} (-{4 * gw['hits']} pts), bank after: £{gw['bank'] / 10:.1f}m"]


def later_weeks(p):
    return "; ".join(
        f"GW{x['gameweek']}: " + (", ".join(f"{players[o]['name']}->{players[i]['name']}"
                                            for o, i in zip(by_pos(x["transfers_out"]), by_pos(x["transfers_in"])))
                                  or "no transfers")
        + (f" [{S.CHIP_NAMES[x['chip']]}]" if x["chip"] else "") for x in p["gameweeks"][1:])


lines = [f"GW{g['gameweek']} for team {TEAM_ID}: {state['free_transfers']} free transfer(s), "
         f"bank £{state['bank'] / 10:.1f}m, chips left {[S.CHIP_NAMES[c] for c in chips]}"]
lines.append(f"CHIP: play {S.CHIP_NAMES[g['chip']]}" if g["chip"] else "CHIP: none this week")
lines += transfer_lines(g)
lines.append("XI: " + ", ".join(name(p) + (" (C)" if p == g["captain"] else " (V)" if p == g["vice_captain"] else "")
                                for p in g["starters"]))
lines.append("Bench (in order): " + ", ".join(name(p) for p in g["bench"]))
lines.append(f"Expected GW{g['gameweek']} points: {g['xpts']:.1f} ({g['xpts_after_hits']:.1f} after hits); "
             f"same squad without changes: {keep['xpts']:.1f}")
lines.append(f"Plan beyond this week (re-planned every week): {later_weeks(plan)}")
lines.append("Chips (gain in xPts over the horizon vs no chip):")
for e in chip_evals:
    later = (f", best later in horizon: GW{e['best_later_gw']} +{e['best_later_gain']:.1f}"
             if e["best_later_gw"] else "")
    verdict = "PLAY" if e["chosen"] else "worth it, but one chip per week" if e["qualifies"] else "keep"
    lines.append(f"  {S.CHIP_NAMES[e['chip']]}: +{e['gain_now']:.1f} now (bar {e['threshold']:.1f}){later} -> {verdict}")
if g["chip"]:
    b = base["gameweeks"][0]
    lines.append(f"Without a chip, the best GW{b['gameweek']} moves would be ({b['xpts_after_hits']:.1f} xPts after "
                 f"hits; {plan['objective'] - base['objective']:.1f} fewer over the horizon):")
    lines += transfer_lines(b)
    lines.append(f"  captain {name(b['captain'])}; later: {later_weeks(base)}")
print("\n".join(lines))

# COMMAND ----------

names = lambda ps: [name(p) for p in ps]   # noqa: E731
row = {
    "planned_at": datetime.now(timezone.utc), "team_id": int(TEAM_ID), "season": season, "gameweek": g["gameweek"],
    "chip": g["chip"], "transfers_out": names(g["transfers_out"]), "transfers_in": names(g["transfers_in"]),
    "hits": g["hits"], "free_transfers": state["free_transfers"], "bank_after": g["bank"] / 10,
    "starters": names(g["starters"]), "captain": name(g["captain"]), "vice_captain": name(g["vice_captain"]),
    "bench": names(g["bench"]), "xpts": g["xpts"], "xpts_after_hits": g["xpts_after_hits"],
    "xpts_no_changes": keep["xpts"], "model_version": spark.sql(
        f"SELECT first(model_version) FROM {CATALOG}.gold.xpts").first()[0],
    "plan_json": json.dumps({"gameweeks": plan["gameweeks"], "names": {p: players[p]["name"] for p in pool}}),
    "chip_evaluations_json": json.dumps(chip_evals),
    "summary": "\n".join(lines),
}
SCHEMA = ("planned_at timestamp, team_id bigint, season string, gameweek int, chip string, "
          "transfers_out array<string>, transfers_in array<string>, hits int, free_transfers int, bank_after double, "
          "starters array<string>, captain string, vice_captain string, bench array<string>, xpts double, "
          "xpts_after_hits double, xpts_no_changes double, model_version string, plan_json string, "
          "chip_evaluations_json string, summary string")
values = tuple(row[col.split(" ")[0]] for col in SCHEMA.split(", "))
(spark.createDataFrame([values], SCHEMA).write.mode("append")
    .option("comment", "One row per optimiser run: recommended transfers, chip and lineup for one FPL team's next "
                       "gameweek. Written by src/optimiser/plan_gameweek.py.")
    .saveAsTable(f"{CATALOG}.gold.recommendations"))
dbutils.notebook.exit(row["summary"])
