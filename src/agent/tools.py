"""The agent's tools: every number the agent says comes from one of these, never from the LLM itself.

Each tool is a plain Python method that returns a small JSON-able dict (small = cheap and easy for the LLM to read).
TOOL_SPECS describes them to the LLM in the OpenAI "function calling" format that Databricks model serving accepts.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimiser"))
import inputs as I  # noqa: E402
import solver as S  # noqa: E402

POSITIONS = ["GKP", "DEF", "MID", "FWD"]
CHIPS = ["wildcard", "freehit", "bboost", "3xc"]


def _fn(name, description, properties=None, required=()):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": list(required)}}}


TOOL_SPECS = [
    _fn("get_my_squad", "The user's current FPL squad (15 players) with price, selling price, availability and xPts, "
        "plus bank, free transfers and chips left. Read live from the FPL API."),
    _fn("get_recommendation", "The latest saved optimiser recommendation for the user's team (transfers, chip, XI, "
        "captain, bench, chip analysis). Fast. Use it for 'what should I do this week?'."),
    _fn("run_optimizer", "Re-run the transfer/lineup optimiser for the user's team, optionally with what-if "
        "constraints. Use for 'what if I keep X / never buy Y / have N free transfers / play chip Z'. "
        "chip='none' (default, ~5s) plans without chips; 'auto' also evaluates every chip (~1 min).",
        {"must_have": {"type": "array", "items": {"type": "string"}, "description": "player names to keep/own"},
         "exclude": {"type": "array", "items": {"type": "string"}, "description": "player names to sell/avoid"},
         "chip": {"type": "string", "enum": ["none", "auto"] + CHIPS},
         "free_transfers": {"type": "integer", "description": "override the free transfers available"},
         "horizon": {"type": "integer", "description": "gameweeks to plan over (1-5, default 5)"}}),
    _fn("get_player", "Details for one player: team, position, price, injury status and news, form, season stats, "
        "ownership, xPts for each upcoming gameweek and the next fixtures.",
        {"name": {"type": "string"}}, ["name"]),
    _fn("top_players", "Players ranked by xPts, optionally filtered by position, maximum price (£m) or team.",
        {"position": {"type": "string", "enum": POSITIONS},
         "max_price": {"type": "number"},
         "team": {"type": "string", "description": "3-letter team code, e.g. ARS"},
         "span": {"type": "string", "enum": ["next_gw", "horizon"],
                  "description": "rank by next gameweek or by the sum over the next 5 (default horizon)"},
         "limit": {"type": "integer"}}),
    _fn("get_fixtures", "A team's next fixtures with opponent, home/away and FPL difficulty (1 easy - 5 hard).",
        {"team": {"type": "string", "description": "3-letter team code, e.g. LIV"}, "count": {"type": "integer"}},
        ["team"]),
    _fn("get_news", "Recent changes to players' injury/suspension status and FPL news, newest first. "
        "Optionally for one player. Only cite injuries that appear here or in get_player.",
        {"player": {"type": "string"}, "days": {"type": "integer"}}),
]


class FplTools:
    def __init__(self, data, team_id=None, fpl_get=None):
        """data: FplData. fpl_get(path) -> JSON from the FPL API (needed by the squad/optimiser tools)."""
        self.data, self.team_id, self.fpl_get = data, team_id, fpl_get
        self._team = None
        x = data.xpts
        self.xp = {}
        for r in x.itertuples():
            self.xp.setdefault(int(r.player_id), {})[int(r.gameweek)] = 0.0 if r.n_fixtures == 0 else float(r.xpts)
        self.gws = sorted({g for d in self.xp.values() for g in d})
        self.by_id = data.players.set_index("player_id")
        self.team_names = dict(zip(data.players["team_short_name"], data.players["team_name"]))

    # ---- dispatch -------------------------------------------------------------------------------------------
    def call(self, name, arguments):
        """Run tool `name` with a JSON string or dict of arguments; errors are returned to the LLM, not raised."""
        try:
            args = json.loads(arguments or "{}") if isinstance(arguments, str) else (arguments or {})
            if name not in {t["function"]["name"] for t in TOOL_SPECS}:
                return {"error": f"unknown tool {name}"}
            return getattr(self, name)(**args)
        except Exception as e:             # noqa: BLE001 - the model gets the message and can try again
            return {"error": f"{type(e).__name__}: {e}"}

    # ---- helpers --------------------------------------------------------------------------------------------
    def _resolve(self, name):
        rows = self.data.find_player(name)
        if rows.empty:
            raise ValueError(f"no player called '{name}'")
        if len(rows) > 1:
            # prefer the most relevant (most xPts) but tell the model there were others
            rows = rows.assign(_x=rows["player_id"].map(lambda p: sum(self.xp.get(p, {}).values())))
            rows = rows.sort_values("_x", ascending=False)
        return rows.iloc[0], [r.web_name + f" ({r.team_short_name})" for r in rows.iloc[1:6].itertuples()]

    def _xpts(self, pid):
        return {f"GW{g}": round(v, 2) for g, v in sorted(self.xp.get(pid, {}).items())}

    def _brief(self, pid):
        p = self.by_id.loc[pid]
        xp = self.xp.get(pid, {})
        out = {"name": p.web_name, "team": p.team_short_name, "position": p.position, "price": round(p.price, 1),
               "xpts_next_gw": round(xp.get(self.gws[0], 0.0), 2) if self.gws else None,
               "xpts_next_5": round(sum(xp.values()), 2)}
        if p.status != "a":
            out.update(status=p.status, chance_of_playing=p.chance_of_playing_next_round, news=p.news)
        return out

    def _fixtures(self, team_id, count=5):
        f = self.data.fixtures
        f = f[(~f["finished"]) & ((f["home_team_id"] == team_id) | (f["away_team_id"] == team_id))]
        out = []
        for r in f.sort_values(["gameweek", "kickoff_time"]).head(count).itertuples():
            home = r.home_team_id == team_id
            opp = r.away if home else r.home
            out.append({"gw": int(r.gameweek), "opponent": self.team_names.get(opp, opp), "venue": "H" if home else "A",
                        "difficulty": int(r.home_difficulty if home else r.away_difficulty)})
        return out

    def _need_team(self):
        if not self.team_id:
            raise ValueError("no FPL team id set: ask the user for it")
        if self.fpl_get is None:
            raise ValueError("FPL API access is not configured")
        if self._team is None:
            self._team = I.fetch_team(self.fpl_get, self.team_id)
        return self._team

    # ---- tools ----------------------------------------------------------------------------------------------
    def get_my_squad(self):
        boot, st = self._need_team()
        squad = []
        for pid in st["squad"]:
            b = self._brief(pid) if pid in self.by_id.index else {"player_id": pid}
            b["selling_price"] = st["selling_price"][pid] / 10
            squad.append(b)
        squad.sort(key=lambda b: POSITIONS.index(b.get("position", "FWD")))
        return {"gameweek": st["next_gw"], "bank": st["bank"] / 10, "free_transfers": st["free_transfers"],
                "chips_left": [S.CHIP_NAMES[c] for c in st["chips"]], "chip_deadline_gw": st["last_chip_gw"],
                "squad": squad, "note": "xPts here are the model's, before the injury adjustment the optimiser uses"}

    def _card(self, pid, gw):
        """One player in a plan: everything the LLM may want to quote, so it never has to make anything up."""
        p = self.by_id.loc[pid]
        fx = [f for f in self._fixtures(int(p.team_id), 8) if f["gw"] == gw]
        out = {"name": p.web_name, "team": p.team_name, "position": p.position, "price": round(float(p.price), 1),
               f"xpts_gw{gw}": round(self.xp.get(pid, {}).get(gw, 0.0), 2),
               "xpts_next_5": round(sum(self.xp.get(pid, {}).values()), 2),
               f"gw{gw}_fixtures": [f"{f['opponent']} ({f['venue']}, difficulty {f['difficulty']})" for f in fx]
               or ["no fixture (blank)"]}
        if p.status != "a":
            out.update(status=p.status, chance_of_playing=None if pd.isna(p.chance_of_playing_next_round)
                       else int(p.chance_of_playing_next_round), news=p.news)
        return out

    def _describe(self, gws_plan, sell_prices=None):
        """Plan (solver output, player ids) -> structured, self-explanatory dict for the LLM."""
        g = gws_plan[0]
        gw = g["gameweek"]
        by_pos = lambda ps: sorted(ps, key=lambda p: POSITIONS.index(self.by_id.loc[p].position))   # noqa: E731
        name = lambda p: self.by_id.loc[p].web_name   # noqa: E731
        transfers = []
        for o, i in zip(by_pos(g["transfers_out"]), by_pos(g["transfers_in"])):   # same count per position
            out_c, in_c = self._card(o, gw), self._card(i, gw)
            if sell_prices and o in sell_prices:
                out_c["selling_price"] = sell_prices[o] / 10
            transfers.append({"out": out_c, "in": in_c,
                              "xpts_next_5_gain": round(in_c["xpts_next_5"] - out_c["xpts_next_5"], 2)})
        return {
            "gameweek": gw, "chip": S.CHIP_NAMES.get(g["chip"]) if g["chip"] else None,
            "transfers": transfers, "hits": g["hits"], "points_cost_of_hits": 4 * g["hits"],
            "bank_after_transfers": g["bank"] / 10,
            "starting_xi": [self._card(p, gw) for p in by_pos(g["starters"])],
            "captain": name(g["captain"]), "vice_captain": name(g["vice_captain"]),
            "bench_in_order": [self._card(p, gw) for p in g["bench"]],
            "expected_points_this_gw": round(g["xpts"], 2),
            "expected_points_after_hits": round(g["xpts_after_hits"], 2),
            "later_gameweeks_tentative": [
                {"gw": x["gameweek"], "chip": S.CHIP_NAMES.get(x["chip"]) if x["chip"] else None,
                 "transfers": [f"{name(o)} -> {name(i)}" for o, i in
                               zip(by_pos(x["transfers_out"]), by_pos(x["transfers_in"]))] or "none"}
                for x in gws_plan[1:]],
        }

    def get_recommendation(self):
        if not self.team_id:
            raise ValueError("no FPL team id set: ask the user for it")
        row = self.data.recommendations(self.team_id) if self.data.recommendations else None
        if row is None:
            return {"error": "no saved recommendation for this team yet: use run_optimizer"}
        plan = json.loads(row["plan_json"])["gameweeks"]
        out = {"planned_at": str(row["planned_at"]), "free_transfers_available": int(row["free_transfers"]),
               **self._describe(plan),
               "expected_points_if_no_changes": round(float(row["xpts_no_changes"]), 2),
               "chip_evaluations": json.loads(row["chip_evaluations_json"]),
               "chip_note": "gain_now = extra discounted xPts over the 5-GW plan from playing the chip now vs no "
                            "chip; it must beat 'threshold' (value of keeping the chip) to be played."}
        if self.data.next_gw and int(row["gameweek"]) != self.data.next_gw:
            out["warning"] = f"this plan is for GW{row['gameweek']}, but the next GW is {self.data.next_gw}"
        elif str(row["planned_at"]) < str(self.data.xpts["scored_at"].max()):
            out["warning"] = "xPts were re-scored after this plan was made: consider run_optimizer"
        return out

    def run_optimizer(self, must_have=(), exclude=(), chip="none", free_transfers=None, horizon=5):
        boot, st = self._need_team()
        gws = [g for g in self.gws if g >= st["next_gw"]][:max(1, min(int(horizon), 5))]
        ids = lambda names: [int(self._resolve(n)[0]["player_id"]) for n in names]   # noqa: E731
        keep, avoid = ids(must_have), ids(exclude)
        players = I.build_players(boot, self.xp, gws, st["squad"])
        pool = S.prune(players, st["squad"], extra=keep)
        ft = st["free_transfers"] if free_transfers is None else int(free_transfers)
        args = (pool, st["squad"], st["selling_price"], st["bank"], ft, gws)
        kw = {"must_have": keep, "exclude": avoid}
        evals = None
        if chip == "auto":
            plan, evals, _ = S.plan_with_chips(*args, st["chips"], st["last_chip_gw"], **kw)
        elif chip in CHIPS:
            if chip not in st["chips"]:
                return {"error": f"{S.CHIP_NAMES[chip]} is not available (chips left: {st['chips']})"}
            plan = S.solve(*args, chip=chip, chip_gw=gws[0], **kw)
        else:
            plan = S.solve(*args, **kw)
        keep_xp = S.solve(pool, st["squad"], st["selling_price"], st["bank"], 0, gws[:1])["gameweeks"][0]["xpts"]
        out = {"free_transfers_available": ft, **self._describe(plan["gameweeks"], st["selling_price"]),
               "expected_points_if_no_changes": round(keep_xp, 2), "horizon_score": round(plan["objective"], 2),
               "constraints": {"must_have": [self.by_id.loc[p].web_name for p in keep],
                               "exclude": [self.by_id.loc[p].web_name for p in avoid]},
               "note": "horizon_score = discounted xPts over the whole plan minus hits; compare it between "
                       "what-ifs. Only this gameweek's moves are advice; later weeks are re-planned."}
        if evals is not None:
            out["chip_evaluations"] = evals
        return out

    def get_player(self, name):
        p, others = self._resolve(name)
        pid = int(p["player_id"])
        out = {"name": p.web_name, "full_name": f"{p.first_name} {p.second_name}", "team": p.team_name,
               "position": p.position, "price": round(p.price, 1), "status": p.status,
               "chance_of_playing_next_gw": p.chance_of_playing_next_round, "news": p.news or None,
               "form": p.form, "total_points": int(p.total_points), "minutes": int(p.minutes),
               "starts": int(p.starts), "goals": int(p.goals_scored), "assists": int(p.assists),
               "clean_sheets": int(p.clean_sheets), "selected_by_percent": p.selected_by_percent,
               "xpts": self._xpts(pid), "xpts_next_5": round(sum(self.xp.get(pid, {}).values()), 2),
               "fpl_official_ep_next": p.ep_next, "next_fixtures": self._fixtures(int(p.team_id))}
        if others:
            out["other_matches"] = others
        return out

    def top_players(self, position=None, max_price=None, team=None, span="horizon", limit=10):
        P = self.data.players
        if position:
            P = P[P["position"] == position]
        if max_price is not None:
            P = P[P["price"] <= float(max_price) + 1e-9]
        if team:
            P = P[P["team_short_name"].str.upper() == team.upper()]
        key = (lambda pid: self.xp.get(pid, {}).get(self.gws[0], 0.0)) if span == "next_gw" else \
              (lambda pid: sum(self.xp.get(pid, {}).values()))
        ranked = sorted(P["player_id"], key=lambda pid: -key(int(pid)))[:max(1, min(int(limit), 25))]
        return {"ranked_by": f"xPts {'GW' + str(self.gws[0]) if span == 'next_gw' else 'next 5 GWs'}",
                "players": [self._brief(int(pid)) for pid in ranked]}

    def get_fixtures(self, team, count=5):
        t = self.data.players[self.data.players["team_short_name"].str.upper() == team.upper()]
        if t.empty:
            return {"error": f"unknown team code {team}; use 3-letter codes like ARS, LIV, MCI"}
        return {"team": t["team_name"].iloc[0], "fixtures": self._fixtures(int(t["team_id"].iloc[0]), int(count))}

    def get_news(self, player=None, days=7):
        n = self.data.news
        n = n[n["observed_at"] >= n["observed_at"].max() - pd.Timedelta(days=int(days))]
        if player:
            p, _ = self._resolve(player)
            n = n[n["player_id"] == p["player_id"]]
        else:
            n = n[~n["is_first_observation"]]       # changes only, not the first snapshot of every player
        n = n.sort_values("observed_at", ascending=False).head(30)
        return {"changes": [{"player": r.web_name, "when": str(r.observed_at)[:16], "status": r.status,
                             "chance_of_playing": None if pd.isna(r.chance_of_playing_next_round)
                             else int(r.chance_of_playing_next_round), "news": r.news or None,
                             "previous_status": r.prev_status} for r in n.itertuples()]}
