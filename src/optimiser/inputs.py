"""Build the solver's inputs for one FPL team. Shared by the fpl-optimise notebook and the agent's tools.

`get(path)` fetches an FPL API path (e.g. "bootstrap-static/") and returns JSON; it's injected so this module has
no network code of its own and can be tested with fake data.
"""

import fpl_entry as E

POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
# chance_of_playing_next_round is often blank; then use the status letter: a=available, d=doubtful, i=injured,
# s=suspended, u=unavailable (left the club / loan), n=not available.
STATUS_FACTOR = {"a": 1.0, "d": 0.5, "i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0}


def fetch_team(get, team_id):
    """Live bootstrap + the team's history, transfers and current squad picks -> (boot, state, chips, next_gw)."""
    boot = get("bootstrap-static/")
    history = get(f"entry/{team_id}/history/")
    transfers = get(f"entry/{team_id}/transfers/")
    picks = get(f"entry/{team_id}/event/{E.squad_event(history)}/picks/")
    elements = {e["id"]: e for e in boot["elements"]}
    now_cost = {p: e["now_cost"] for p, e in elements.items()}
    start_price = {p: e["now_cost"] - e["cost_change_start"] for p, e in elements.items()}
    state = E.starting_state(history, picks, transfers, now_cost, start_price)
    next_gw = next(ev["id"] for ev in boot["events"] if ev["is_next"])
    if next_gw != state["next_gw"]:
        raise RuntimeError(f"team history ends at GW{state['next_gw'] - 1} but the next GW is {next_gw}")
    state["chips"] = E.chips_available(history, next_gw)
    state["last_chip_gw"] = E.last_gw_for_chips(next_gw)
    return boot, state


def availability(element, i):
    """Multiplier for xPts i gameweeks ahead (0 = next GW): FPL's chance of playing, recovering over two weeks."""
    cop = element["chance_of_playing_next_round"]
    f0 = cop / 100 if cop is not None else STATUS_FACTOR.get(element["status"], 1.0)
    if element["status"] in ("u", "n"):
        return f0
    return f0 + (1 - f0) * min(i, 2) / 2


def build_players(boot, xpts, gws, squad):
    """Solver player dict from live bootstrap + model xPts {player_id: {gw: xpts}} (blank GWs already 0)."""
    players = {}
    for e in boot["elements"]:
        p = e["id"]
        if p not in xpts and p not in squad:
            continue
        players[p] = {"name": e["web_name"], "position": POS[e["element_type"]], "team": e["team"],
                      "price": e["now_cost"],
                      "xpts": {t: xpts.get(p, {}).get(t, 0.0) * availability(e, i) for i, t in enumerate(gws)}}
    return players
