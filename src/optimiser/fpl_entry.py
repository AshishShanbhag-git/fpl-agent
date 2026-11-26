"""Turn a manager's public FPL API data into the optimiser's starting state.

Pure Python (no Spark, no network) so it can be unit tested. The notebook fetches these public endpoints:
  bootstrap-static/                   -> players' current price (now_cost) and start price (now_cost - cost_change_start)
  entry/{id}/history/                 -> per-gameweek bank, transfers made, chips played
  entry/{id}/transfers/               -> purchase price of every player bought
  entry/{id}/event/{gw}/picks/        -> the 15 players in the squad at that gameweek's deadline

All money is in FPL's own unit: tenths of £1m (now_cost 105 = £10.5m), so there is no rounding error.

Rules used (FPL 2025-26 rules; check each season):
* Selling price: you keep half of any price rise, rounded down to £0.1m; you take the full fall.
* Free transfers: +1 per gameweek, at most 5 banked. Playing a Wildcard or Free Hit keeps the banked number.
* Chips: two sets of Wildcard / Free Hit / Bench Boost / Triple Captain. Set 1 is for GW1-19, set 2 for GW20-38.
  One chip per gameweek.
"""

MAX_FREE_TRANSFERS = 5
FIRST_HALF_LAST_GW = 19
CHIPS = ("wildcard", "freehit", "bboost", "3xc")


def selling_price(purchase, now):
    """FPL selling price in tenths: half the profit (rounded down), all of any loss."""
    if now <= purchase:
        return now
    return purchase + (now - purchase) // 2


def chips_by_event(history):
    return {c["event"]: c["name"] for c in history.get("chips", [])}


def squad_event(history):
    """The gameweek whose picks are the squad going forward (a Free Hit squad reverts, so use the GW before it)."""
    last = max(e["event"] for e in history["current"])
    return last - 1 if chips_by_event(history).get(last) == "freehit" else last


def bank(history):
    """Money in the bank (tenths) for the next gameweek. After a Free Hit the bank reverts too."""
    gw = squad_event(history)
    return next(e["bank"] for e in history["current"] if e["event"] == gw)


def free_transfers(history):
    """Free transfers available for the next gameweek, replayed from the season's history."""
    events = sorted(history["current"], key=lambda e: e["event"])
    chips = chips_by_event(history)
    ft = None
    for e in events:
        gw = e["event"]
        if ft is None:                       # the gameweek the team was created: squad is free, then 1 FT
            ft = 1
        elif chips.get(gw) in ("wildcard", "freehit"):
            pass                             # banked free transfers are kept
        else:
            ft = min(MAX_FREE_TRANSFERS, max(ft - e["event_transfers"], 0) + 1)
    return ft


def purchase_prices(squad, transfers, history, start_price):
    """Purchase price (tenths) of each squad player: the latest time it was bought, else its season-start price.

    Transfers made in a Free Hit gameweek are ignored (that squad reverted afterwards).
    start_price: {player_id: now_cost - cost_change_start}, for players kept since the team was created.
    """
    fh = {gw for gw, chip in chips_by_event(history).items() if chip == "freehit"}
    latest = {}
    for t in sorted(transfers, key=lambda t: (t["event"], t["time"])):
        if t["event"] not in fh:
            latest[t["element_in"]] = t["element_in_cost"]
    return {p: latest.get(p, start_price[p]) for p in squad}


def chips_available(history, gw):
    """Chips that can still be played in gameweek `gw` (each set covers one half of the season)."""
    first_half = gw <= FIRST_HALF_LAST_GW
    used = {name for ev, name in chips_by_event(history).items() if (ev <= FIRST_HALF_LAST_GW) == first_half}
    return [c for c in CHIPS if c not in used]


def last_gw_for_chips(gw):
    """Last gameweek in which the chips available at `gw` can be played."""
    return FIRST_HALF_LAST_GW if gw <= FIRST_HALF_LAST_GW else 38


def starting_state(history, picks, transfers, now_cost, start_price):
    """Everything the solver needs about the manager's team.

    picks: the picks JSON for squad_event(history). now_cost / start_price: {player_id: tenths}.
    """
    squad = [p["element"] for p in picks["picks"]]
    bought = purchase_prices(squad, transfers, history, start_price)
    return {
        "squad": squad,
        "selling_price": {p: selling_price(bought[p], now_cost[p]) for p in squad},
        "purchase_price": bought,
        "bank": bank(history),
        "free_transfers": free_transfers(history),
        "next_gw": max(e["event"] for e in history["current"]) + 1,
    }
