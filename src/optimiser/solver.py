"""Transfer + chip + lineup optimiser: an integer linear programme (ILP) solved with PuLP's bundled CBC solver.

An ILP picks yes/no decisions (own player p in GW t? start him? captain him? buy/sell him?) that maximise a
linear score subject to linear rules (15 players, 2/5/5/3, max 3 per club, budget, valid formation...).
The solver proves the answer is the best possible under those rules, which a hand-written heuristic can't.

We plan over the next few gameweeks (the xPts horizon) but only the first gameweek's moves are recommended:
next week we re-plan with fresh data. Looking ahead stops the solver from, say, buying a player for one good
fixture and selling him (with a -4 hit) the week after.

Money is in tenths of £1m. Points are xPts (expected points) from fpl.gold.xpts.
"""

import pulp

SQUAD = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
MIN_START = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
MAX_START = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3

# A chip is played now only if it adds at least this many (discounted) xPts over the horizon, because using it
# now means not having it for a better week later in the same half-season. Roughly: what the chip is worth in an
# ordinary gameweek, so "worth it" means "better than average". plan_with_chips lowers it as the expiry nears.
KEEP_VALUE = {"wildcard": 12.0, "freehit": 8.0, "bboost": 10.0, "3xc": 6.0}
CHIP_NAMES = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}


def prune(players, owned, keep=None):
    """Shrink the pool so each solve takes seconds: owned players + the best and the cheapest per position."""
    keep = keep or {"GKP": 12, "DEF": 40, "MID": 45, "FWD": 25}
    chosen = set(owned)
    for pos, n in keep.items():
        group = [p for p, d in players.items() if d["position"] == pos]
        chosen |= set(sorted(group, key=lambda p: -sum(players[p]["xpts"].values()))[:n])
        chosen |= set(sorted(group, key=lambda p: (players[p]["price"], -sum(players[p]["xpts"].values())))[:4])
    return {p: players[p] for p in chosen}


def solve(players, squad, selling_price, bank, free_transfers, gws, chip=None, chip_gw=None,
          decay=0.9, bench_weight=0.1, ft_value=1.5, hit_cost=4, max_ft=5, time_limit=120):
    """Best plan over `gws` starting from `squad` (15 player ids).

    players: {id: {"name", "position", "team", "price" (tenths, buy price now), "xpts": {gw: xpts}}}
    selling_price: {id: tenths} for the 15 owned players. bank: tenths. free_transfers: FTs for gws[0].
    chip / chip_gw: play one of "wildcard", "freehit", "bboost", "3xc" in that gameweek.
    decay: weight of each later gameweek (forecasts get less reliable). bench_weight: value of a bench player's xPts
    (covers auto-subs). ft_value: xPts value of each free transfer still banked at the end (stops the solver
    spending free transfers on tiny gains in the last gameweek of the horizon).
    """
    P, owned = list(players), set(squad)
    sell = {p: selling_price[p] if p in owned else players[p]["price"] for p in P}
    xp = {(p, t): players[p]["xpts"].get(t, 0.0) for p in P for t in gws}
    teams = {d["team"] for d in players.values()}

    m = pulp.LpProblem("fpl_plan", pulp.LpMaximize)
    B = pulp.LpBinary
    x = pulp.LpVariable.dicts("own", (P, gws), cat=B)          # in the squad after GW t's transfers
    buy = pulp.LpVariable.dicts("buy", (P, gws), cat=B)
    sold = pulp.LpVariable.dicts("sell", (P, gws), cat=B)
    st = pulp.LpVariable.dicts("start", (P, gws), cat=B)
    cap = pulp.LpVariable.dicts("cap", (P, gws), cat=B)
    money = pulp.LpVariable.dicts("bank", gws, lowBound=0)
    hits = pulp.LpVariable.dicts("hits", gws, lowBound=0, cat="Integer")
    free = pulp.LpVariable.dicts("free", gws, lowBound=0, cat="Integer")
    ft = pulp.LpVariable.dicts("ft", list(gws) + ["end"], lowBound=0, upBound=max_ft, cat="Integer")
    fh = pulp.LpVariable.dicts("fh", P, cat=B) if chip == "freehit" else None   # one-week Free Hit squad
    m += ft[gws[0]] == free_transfers

    def squad_rules(own, label):
        m.addConstraint(pulp.lpSum(own[p] for p in P) == 15, f"size_{label}")
        for pos, n in SQUAD.items():
            m.addConstraint(pulp.lpSum(own[p] for p in P if players[p]["position"] == pos) == n, f"{pos}_{label}")
        for team in teams:
            m.addConstraint(pulp.lpSum(own[p] for p in P if players[p]["team"] == team) <= MAX_PER_TEAM,
                            f"team{team}_{label}")

    objective = []
    for i, t in enumerate(gws):
        prev = {p: (1 if p in owned else 0) for p in P} if i == 0 else {p: x[p][gws[i - 1]] for p in P}
        prev_bank = bank if i == 0 else money[gws[i - 1]]
        nxt = gws[i + 1] if i + 1 < len(gws) else "end"
        own = {p: x[p][t] for p in P}
        for p in P:
            m += own[p] == prev[p] + buy[p][t] - sold[p][t]
            m += buy[p][t] + sold[p][t] <= 1
        m += money[t] == prev_bank + pulp.lpSum(sell[p] * sold[p][t] - players[p]["price"] * buy[p][t] for p in P)
        squad_rules(own, t)

        n_buys = pulp.lpSum(buy[p][t] for p in P)
        if chip in ("wildcard", "freehit") and t == chip_gw:
            m += hits[t] == 0                        # Wildcard: unlimited free transfers this week
            m += ft[nxt] <= ft[t]                    # banked free transfers are kept
            if chip == "freehit":                    # real squad unchanged; play a one-week squad instead
                m += n_buys == 0
                squad_rules(fh, "fh")
                cost = {p: sell[p] if p in owned else players[p]["price"] for p in P}
                m += pulp.lpSum(cost[p] * fh[p] for p in P) <= prev_bank + pulp.lpSum(sell[p] * prev[p] for p in P)
                own = fh
        else:
            # free[t] = free transfers used; every other transfer is a hit (-4). Unused ones roll over (+1 a week).
            m += free[t] <= ft[t]
            m += free[t] <= n_buys
            m += hits[t] == n_buys - free[t]
            m += ft[nxt] <= ft[t] - free[t] + 1

        m += pulp.lpSum(st[p][t] for p in P) == 11
        m += pulp.lpSum(cap[p][t] for p in P) == 1
        for pos in SQUAD:
            n_start = pulp.lpSum(st[p][t] for p in P if players[p]["position"] == pos)
            m += n_start >= MIN_START[pos]
            m += n_start <= MAX_START[pos]
        for p in P:
            m += st[p][t] <= own[p]
            m += cap[p][t] <= st[p][t]

        cap_extra = 2 if (chip == "3xc" and t == chip_gw) else 1
        bench_w = 1.0 if (chip == "bboost" and t == chip_gw) else bench_weight
        w = decay ** i
        objective.append(w * (pulp.lpSum(xp[p, t] * (st[p][t] + cap_extra * cap[p][t]) for p in P)
                              + bench_w * pulp.lpSum(xp[p, t] * (own[p] - st[p][t]) for p in P)
                              - hit_cost * hits[t]))
    m += pulp.lpSum(objective) + ft_value * ft["end"]

    status = m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"solver status {pulp.LpStatus[status]} (chip={chip}, gw={chip_gw})")

    on = lambda v: v.value() is not None and v.value() > 0.5   # noqa: E731
    plan = []
    for i, t in enumerate(gws):
        lineup_squad = [p for p in P if on(fh[p])] if (chip == "freehit" and t == chip_gw) else \
                       [p for p in P if on(x[p][t])]
        starters = [p for p in lineup_squad if on(st[p][t])]
        captain = next(p for p in starters if on(cap[p][t]))
        vice = max((p for p in starters if p != captain), key=lambda p: xp[p, t])
        bench = [p for p in lineup_squad if p not in starters]
        bench_gk = [p for p in bench if players[p]["position"] == "GKP"]
        bench_out = sorted((p for p in bench if p not in bench_gk), key=lambda p: -xp[p, t])
        cap_mult = 3 if (chip == "3xc" and t == chip_gw) else 2
        points = sum(xp[p, t] for p in starters) + (cap_mult - 1) * xp[captain, t]
        if chip == "bboost" and t == chip_gw:
            points += sum(xp[p, t] for p in bench)
        plan.append({
            "gameweek": t,
            "chip": chip if t == chip_gw else None,
            "transfers_in": [p for p in P if on(buy[p][t])],
            "transfers_out": [p for p in P if on(sold[p][t])],
            "free_transfers": round(ft[t].value()),
            "hits": round(hits[t].value()),
            "bank": round(money[t].value()),
            "squad": lineup_squad,
            "starters": sorted(starters, key=lambda p: (list(SQUAD).index(players[p]["position"]), -xp[p, t])),
            "captain": captain,
            "vice_captain": vice,
            "bench": bench_gk + bench_out,          # FPL bench order: substitute GK first, then outfield by xPts
            "xpts": points,
            "xpts_after_hits": points - hit_cost * round(hits[t].value()),
        })
    return {"objective": pulp.value(m.objective), "chip": chip, "chip_gw": chip_gw, "gameweeks": plan}


def plan_with_chips(players, squad, selling_price, bank, free_transfers, gws, chips_available, last_chip_gw,
                    keep_value=None, **kw):
    """Solve without chips, then with each available chip in each horizon week it could be played.

    A chip is recommended for gws[0] only if its gain beats its keep-value (scaled down as the chips' expiry nears,
    0 in the last week it can be used) and no later week in the horizon is better for it. At most one chip (the biggest gain) is recommended.
    Returns (recommended plan, chip evaluations, no-chip plan); each evaluation says if that chip "qualifies" and
    which one is "chosen".
    """
    keep_value = keep_value or KEEP_VALUE
    args = (players, squad, selling_price, bank, free_transfers, gws)
    base = solve(*args, **kw)
    # Chips are use-it-or-lose-it: with 4 chips and 8 weeks left, waiting for "the perfect week" is a luxury.
    # Scale the bar down as chips left approach weeks left (0 when every remaining week needs a chip).
    weeks_left = last_chip_gw - gws[0] + 1
    scarcity = max(0.0, 1 - len(chips_available) / weeks_left)
    solved, evals = {}, []
    for chip in chips_available:
        for t in gws:
            if t <= last_chip_gw:
                solved[chip, t] = solve(*args, chip=chip, chip_gw=t, **kw)
        gains = {t: r["objective"] - base["objective"] for (c, t), r in solved.items() if c == chip}
        if not gains:
            continue
        now, later = gains.get(gws[0]), max((g for t, g in gains.items() if t != gws[0]), default=None)
        threshold = keep_value[chip] * scarcity
        evals.append({
            "chip": chip, "gain_now": now, "threshold": threshold,
            "best_later_gw": max((t for t in gains if t != gws[0]), key=gains.get, default=None),
            "best_later_gain": later,
            "qualifies": now is not None and now >= threshold and (later is None or now >= later),
        })
    picks = [e for e in evals if e["qualifies"]]
    best = max(picks, key=lambda e: e["gain_now"]) if picks else None   # one chip per gameweek
    for e in evals:
        e["chosen"] = e is best
    return (solved[best["chip"], gws[0]] if best else base), evals, base
