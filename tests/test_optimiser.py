import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "optimiser"))
import fpl_entry as E  # noqa: E402

try:
    import pulp  # noqa: F401
    import solver as S  # noqa: E402
except ImportError:          # solver tests need PuLP:  pip install pulp==2.9.0
    S = None


def history(events, chips=()):
    return {"current": [{"event": gw, "bank": bank, "event_transfers": n} for gw, bank, n in events],
            "chips": [{"event": gw, "name": name} for gw, name in chips]}


class EntryTest(unittest.TestCase):
    def test_selling_price(self):
        self.assertEqual(E.selling_price(50, 53), 51)      # +0.3 -> keep +0.1 (half, rounded down)
        self.assertEqual(E.selling_price(50, 54), 52)
        self.assertEqual(E.selling_price(50, 47), 47)      # falls are taken in full
        self.assertEqual(E.selling_price(50, 50), 50)

    def test_free_transfers_bank_and_cap(self):
        h = history([(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)])
        self.assertEqual(E.free_transfers(h), 4)           # 1 after GW1, +1 for each of GW2-4 unused
        h = history([(gw, 0, 0) for gw in range(1, 10)])
        self.assertEqual(E.free_transfers(h), 5)           # capped at 5

    def test_free_transfers_with_hits_and_chips(self):
        h = history([(1, 0, 0), (2, 0, 3), (3, 0, 0)])
        self.assertEqual(E.free_transfers(h), 2)           # GW2: 1 FT, 3 made (2 hits) -> 0 left +1, GW3 unused +1
        h = history([(1, 0, 0), (2, 0, 0), (3, 0, 9), (4, 0, 1)], chips=[(3, "wildcard")])
        self.assertEqual(E.free_transfers(h), 2)           # 2 before WC, kept through it, GW4 uses 1, +1

    def test_late_start(self):
        h = history([(5, 0, 0), (6, 0, 0)])
        self.assertEqual(E.free_transfers(h), 2)

    def test_free_hit_reverts_squad_and_bank(self):
        h = history([(1, 5, 0), (2, 12, 0), (3, 40, 6)], chips=[(3, "freehit")])
        self.assertEqual(E.squad_event(h), 2)
        self.assertEqual(E.bank(h), 12)

    def test_purchase_prices_ignore_free_hit(self):
        h = history([(1, 0, 0), (2, 0, 1), (3, 0, 1)], chips=[(3, "freehit")])
        transfers = [{"element_in": 7, "element_in_cost": 60, "event": 2, "time": "a"},
                     {"element_in": 7, "element_in_cost": 65, "event": 3, "time": "b"}]
        self.assertEqual(E.purchase_prices([7, 8], transfers, h, {7: 55, 8: 45}), {7: 60, 8: 45})

    def test_chips_available_by_half(self):
        h = history([(1, 0, 0)], chips=[(3, "wildcard"), (21, "3xc")])
        self.assertEqual(E.chips_available(h, 12), ["freehit", "bboost", "3xc"])
        self.assertEqual(E.chips_available(h, 25), ["wildcard", "freehit", "bboost"])
        self.assertEqual(E.last_gw_for_chips(12), 19)


def pool(n_gws=3):
    """A tiny league: 10 teams; per position, cheap 'fodder' (40-45) and good players (60-120)."""
    players, pid = {}, 0
    for pos, n in [("GKP", 6), ("DEF", 14), ("MID", 14), ("FWD", 9)]:
        for k in range(n):
            pid += 1
            good = k % 2 == 1
            players[pid] = {"name": f"{pos}{k}", "position": pos, "team": pid % 10,
                            "price": 45 + 5 * k if good else 40 + k % 3,
                            "xpts": {gw: (2.0 + 0.4 * k if good else 0.5) for gw in range(1, n_gws + 1)}}
    return players


def cheapest_squad(players):
    squad, teams = [], {}
    for pos, n in S.SQUAD.items():
        group = sorted((p for p, d in players.items() if d["position"] == pos), key=lambda p: players[p]["price"])
        pick = []
        for p in group:
            if len(pick) < n and teams.get(players[p]["team"], 0) < 3:
                pick.append(p)
                teams[players[p]["team"]] = teams.get(players[p]["team"], 0) + 1
        squad += pick
    return squad


@unittest.skipIf(S is None, "PuLP not installed")
class SolverTest(unittest.TestCase):
    def setUp(self):
        self.players = pool()
        self.squad = cheapest_squad(self.players)
        self.sell = {p: self.players[p]["price"] for p in self.squad}

    def check_valid(self, gw):
        P = self.players
        self.assertEqual(len(gw["squad"]), 15)
        self.assertEqual(len(gw["starters"]), 11)
        self.assertEqual(len(gw["bench"]), 4)
        self.assertEqual(P[gw["bench"][0]]["position"], "GKP")
        self.assertIn(gw["captain"], gw["starters"])
        self.assertNotEqual(gw["captain"], gw["vice_captain"])
        for pos, n in S.SQUAD.items():
            self.assertEqual(sum(P[p]["position"] == pos for p in gw["squad"]), n)
        self.assertGreaterEqual(sum(P[p]["position"] == "DEF" for p in gw["starters"]), 3)
        self.assertGreaterEqual(gw["bank"], 0)

    def test_no_money_no_free_transfers_keeps_squad(self):
        r = S.solve(self.players, self.squad, self.sell, bank=0, free_transfers=1, gws=[1, 2, 3])
        for gw in r["gameweeks"]:
            self.check_valid(gw)
        self.assertEqual(r["gameweeks"][0]["hits"], 0)

    def test_uses_free_transfer_and_budget(self):
        r = S.solve(self.players, self.squad, self.sell, bank=100, free_transfers=1, gws=[1, 2, 3])
        g1 = r["gameweeks"][0]
        self.check_valid(g1)
        self.assertGreaterEqual(len(g1["transfers_in"]), 1)
        spent = sum(self.players[p]["price"] for p in g1["transfers_in"]) - \
            sum(self.sell[p] for p in g1["transfers_out"])
        self.assertEqual(g1["bank"], 100 - spent)
        self.assertEqual(g1["hits"], max(0, len(g1["transfers_in"]) - 1))
        best = max(g1["starters"], key=lambda p: self.players[p]["xpts"][1])
        self.assertEqual(self.players[g1["captain"]]["xpts"][1], self.players[best]["xpts"][1])

    def test_hit_only_when_worth_it(self):
        flat = {p: dict(d, xpts={1: 1.0, 2: 1.0, 3: 1.0}) for p, d in self.players.items()}
        r = S.solve(flat, self.squad, self.sell, bank=500, free_transfers=1, gws=[1, 2, 3])
        self.assertEqual(sum(g["hits"] for g in r["gameweeks"]), 0)   # equal players: never pay -4

    def test_wildcard_has_no_hits_and_free_hit_reverts(self):
        wc = S.solve(self.players, self.squad, self.sell, 100, 1, [1, 2, 3], chip="wildcard", chip_gw=1)
        self.assertEqual(wc["gameweeks"][0]["hits"], 0)
        self.assertGreater(len(wc["gameweeks"][0]["transfers_in"]), 3)
        fh = S.solve(self.players, self.squad, self.sell, 0, 1, [1, 2, 3], chip="freehit", chip_gw=1)
        self.check_valid(fh["gameweeks"][0])
        self.assertEqual(fh["gameweeks"][0]["transfers_in"], [])          # real squad untouched in the FH week
        self.assertNotEqual(set(fh["gameweeks"][0]["squad"]), set(self.squad))

    def test_triple_captain_and_bench_boost_gain(self):
        base = S.solve(self.players, self.squad, self.sell, 0, 1, [1])
        tc = S.solve(self.players, self.squad, self.sell, 0, 1, [1], chip="3xc", chip_gw=1)
        cap = base["gameweeks"][0]["captain"]
        self.assertAlmostEqual(tc["objective"] - base["objective"], self.players[cap]["xpts"][1], places=4)
        bb = S.solve(self.players, self.squad, self.sell, 0, 1, [1], chip="bboost", chip_gw=1)
        bench = sum(self.players[p]["xpts"][1] for p in base["gameweeks"][0]["bench"])
        self.assertAlmostEqual(bb["objective"] - base["objective"], 0.9 * bench, places=4)

    def test_plan_with_chips_thresholds(self):
        _, evals, _ = S.plan_with_chips(self.players, self.squad, self.sell, 0, 1, [1, 2], ["3xc"], last_chip_gw=19,
                                     keep_value={"3xc": 1000})
        self.assertFalse(evals[0]["qualifies"])
        plan, evals, base = S.plan_with_chips(self.players, self.squad, self.sell, 0, 1, [19], ["3xc"], last_chip_gw=19)
        self.assertTrue(evals[0]["qualifies"] and evals[0]["chosen"])     # last chance: threshold 0
        self.assertIsNone(base["gameweeks"][0]["chip"])
        self.assertEqual(plan["gameweeks"][0]["chip"], "3xc")


    def test_must_have_and_exclude(self):
        free = S.solve(self.players, self.squad, self.sell, 100, 1, [1, 2], chip="wildcard", chip_gw=1)
        star = max(free["gameweeks"][0]["squad"], key=lambda p: self.players[p]["xpts"][1])
        r = S.solve(self.players, self.squad, self.sell, 100, 1, [1, 2], chip="wildcard", chip_gw=1, exclude=[star])
        self.assertNotIn(star, r["gameweeks"][0]["squad"] + r["gameweeks"][1]["squad"])
        keep = self.squad[0]
        r = S.solve(self.players, self.squad, self.sell, 100, 1, [1, 2], chip="wildcard", chip_gw=1, must_have=[keep])
        self.assertTrue(all(keep in g["squad"] for g in r["gameweeks"]))
        with self.assertRaises(ValueError):
            S.solve(self.players, self.squad, self.sell, 100, 1, [1], must_have=[99999])


if __name__ == "__main__":
    unittest.main()
