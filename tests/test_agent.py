import json
import sys
import unittest
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT / "agent"))


def fake_tables():
    players = pd.DataFrame([
        # player_id, web_name, first, second, team_id, team, position, price, status, news, cop
        (1, "Haaland", "Erling", "Haaland", 13, "MCI", "Man City", "FWD", 15.8, "a", "", None),
        (2, "Ødegaard", "Martin", "Ødegaard", 1, "ARS", "Arsenal", "MID", 6.7, "d", "Knee - 50% chance", 50),
        (3, "Gabriel", "Gabriel", "dos Santos Magalhães", 1, "ARS", "Arsenal", "DEF", 8.1, "a", "", None),
        (4, "Gabriel Jesus", "Gabriel", "Fernando de Jesus", 1, "ARS", "Arsenal", "FWD", 6.5, "a", "", None),
    ], columns=["player_id", "web_name", "first_name", "second_name", "team_id", "team_short_name", "team_name",
                "position", "price", "status", "news", "chance_of_playing_next_round"])
    for c, v in {"season": "2026-27", "form": 5.0, "total_points": 50, "minutes": 900, "starts": 10,
                 "goals_scored": 3, "assists": 2, "clean_sheets": 1, "selected_by_percent": 20.0, "ep_next": 4.0}.items():
        players[c] = v
    xpts = pd.DataFrame([(p, gw, 1, x * (1 + 0.01 * gw)) for p, x in [(1, 5.3), (2, 4.0), (3, 5.0), (4, 2.0)]
                         for gw in (12, 13)], columns=["player_id", "gameweek", "n_fixtures", "xpts"])
    xpts["scored_at"] = "2026-11-26T22:00:00"
    xpts["season"], xpts["model_version"] = "2026-27", "2"
    fixtures = pd.DataFrame([(12, "2026-11-28", 1, "ARS", 13, "MCI", 4, 4, False, None, None),
                             (13, "2026-12-02", 13, "MCI", 1, "ARS", 4, 5, False, None, None),
                             (11, "2026-11-21", 1, "ARS", 13, "MCI", 4, 4, True, 1, 1)],
                            columns=["gameweek", "kickoff_time", "home_team_id", "home", "away_team_id", "away",
                                     "home_difficulty", "away_difficulty", "finished", "home_score", "away_score"])
    gameweeks = pd.DataFrame([(12, "2026-11-27T18:30:00", False, True, False)],
                             columns=["gameweek", "deadline_time", "is_current", "is_next", "finished"])
    news = pd.DataFrame([(2, "Ødegaard", pd.Timestamp("2026-11-26T10:00Z"), "d", "Knee - 50% chance", 50, "a", "",
                          False)], columns=["player_id", "web_name", "observed_at", "status", "news",
                                            "chance_of_playing_next_round", "prev_status", "prev_news",
                                            "is_first_observation"])
    return {"players": players, "xpts": xpts, "fixtures": fixtures, "gameweeks": gameweeks, "news": news}


@unittest.skipIf(pd is None, "pandas not installed")
class ToolsTest(unittest.TestCase):
    def setUp(self):
        import fpl_data
        import tools
        self.data = fpl_data.FplData(fake_tables())
        self.tools = tools.FplTools(self.data)

    def test_name_matching(self):
        self.assertEqual(self.data.find_player("odegaard")["player_id"].tolist(), [2])
        self.assertEqual(self.data.find_player("Gabriel")["player_id"].tolist(), [3])      # exact web name wins
        self.assertEqual(self.data.find_player("jesus")["player_id"].tolist(), [4])
        self.assertEqual(self.data.find_player("Erling Haaland")["player_id"].tolist(), [1])
        self.assertEqual(self.data.find_player("E Haaland")["player_id"].tolist(), [1])

    def test_get_player(self):
        r = self.tools.call("get_player", '{"name": "Odegaard"}')
        self.assertEqual(r["status"], "d")
        self.assertEqual(r["chance_of_playing_next_gw"], 50)
        self.assertEqual(list(r["xpts"]), ["GW12", "GW13"])
        self.assertEqual(r["next_fixtures"][0], {"gw": 12, "opponent": "Man City", "venue": "H", "difficulty": 4})

    def test_top_players_filters(self):
        r = self.tools.call("top_players", {"position": "FWD", "max_price": 10})
        self.assertEqual([p["name"] for p in r["players"]], ["Gabriel Jesus"])
        r = self.tools.call("top_players", {"team": "ars", "span": "next_gw"})
        self.assertEqual([p["name"] for p in r["players"]], ["Gabriel", "Ødegaard", "Gabriel Jesus"])
        self.assertIn("news", r["players"][1])          # non-available players carry their news

    def test_errors_are_returned(self):
        self.assertIn("error", self.tools.call("get_player", {"name": "Nobody"}))
        self.assertIn("error", self.tools.call("get_my_squad", {}))            # no team id
        self.assertIn("error", self.tools.call("no_such_tool", {}))
        self.assertIn("error", self.tools.call("get_fixtures", {"team": "XXX"}))

    def test_news(self):
        r = self.tools.call("get_news", {"player": "Ødegaard"})
        self.assertEqual(r["changes"][0]["status"], "d")


@unittest.skipIf(pd is None, "pandas not installed")
class AgentLoopTest(unittest.TestCase):
    def setUp(self):
        import agent
        import fpl_data
        import tools
        self.agent_mod = agent
        self.tools = tools.FplTools(fpl_data.FplData(fake_tables()))

    def scripted(self, replies):
        seen = []

        def llm(messages, tools):
            seen.append([dict(m) for m in messages])
            return replies.pop(0)
        return llm, seen

    def test_tool_then_answer(self):
        llm, seen = self.scripted([
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_player", "arguments": '{"name": "Haaland"}'}}]},
            {"role": "assistant", "content": [{"type": "reasoning", "summary": []},
                                              {"type": "text", "text": "Haaland: 5.4 xPts."}]},
        ])
        out = self.agent_mod.FplAgent(llm, self.tools).run([{"role": "user", "content": "Haaland?"}])
        self.assertEqual(out["answer"], "Haaland: 5.4 xPts.")
        self.assertEqual([s["tool"] for s in out["steps"]], ["get_player"])
        tool_msg = seen[1][-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(json.loads(tool_msg["content"])["name"], "Haaland")
        self.assertIn("GW12", seen[0][0]["content"])                # system prompt knows the next GW

    def test_step_limit(self):
        call = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "get_news", "arguments": "{}"}}]}
        llm, _ = self.scripted([call] * 3)
        out = self.agent_mod.FplAgent(llm, self.tools, max_steps=3).run([{"role": "user", "content": "?"}])
        self.assertIn("tool-call limit", out["answer"])
        self.assertEqual(len(out["steps"]), 3)


if __name__ == "__main__":
    unittest.main()
