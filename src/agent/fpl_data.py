"""The data the agent's tools read: small tables loaded once into pandas.

Loading everything once (a handful of queries, < 5k rows each) and answering tool calls from memory keeps each
question fast and keeps the SQL warehouse (the biggest compute cost so far) idle between sessions.

`run_sql(query) -> pandas.DataFrame` is injected: `sql_via_warehouse` (laptop / Databricks App) or
`sql_via_spark` (notebook / job). Tests pass a fake.
"""

import unicodedata

import pandas as pd

QUERIES = {
    "players": """
        SELECT p.season, p.player_id, p.web_name, p.first_name, p.second_name, p.team_id, p.team_short_name,
               p.team_name, p.position, cast(p.price AS DOUBLE) AS price, p.status, p.news,
               p.chance_of_playing_next_round, p.form, p.total_points, p.minutes, p.starts, p.goals_scored,
               p.assists, p.clean_sheets, p.selected_by_percent, p.ep_next
        FROM {c}.silver.players p
        WHERE p.season = (SELECT max(season) FROM {c}.silver.players)""",
    "xpts": """
        SELECT season, player_id, gameweek, n_fixtures, xpts, model_version, scored_at FROM {c}.gold.xpts""",
    "fixtures": """
        SELECT f.gameweek, f.kickoff_time, f.home_team_id, h.team_short_name AS home, f.away_team_id,
               a.team_short_name AS away, f.home_difficulty, f.away_difficulty, f.finished,
               f.home_score, f.away_score
        FROM {c}.silver.fixtures f
        JOIN {c}.silver.teams h ON h.team_id = f.home_team_id AND h.season = f.season
        JOIN {c}.silver.teams a ON a.team_id = f.away_team_id AND a.season = f.season
        WHERE f.season = (SELECT max(season) FROM {c}.silver.fixtures)""",
    "gameweeks": """
        SELECT gameweek, deadline_time, is_current, is_next, finished FROM {c}.silver.gameweeks
        WHERE season = (SELECT max(season) FROM {c}.silver.gameweeks)""",
    "news": """
        SELECT player_id, web_name, observed_at, status, news, chance_of_playing_next_round, prev_status, prev_news,
               is_first_observation
        FROM {c}.silver.news_snapshots
        WHERE season = (SELECT max(season) FROM {c}.silver.news_snapshots)
          AND observed_at >= current_timestamp() - INTERVAL 30 DAYS""",
}


def normalise(name):
    """'Ødegaard' -> 'odegaard', 'Gibbs-White' -> 'gibbs white' (for forgiving name matching)."""
    s = unicodedata.normalize("NFKD", str(name).replace("Ø", "O").replace("ø", "o"))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return " ".join(s.replace("-", " ").replace(".", " ").replace("'", "").split())


class FplData:
    def __init__(self, tables, recommendations=None):
        self.players = tables["players"].copy()
        self.xpts = tables["xpts"]
        self.fixtures = tables["fixtures"]
        self.gameweeks = tables["gameweeks"]
        self.news = tables["news"]
        self.recommendations = recommendations    # callable(team_id) -> latest row as dict, or None
        self.players["_web"] = self.players["web_name"].map(normalise)
        self.players["_full"] = (self.players["first_name"] + " " + self.players["second_name"]).map(normalise)
        nxt = self.gameweeks[self.gameweeks["is_next"]]
        self.next_gw = int(nxt["gameweek"].iloc[0]) if len(nxt) else None
        self.next_deadline = nxt["deadline_time"].iloc[0] if len(nxt) else None

    @classmethod
    def load(cls, run_sql, catalog="fpl"):
        tables = {k: run_sql(q.format(c=catalog)) for k, q in QUERIES.items()}

        def latest_recommendation(team_id):
            df = run_sql(f"SELECT * FROM {catalog}.gold.recommendations WHERE team_id = {int(team_id)} "
                         f"ORDER BY planned_at DESC LIMIT 1")
            return None if df.empty else df.iloc[0].to_dict()
        return cls(tables, latest_recommendation)

    def find_player(self, name):
        """Rows matching a player name: exact web name, then exact full name, then 'contains'."""
        q = normalise(name)
        P = self.players
        words = q.split()
        every_word = lambda col: P[col].map(lambda v: all(w in v.split() for w in words))   # noqa: E731
        initial_form = f"{words[0][0]} {words[-1]}" if len(words) >= 2 else None   # Bruno Fernandes -> b fernandes
        for mask in (P["_web"] == q, P["_full"] == q,
                     every_word("_full") | every_word("_web"),
                     P["_web"] == initial_form,
                     # "E Haaland" / "Bruno Fernandes": first initial + a surname word of the full name
                     P["_full"].map(lambda v: len(words) >= 2 and v[:1] == words[0][:1] and words[-1] in v.split()),
                     P["_full"].str.contains(q, regex=False) | P["_web"].str.contains(q, regex=False)):
            if mask.any():
                return P[mask]
        return P.iloc[0:0]


def sql_via_warehouse(warehouse_id, client=None):
    """run_sql using a Databricks SQL warehouse through the Databricks SDK (works on a laptop or in an App)."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import Disposition, Format, StatementState

    w = client or WorkspaceClient()

    def run_sql(query):
        r = w.statement_execution.execute_statement(statement=query, warehouse_id=warehouse_id, wait_timeout="50s",
                                                    disposition=Disposition.INLINE, format=Format.JSON_ARRAY)
        if r.status.state != StatementState.SUCCEEDED:
            raise RuntimeError(f"SQL failed: {r.status.error}")
        cols = [c.name for c in r.manifest.schema.columns]
        types = {c.name: c.type_name.value for c in r.manifest.schema.columns}
        rows, chunk = list(r.result.data_array or []), r.result
        while chunk.next_chunk_index is not None:
            chunk = w.statement_execution.get_statement_result_chunk_n(r.statement_id, chunk.next_chunk_index)
            rows += chunk.data_array or []
        df = pd.DataFrame(rows, columns=cols)
        for c, t in types.items():         # the API returns strings: restore the types we use
            if t in ("INT", "LONG", "SHORT", "BYTE", "DOUBLE", "FLOAT", "DECIMAL"):
                df[c] = pd.to_numeric(df[c])
            elif t == "BOOLEAN":
                df[c] = df[c].map({"true": True, "false": False})
            elif t == "TIMESTAMP":
                df[c] = pd.to_datetime(df[c], utc=True)
        return df
    return run_sql


def sql_via_spark(spark):
    return lambda query: spark.sql(query).toPandas()
