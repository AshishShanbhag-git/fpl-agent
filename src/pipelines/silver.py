"""Silver layer: clean, typed tables built from bronze.

Runs in the same Lakeflow Declarative Pipeline as bronze, right after it. Tables are published to
<catalog>.silver.* using fully-qualified names.

  player_snapshots  STREAMING  one row per player per snapshot (append-only full history)
  news_snapshots    MV         one row each time a player's status / news / chance-of-playing changes
  players           MV         current state of every player (latest snapshot), with team + position names
  teams             MV         current teams and strength ratings
  gameweeks         MV         current gameweeks (deadlines, current/next flags)
  fixtures          MV         current fixture list with difficulty ratings and scores

Bronze stores nested data as raw JSON text. Here we parse it with parse_json() into VARIANT and pull out
each field with try_variant_get(), which returns NULL (instead of failing) if FPL renames or drops a field.
"""
import dlt
from pyspark.sql import DataFrame

CATALOG = spark.conf.get("fpl.catalog", "fpl")
BRONZE = f"{CATALOG}.bronze"
SILVER = f"{CATALOG}.silver"

# FPL season label from a timestamp: Jul 2026 - Jun 2027 -> "2026-27".
# Player ids are reused for different players each season, so (season, player_id) is the real key.
SEASON_EXPR = """CASE WHEN month(snapshot_ts) >= 7
    THEN concat(year(snapshot_ts), '-', lpad(cast((year(snapshot_ts) + 1) % 100 AS STRING), 2, '0'))
    ELSE concat(year(snapshot_ts) - 1, '-', lpad(cast(year(snapshot_ts) % 100 AS STRING), 2, '0')) END"""


def fields(var: str, spec: list[tuple[str, str, str]]) -> list[str]:
    """Build `try_variant_get(var, '$.json_field', 'type') AS column` expressions."""
    return [f"try_variant_get({var}, '$.{src}', '{typ}') AS {dst}" for src, dst, typ in spec]


# (json field, output column, type)
PLAYER_FIELDS = [
    ("id", "player_id", "int"),
    ("code", "player_code", "bigint"),          # stable across seasons
    ("web_name", "web_name", "string"),
    ("first_name", "first_name", "string"),
    ("second_name", "second_name", "string"),
    ("team", "team_id", "int"),
    ("element_type", "position_id", "int"),
    ("status", "status", "string"),             # a=available d=doubtful i=injured s=suspended u=unavailable n=not in squad
    ("news", "news", "string"),
    ("news_added", "news_added", "timestamp"),
    ("chance_of_playing_next_round", "chance_of_playing_next_round", "int"),
    ("chance_of_playing_this_round", "chance_of_playing_this_round", "int"),
    ("form", "form", "double"),
    ("points_per_game", "points_per_game", "double"),
    ("total_points", "total_points", "int"),
    ("event_points", "event_points", "int"),
    ("minutes", "minutes", "int"),
    ("starts", "starts", "int"),
    ("goals_scored", "goals_scored", "int"),
    ("assists", "assists", "int"),
    ("clean_sheets", "clean_sheets", "int"),
    ("goals_conceded", "goals_conceded", "int"),
    ("saves", "saves", "int"),
    ("bonus", "bonus", "int"),
    ("bps", "bps", "int"),
    ("defensive_contribution", "defensive_contribution", "int"),
    ("influence", "influence", "double"),
    ("creativity", "creativity", "double"),
    ("threat", "threat", "double"),
    ("ict_index", "ict_index", "double"),
    ("expected_goals", "expected_goals", "double"),
    ("expected_assists", "expected_assists", "double"),
    ("expected_goal_involvements", "expected_goal_involvements", "double"),
    ("expected_goals_conceded", "expected_goals_conceded", "double"),
    ("selected_by_percent", "selected_by_percent", "double"),
    ("transfers_in_event", "transfers_in_event", "int"),
    ("transfers_out_event", "transfers_out_event", "int"),
    ("cost_change_event", "cost_change_event_raw", "int"),
    ("now_cost", "now_cost_raw", "int"),        # in tenths of £m: 105 = £10.5m
    ("ep_next", "ep_next", "double"),           # FPL's own expected points (a baseline to beat)
    ("ep_this", "ep_this", "double"),
]

TEAM_FIELDS = [
    ("id", "team_id", "int"),
    ("code", "team_code", "int"),
    ("name", "team_name", "string"),
    ("short_name", "team_short_name", "string"),
    ("strength", "strength", "int"),
    ("strength_overall_home", "strength_overall_home", "int"),
    ("strength_overall_away", "strength_overall_away", "int"),
    ("strength_attack_home", "strength_attack_home", "int"),
    ("strength_attack_away", "strength_attack_away", "int"),
    ("strength_defence_home", "strength_defence_home", "int"),
    ("strength_defence_away", "strength_defence_away", "int"),
]

GAMEWEEK_FIELDS = [
    ("id", "gameweek", "int"),
    ("name", "name", "string"),
    ("deadline_time", "deadline_time", "timestamp"),
    ("is_previous", "is_previous", "boolean"),
    ("is_current", "is_current", "boolean"),
    ("is_next", "is_next", "boolean"),
    ("finished", "finished", "boolean"),
    ("data_checked", "data_checked", "boolean"),
    ("average_entry_score", "average_entry_score", "int"),
    ("highest_score", "highest_score", "int"),
]


def latest_bootstrap_array(column: str) -> DataFrame:
    """Explode one JSON array (e.g. 'teams') from the most recent bootstrap snapshot."""
    return spark.sql(f"""
        SELECT _snapshot_ts AS snapshot_ts, item
        FROM {BRONZE}.bootstrap_static
        LATERAL VIEW explode(cast(parse_json({column}) AS ARRAY<VARIANT>)) AS item
        WHERE _snapshot_ts = (SELECT max(_snapshot_ts) FROM {BRONZE}.bootstrap_static)
    """)


# --------------------------------------------------------------------------------------------
# Full history: every player in every snapshot (streaming = only new bronze rows are processed)
# --------------------------------------------------------------------------------------------
@dlt.table(
    name=f"{SILVER}.player_snapshots",
    comment="One row per player per FPL snapshot. Full typed history; key = (season, player_id, snapshot_ts).",
    table_properties={"quality": "silver"},
    cluster_by=["season", "player_id"],
)
@dlt.expect_or_drop("valid player id", "player_id IS NOT NULL")
@dlt.expect("price is positive", "price > 0")
def player_snapshots():
    return (
        spark.readStream.table(f"{BRONZE}.bootstrap_static")
        .selectExpr("_snapshot_ts AS snapshot_ts", "explode(cast(parse_json(elements) AS ARRAY<VARIANT>)) AS p")
        .selectExpr("snapshot_ts", *fields("p", PLAYER_FIELDS))
        # season/player_id/price come first: Delta only keeps stats on the first 32 columns,
        # and clustering (fast lookups by season + player) needs stats on those columns.
        .selectExpr(
            f"{SEASON_EXPR} AS season",
            "*",
            "now_cost_raw / 10.0 AS price",
            "cost_change_event_raw / 10.0 AS price_change_gw",
            "CASE position_id WHEN 1 THEN 'GKP' WHEN 2 THEN 'DEF' WHEN 3 THEN 'MID' WHEN 4 THEN 'FWD' END AS position",
        )
        .drop("now_cost_raw", "cost_change_event_raw")
        .selectExpr("season", "player_id", "snapshot_ts", "web_name", "team_id", "position", "price", "* EXCEPT (season, player_id, snapshot_ts, web_name, team_id, position, price)")
    )


# --------------------------------------------------------------------------------------------
# News change log: the input for the LLM news -> structured signal step and its backtest
# --------------------------------------------------------------------------------------------
@dlt.table(
    name=f"{SILVER}.news_snapshots",
    comment=(
        "One row each time a player's availability info changes (status, news text or chance of playing), "
        "plus each player's first observation as a baseline. observed_at = first snapshot showing the new state."
    ),
    table_properties={"quality": "silver"},
)
def news_snapshots():
    return spark.sql(f"""
        WITH keyed AS (
          SELECT *,
                 concat_ws('|', coalesce(status, ''), coalesce(news, ''),
                           coalesce(cast(chance_of_playing_next_round AS STRING), '')) AS state_key
          FROM {SILVER}.player_snapshots
        ),
        with_prev AS (
          SELECT *,
                 lag(state_key) OVER w AS prev_key,
                 lag(status) OVER w AS prev_status,
                 lag(news) OVER w AS prev_news,
                 lag(chance_of_playing_next_round) OVER w AS prev_chance_of_playing_next_round
          FROM keyed
          WINDOW w AS (PARTITION BY season, player_id ORDER BY snapshot_ts)
        )
        SELECT season, player_id, player_code, web_name, team_id, position,
               snapshot_ts AS observed_at,
               status, news, news_added,
               chance_of_playing_next_round, chance_of_playing_this_round,
               prev_status, prev_news, prev_chance_of_playing_next_round,
               prev_key IS NULL AS is_first_observation
        FROM with_prev
        WHERE prev_key IS NULL OR prev_key <> state_key
    """)


# --------------------------------------------------------------------------------------------
# Current-state tables (rebuilt from the latest snapshot each run)
# --------------------------------------------------------------------------------------------
@dlt.table(
    name=f"{SILVER}.teams",
    comment="Current Premier League teams with FPL strength ratings (latest snapshot).",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_fail("20 teams", "team_count = 20")
def teams():
    return (
        latest_bootstrap_array("teams")
        .selectExpr("snapshot_ts", f"{SEASON_EXPR} AS season", *fields("item", TEAM_FIELDS))
        .selectExpr("*", "count(*) OVER () AS team_count")
    )


@dlt.table(
    name=f"{SILVER}.gameweeks",
    comment="Current gameweek calendar: deadlines and previous/current/next flags (latest snapshot).",
    table_properties={"quality": "silver"},
)
def gameweeks():
    return latest_bootstrap_array("events").selectExpr(
        "snapshot_ts", f"{SEASON_EXPR} AS season", *fields("item", GAMEWEEK_FIELDS)
    )


@dlt.table(
    name=f"{SILVER}.players",
    comment="Current state of every player (latest snapshot), with team and position names.",
    table_properties={"quality": "silver"},
)
def players():
    return spark.sql(f"""
        SELECT p.*, t.team_name, t.team_short_name
        FROM {SILVER}.player_snapshots p
        LEFT JOIN {SILVER}.teams t ON p.team_id = t.team_id
        WHERE p.snapshot_ts = (SELECT max(snapshot_ts) FROM {SILVER}.player_snapshots)
    """)


@dlt.table(
    name=f"{SILVER}.fixtures",
    comment="Current fixture list (latest snapshot). gameweek is NULL for postponed/unscheduled matches.",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_fail("valid fixture id", "fixture_id IS NOT NULL")
def fixtures():
    return spark.sql(f"""
        SELECT _snapshot_ts AS snapshot_ts,
               {SEASON_EXPR.replace('snapshot_ts', '_snapshot_ts')} AS season,
               try_cast(id AS INT)                AS fixture_id,
               try_cast(code AS BIGINT)           AS fixture_code,
               try_cast(event AS INT)             AS gameweek,
               try_cast(kickoff_time AS TIMESTAMP) AS kickoff_time,
               try_cast(team_h AS INT)            AS home_team_id,
               try_cast(team_a AS INT)            AS away_team_id,
               try_cast(team_h_difficulty AS INT) AS home_difficulty,
               try_cast(team_a_difficulty AS INT) AS away_difficulty,
               try_cast(team_h_score AS INT)      AS home_score,
               try_cast(team_a_score AS INT)      AS away_score,
               try_cast(started AS BOOLEAN)       AS started,
               try_cast(finished AS BOOLEAN)      AS finished,
               try_cast(minutes AS INT)           AS minutes
        FROM {BRONZE}.fixtures
        WHERE _snapshot_ts = (SELECT max(_snapshot_ts) FROM {BRONZE}.fixtures)
    """)


# --------------------------------------------------------------------------------------------
# Per-gameweek player history: past seasons (vaastav) + current season (FPL element-summary)
# --------------------------------------------------------------------------------------------
# Columns shared by both sources: (name, SQL type). Missing stats in old seasons become NULL.
GW_STAT_COLUMNS = [
    ("minutes", "INT"), ("starts", "INT"), ("total_points", "INT"),
    ("goals_scored", "INT"), ("assists", "INT"), ("clean_sheets", "INT"), ("goals_conceded", "INT"),
    ("own_goals", "INT"), ("penalties_saved", "INT"), ("penalties_missed", "INT"),
    ("yellow_cards", "INT"), ("red_cards", "INT"), ("saves", "INT"), ("bonus", "INT"), ("bps", "INT"),
    ("clearances_blocks_interceptions", "INT"), ("recoveries", "INT"), ("tackles", "INT"),
    ("defensive_contribution", "INT"),
    ("influence", "DOUBLE"), ("creativity", "DOUBLE"), ("threat", "DOUBLE"), ("ict_index", "DOUBLE"),
    ("expected_goals", "DOUBLE"), ("expected_assists", "DOUBLE"),
    ("expected_goal_involvements", "DOUBLE"), ("expected_goals_conceded", "DOUBLE"),
    ("selected", "BIGINT"), ("transfers_in", "BIGINT"), ("transfers_out", "BIGINT"), ("transfers_balance", "BIGINT"),
]
_VARIANT_TYPE = {"INT": "int", "BIGINT": "bigint", "DOUBLE": "double"}


def _current_stats(var: str) -> str:
    return ",\n               ".join(
        f"try_variant_get({var}, '$.{c}', '{_VARIANT_TYPE[t]}') AS {c}" for c, t in GW_STAT_COLUMNS
    )


def _hist_stats(alias: str) -> str:
    return ",\n               ".join(f"try_cast({alias}.{c} AS {t}) AS {c}" for c, t in GW_STAT_COLUMNS)


@dlt.table(
    name=f"{SILVER}.player_gw_history",
    comment=(
        "One row per player per fixture played (or squad-listed) in a gameweek. Double gameweeks have two rows. "
        "Past seasons from vaastav/Fantasy-Premier-League; current season from FPL element-summary. "
        "Key = (season, player_id, fixture_id). price is the price at that gameweek in £m."
    ),
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("has keys", "season IS NOT NULL AND player_id IS NOT NULL AND fixture_id IS NOT NULL")
@dlt.expect("known position", "position IN ('GKP','DEF','MID','FWD')")
def player_gw_history():
    current_season_expr = SEASON_EXPR.replace("snapshot_ts", "es._snapshot_ts")
    return spark.sql(f"""
        WITH
        -- ---------- current season: latest daily element-summary snapshot ----------
        es AS (
          SELECT * FROM {BRONZE}.element_summary
          WHERE _snapshot_ts = (SELECT max(_snapshot_ts) FROM {BRONZE}.element_summary)
        ),
        cur_raw AS (
          SELECT {current_season_expr} AS season, h
          FROM es LATERAL VIEW explode(cast(parse_json(es.history) AS ARRAY<VARIANT>)) AS h
        ),
        cur_players AS (   -- latest known name / position / code for each player this season
          SELECT season, player_id, player_code, web_name, position
          FROM (SELECT *, row_number() OVER (PARTITION BY season, player_id ORDER BY snapshot_ts DESC) AS rn
                FROM {SILVER}.player_snapshots)
          WHERE rn = 1
        ),
        cur AS (
          SELECT r.season,
                 try_variant_get(h, '$.round', 'int')         AS gameweek,
                 try_variant_get(h, '$.element', 'int')       AS player_id,
                 p.player_code, p.web_name, p.position,
                 CASE WHEN try_variant_get(h, '$.was_home', 'boolean') THEN f.home_team_id ELSE f.away_team_id END AS team_id,
                 try_variant_get(h, '$.opponent_team', 'int') AS opponent_team_id,
                 try_variant_get(h, '$.fixture', 'int')       AS fixture_id,
                 try_variant_get(h, '$.kickoff_time', 'timestamp') AS kickoff_time,
                 try_variant_get(h, '$.was_home', 'boolean')  AS was_home,
                 CASE WHEN try_variant_get(h, '$.was_home', 'boolean') THEN try_variant_get(h, '$.team_h_score', 'int')
                      ELSE try_variant_get(h, '$.team_a_score', 'int') END AS team_score,
                 CASE WHEN try_variant_get(h, '$.was_home', 'boolean') THEN try_variant_get(h, '$.team_a_score', 'int')
                      ELSE try_variant_get(h, '$.team_h_score', 'int') END AS opponent_score,
                 try_variant_get(h, '$.value', 'int') / 10.0  AS price,
                 {_current_stats('h')},
                 'fpl_api' AS source
          FROM cur_raw r
          LEFT JOIN cur_players p
            ON p.season = r.season AND p.player_id = try_variant_get(h, '$.element', 'int')
          LEFT JOIN {SILVER}.fixtures f
            ON f.season = r.season AND f.fixture_id = try_variant_get(h, '$.fixture', 'int')
        ),

        -- ---------- past seasons: latest vaastav backfill of each season ----------
        gw AS (
          SELECT * FROM (SELECT *, max(_snapshot_ts) OVER (PARTITION BY season) AS latest FROM {BRONZE}.vaastav_merged_gw)
          WHERE _snapshot_ts = latest
        ),
        vp AS (
          SELECT * FROM (SELECT *, max(_snapshot_ts) OVER (PARTITION BY season) AS latest FROM {BRONZE}.vaastav_players)
          WHERE _snapshot_ts = latest
        ),
        vt AS (
          SELECT * FROM (SELECT *, max(_snapshot_ts) OVER (PARTITION BY season) AS latest FROM {BRONZE}.vaastav_teams)
          WHERE _snapshot_ts = latest
        ),
        hist AS (
          SELECT g.season,
                 try_cast(g.GW AS INT)                   AS gameweek,
                 try_cast(g.element AS INT)              AS player_id,
                 try_cast(vp.code AS BIGINT)             AS player_code,
                 vp.web_name,
                 CASE g.position WHEN 'GK' THEN 'GKP' ELSE g.position END AS position,
                 try_cast(vt.id AS INT)                  AS team_id,
                 try_cast(g.opponent_team AS INT)        AS opponent_team_id,
                 try_cast(g.fixture AS INT)              AS fixture_id,
                 try_cast(g.kickoff_time AS TIMESTAMP)   AS kickoff_time,
                 try_cast(g.was_home AS BOOLEAN)         AS was_home,
                 CASE WHEN try_cast(g.was_home AS BOOLEAN) THEN try_cast(g.team_h_score AS INT)
                      ELSE try_cast(g.team_a_score AS INT) END AS team_score,
                 CASE WHEN try_cast(g.was_home AS BOOLEAN) THEN try_cast(g.team_a_score AS INT)
                      ELSE try_cast(g.team_h_score AS INT) END AS opponent_score,
                 try_cast(g.value AS INT) / 10.0         AS price,
                 {_hist_stats('g')},
                 'vaastav' AS source
          FROM gw g
          LEFT JOIN vp ON vp.season = g.season AND vp.id = g.element
          LEFT JOIN vt ON vt.season = g.season AND vt.name = g.team
          WHERE g.position IS DISTINCT FROM 'AM'                          -- 2024-25 Assistant Manager chip rows
            AND g.season NOT IN (SELECT DISTINCT season FROM cur_raw)     -- current season comes from the API
        )

        -- vaastav's 2025-26 file contains a few exact duplicate rows: keep one per (season, player, fixture)
        SELECT * FROM hist
        QUALIFY row_number() OVER (PARTITION BY season, player_id, fixture_id ORDER BY minutes DESC NULLS LAST) = 1
        UNION ALL
        SELECT * FROM cur
    """)
