"""Gold layer: model-ready features for the xPts model.

gold.player_gw_features: one row per (season, player_id, gameweek) the player's team plays in.
  * Training rows (is_future = false): every completed gameweek, 2020-21 → now, with targets
    target_points / target_minutes = what actually happened (summed over both games in a double GW).
  * Prediction rows (is_future = true): the next HORIZON unfinished gameweeks of the current season,
    targets NULL. The xPts model scores these.

NO LOOK-AHEAD (the most important rule in this file):
  every feature for gameweek g comes from a "state" snapshot taken after the player's last
  *completed* gameweek strictly before g (column feature_as_of_gw). Nothing from GW g or later is used,
  and an expectation fails the pipeline if that is ever violated. Price, ownership and transfers are
  also taken from that earlier state, not from GW g itself.
"""
import dlt

CATALOG = spark.conf.get("fpl.catalog", "fpl")
SILVER = f"{CATALOG}.silver"
GOLD = f"{CATALOG}.gold"
HORIZON = 5  # gameweeks ahead to build prediction rows for


@dlt.table(
    name=f"{GOLD}.player_gw_features",
    comment=(
        "xPts model features: one row per player per gameweek (training rows with targets + next-5-GW prediction rows). "
        "All features are as of the player's last completed GW before the target GW (feature_as_of_gw): no look-ahead."
    ),
    table_properties={"quality": "gold"},
    cluster_by=["season", "player_id"],
)
@dlt.expect_or_fail("no look-ahead", "feature_as_of_gw IS NULL OR feature_as_of_gw < gameweek")
@dlt.expect_or_fail("unique key", "dup_count = 1")
@dlt.expect("known position", "position IN ('GKP','DEF','MID','FWD')")
@dlt.expect("training rows have targets", "is_future OR target_points IS NOT NULL")
def player_gw_features():
    return spark.sql(f"""
    WITH
    cur AS (SELECT max(season) AS season FROM {SILVER}.gameweeks),

    -- ---------------------------------------------------------------- completed fixtures only
    -- current season: only finished gameweeks (an in-progress GW has partial data)
    done_fix AS (
      SELECT h.*
      FROM {SILVER}.player_gw_history h
      LEFT JOIN {SILVER}.gameweeks g ON g.season = h.season AND g.gameweek = h.gameweek
      WHERE h.season <> (SELECT season FROM cur) OR g.finished
    ),

    -- ---------------------------------------------------------------- player x gameweek (doubles summed)
    pgw AS (
      SELECT season, player_id, gameweek,
             max_by(player_code, kickoff_time) AS player_code,
             max_by(position, kickoff_time)    AS position,
             max_by(team_id, kickoff_time)     AS team_id,
             count(*)                          AS n_fix,
             sum(minutes)                      AS minutes,
             sum(total_points)                 AS points,
             sum(CASE WHEN minutes >= 60 THEN 1 ELSE 0 END) AS played60,
             sum(CASE WHEN minutes > 0 THEN 1 ELSE 0 END)   AS appeared,
             sum(goals_scored) AS goals, sum(assists) AS assists, sum(clean_sheets) AS cs,
             sum(bonus) AS bonus, sum(bps) AS bps, sum(saves) AS saves,
             sum(defensive_contribution) AS dc,
             sum(expected_goals) AS xg, sum(expected_assists) AS xa,
             sum(expected_goal_involvements) AS xgi, sum(expected_goals_conceded) AS xgc,
             max_by(price, kickoff_time)    AS price,
             max_by(selected, kickoff_time) AS selected,
             sum(transfers_in) - sum(transfers_out) AS net_transfers
      FROM done_fix
      GROUP BY season, player_id, gameweek
    ),

    -- ---------------------------------------------------------------- player state AFTER each completed GW
    pstate AS (
      SELECT season, player_id, gameweek AS state_gw,
             minutes AS last_minutes, points AS last_points,
             avg(minutes) OVER w3 AS minutes_avg3,  avg(minutes) OVER w5 AS minutes_avg5,
             avg(points)  OVER w3 AS points_avg3,   avg(points)  OVER w5 AS points_avg5,
             sum(points)  OVER wa AS points_season, sum(minutes) OVER wa AS minutes_season,
             count(*)     OVER wa AS gws_season,
             sum(played60) OVER w5 / sum(n_fix) OVER w5 AS played60_rate5,
             sum(appeared) OVER w5 / sum(n_fix) OVER w5 AS appeared_rate5,
             sum(points) OVER w5 * 90 / nullif(sum(minutes) OVER w5, 0) AS points_per90_5,
             sum(xg)  OVER w5 * 90 / nullif(sum(minutes) OVER w5, 0) AS xg_per90_5,
             sum(xa)  OVER w5 * 90 / nullif(sum(minutes) OVER w5, 0) AS xa_per90_5,
             sum(xgi) OVER w5 * 90 / nullif(sum(minutes) OVER w5, 0) AS xgi_per90_5,
             sum(xgc) OVER w5 * 90 / nullif(sum(minutes) OVER w5, 0) AS xgc_per90_5,
             sum(goals + assists) OVER w5 * 90 / nullif(sum(minutes) OVER w5, 0) AS ga_per90_5,
             avg(bonus) OVER w5 AS bonus_avg5, avg(bps) OVER w5 AS bps_avg5,
             avg(saves) OVER w5 AS saves_avg5, avg(dc) OVER w5 AS dc_avg5,
             sum(cs) OVER w5 / sum(n_fix) OVER w5 AS cs_rate5,
             price AS price, selected AS selected, net_transfers AS last_net_transfers
      FROM pgw
      WINDOW w3 AS (PARTITION BY season, player_id ORDER BY gameweek ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
             w5 AS (PARTITION BY season, player_id ORDER BY gameweek ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
             wa AS (PARTITION BY season, player_id ORDER BY gameweek ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    ),

    -- ---------------------------------------------------------------- team form AFTER each completed GW
    team_gw AS (
      SELECT season, team_id, gameweek, sum(team_score) AS gf, sum(opponent_score) AS ga, count(*) AS nf
      FROM (SELECT DISTINCT season, gameweek, fixture_id, team_id, team_score, opponent_score FROM done_fix
            WHERE team_score IS NOT NULL)
      GROUP BY season, team_id, gameweek
    ),
    tstate AS (
      SELECT season, team_id, gameweek AS state_gw,
             sum(gf) OVER w5 / sum(nf) OVER w5 AS goals_for_avg5,
             sum(ga) OVER w5 / sum(nf) OVER w5 AS goals_against_avg5,
             sum(CASE WHEN ga = 0 THEN 1 ELSE 0 END) OVER w5 / count(*) OVER w5 AS clean_sheet_rate5
      FROM team_gw
      WINDOW w5 AS (PARTITION BY season, team_id ORDER BY gameweek ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
    ),

    -- ---------------------------------------------------------------- target fixtures (past + future)
    past_tf AS (
      SELECT season, player_id, gameweek, fixture_id, team_id, opponent_team_id, was_home, position, player_code,
             false AS is_future
      FROM done_fix
    ),
    next_gws AS (
      SELECT g.season, g.gameweek FROM {SILVER}.gameweeks g
      WHERE g.season = (SELECT season FROM cur) AND NOT g.finished
      ORDER BY g.gameweek LIMIT {HORIZON}
    ),
    future_tf AS (
      SELECT f.season, p.player_id, f.gameweek, f.fixture_id, p.team_id,
             CASE WHEN f.home_team_id = p.team_id THEN f.away_team_id ELSE f.home_team_id END AS opponent_team_id,
             f.home_team_id = p.team_id AS was_home, p.position, p.player_code, true AS is_future
      FROM {SILVER}.fixtures f
      JOIN next_gws n ON n.season = f.season AND n.gameweek = f.gameweek
      JOIN {SILVER}.players p ON p.season = f.season AND p.team_id IN (f.home_team_id, f.away_team_id)
    ),
    tf AS (SELECT * FROM past_tf UNION ALL SELECT * FROM future_tf),

    -- opponent and own-team form as of the last completed GW before the target GW (per fixture)
    tf_opp AS (
      SELECT tf.*, o.state_gw AS opp_state_gw,
             o.goals_for_avg5 AS opp_goals_for_avg5, o.goals_against_avg5 AS opp_goals_against_avg5,
             o.clean_sheet_rate5 AS opp_clean_sheet_rate5
      FROM tf LEFT JOIN tstate o
        ON o.season = tf.season AND o.team_id = tf.opponent_team_id AND o.state_gw < tf.gameweek
      QUALIFY row_number() OVER (PARTITION BY tf.season, tf.player_id, tf.fixture_id ORDER BY o.state_gw DESC NULLS LAST) = 1
    ),
    tf_both AS (
      SELECT x.*, t.goals_for_avg5 AS team_goals_for_avg5, t.goals_against_avg5 AS team_goals_against_avg5,
             t.clean_sheet_rate5 AS team_clean_sheet_rate5
      FROM tf_opp x LEFT JOIN tstate t
        ON t.season = x.season AND t.team_id = x.team_id AND t.state_gw < x.gameweek
      QUALIFY row_number() OVER (PARTITION BY x.season, x.player_id, x.fixture_id ORDER BY t.state_gw DESC NULLS LAST) = 1
    ),

    -- one row per player per target GW (doubles: average the fixture-level context)
    tg AS (
      SELECT season, player_id, gameweek, bool_or(is_future) AS is_future,
             max(team_id) AS team_id, max(position) AS position, max(player_code) AS player_code,
             count(*) AS n_fixtures,
             avg(CASE WHEN was_home THEN 1.0 ELSE 0.0 END) AS home_share,
             avg(opp_goals_for_avg5) AS opp_goals_for_avg5,
             avg(opp_goals_against_avg5) AS opp_goals_against_avg5,
             avg(opp_clean_sheet_rate5) AS opp_clean_sheet_rate5,
             avg(team_goals_for_avg5) AS team_goals_for_avg5,
             avg(team_goals_against_avg5) AS team_goals_against_avg5,
             avg(team_clean_sheet_rate5) AS team_clean_sheet_rate5
      FROM tf_both
      GROUP BY season, player_id, gameweek
    ),

    -- player state as of the last completed GW strictly before the target GW
    tg_state AS (
      SELECT tg.*, s.* EXCEPT (season, player_id, state_gw), s.state_gw AS feature_as_of_gw
      FROM tg LEFT JOIN pstate s
        ON s.season = tg.season AND s.player_id = tg.player_id AND s.state_gw < tg.gameweek
      QUALIFY row_number() OVER (PARTITION BY tg.season, tg.player_id, tg.gameweek ORDER BY s.state_gw DESC NULLS LAST) = 1
    ),

    -- previous season summary (joined on player_code, which is stable across seasons)
    prev AS (
      SELECT season, player_code, sum(points) AS prev_points, sum(minutes) AS prev_minutes,
             sum(points) * 90 / nullif(sum(minutes), 0) AS prev_points_per90,
             max_by(price, gameweek) AS prev_end_price
      FROM pgw GROUP BY season, player_code
    )

    SELECT x.season, x.player_id, x.player_code, x.gameweek, x.is_future, x.position, x.team_id,
           x.feature_as_of_gw,
           x.gameweek - coalesce(x.feature_as_of_gw, 0) AS gws_since_state,
           -- context
           x.n_fixtures, x.home_share,
           x.team_goals_for_avg5, x.team_goals_against_avg5, x.team_clean_sheet_rate5,
           x.opp_goals_for_avg5, x.opp_goals_against_avg5, x.opp_clean_sheet_rate5,
           -- player form (as of feature_as_of_gw)
           x.last_minutes, x.last_points, x.minutes_avg3, x.minutes_avg5, x.points_avg3, x.points_avg5,
           x.points_season, x.minutes_season, x.gws_season,
           x.played60_rate5, x.appeared_rate5, x.points_per90_5,
           x.xg_per90_5, x.xa_per90_5, x.xgi_per90_5, x.xgc_per90_5, x.ga_per90_5,
           x.bonus_avg5, x.bps_avg5, x.saves_avg5, x.dc_avg5, x.cs_rate5,
           x.price, x.selected, x.last_net_transfers,
           -- last season
           p.prev_points, p.prev_minutes, p.prev_points_per90, p.prev_end_price,
           -- targets (NULL for future rows)
           t.points  AS target_points,
           t.minutes AS target_minutes,
           count(*) OVER (PARTITION BY x.season, x.player_id, x.gameweek) AS dup_count
    FROM tg_state x
    LEFT JOIN prev p
      ON p.player_code = x.player_code
     AND p.season = concat(cast(cast(substr(x.season, 1, 4) AS INT) - 1 AS STRING), '-', substr(x.season, 3, 2))
    LEFT JOIN pgw t
      ON t.season = x.season AND t.player_id = x.player_id AND t.gameweek = x.gameweek AND NOT x.is_future
    """)
