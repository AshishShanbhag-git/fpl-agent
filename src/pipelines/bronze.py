"""Bronze layer: load raw FPL JSON snapshots from the landing Volume into Delta tables.

Runs as a Lakeflow Declarative Pipeline. Auto Loader ("cloudFiles") remembers which files it has
already processed, so each run only picks up the *new* snapshots the ingest job just wrote.

Bronze keeps the data as close to the source as possible:
  * every field is kept, as a string (types are fixed in silver), so a format change at FPL
    can never break ingestion. Nested fields stay as raw JSON text, e.g. bootstrap_static.elements
    is the players array; query it with parse_json()/json_array_length() or the `col:field` syntax;
  * new fields FPL adds mid-season are added as new columns automatically;
  * each row records which file it came from and the snapshot time, so we can always trace it back.
"""
import dlt
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

CATALOG = spark.conf.get("fpl.catalog", "fpl")
LANDING = f"/Volumes/{CATALOG}/raw/landing"

# Files are named like bootstrap_static_20261125T174256Z.json -> the part before ".json" is the UTC time.
_TS_PATTERN = r"_(\d{8}T\d{6}Z)\.json$"


def read_landing(endpoint: str, json_lines: bool = False) -> DataFrame:
    """Incrementally read every JSON file under landing/<endpoint>/ with Auto Loader.

    json_lines=False: each file is one JSON document (API responses).
    json_lines=True:  each line of a file is its own record (element_summary, vaastav backfill).
    """
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("multiLine", str(not json_lines).lower())
        .option("cloudFiles.inferColumnTypes", "false")    # keep everything as strings in bronze
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .load(f"{LANDING}/{endpoint}/")
    )


def add_lineage(df: DataFrame) -> DataFrame:
    """Add columns that say where each row came from and when the snapshot was taken."""
    path = F.col("_metadata.file_path")
    return (
        df.withColumn("_source_file", path)
        .withColumn(
            "_snapshot_ts",
            F.to_timestamp(F.regexp_extract(path, _TS_PATTERN, 1), "yyyyMMdd'T'HHmmss'Z'"),
        )
        .withColumn("_ingested_at", F.current_timestamp())
    )


@dlt.table(
    name="bootstrap_static",
    comment="One row per bootstrap-static snapshot: players (elements), teams, gameweeks (events), etc.",
    table_properties={"quality": "bronze"},
)
@dlt.expect_or_fail("snapshot time parsed from file name", "_snapshot_ts IS NOT NULL")
@dlt.expect("payload has players", "elements IS NOT NULL")
def bootstrap_static():
    return add_lineage(read_landing("bootstrap_static"))


@dlt.table(
    name="fixtures",
    comment="One row per fixture per snapshot (the fixtures file is a JSON array of matches).",
    table_properties={"quality": "bronze"},
)
@dlt.expect_or_fail("snapshot time parsed from file name", "_snapshot_ts IS NOT NULL")
@dlt.expect("fixture has an id", "id IS NOT NULL")
def fixtures():
    return add_lineage(read_landing("fixtures"))


@dlt.table(
    name="element_summary",
    comment="One row per player per daily element-summary snapshot: current-season match history (raw JSON text).",
    table_properties={"quality": "bronze"},
)
@dlt.expect_or_fail("snapshot time parsed from file name", "_snapshot_ts IS NOT NULL")
@dlt.expect("row has a player id", "player_id IS NOT NULL")
def element_summary():
    return add_lineage(read_landing("element_summary", json_lines=True))


# --- Historical seasons (vaastav/Fantasy-Premier-League backfill). `season` comes from the folder name. ---

@dlt.table(
    name="vaastav_merged_gw",
    comment="Past seasons: one row per player per fixture, from vaastav merged_gw.csv (all columns as strings).",
    table_properties={"quality": "bronze"},
)
@dlt.expect_or_fail("snapshot time parsed from file name", "_snapshot_ts IS NOT NULL")
@dlt.expect("row has season", "season IS NOT NULL")
def vaastav_merged_gw():
    return add_lineage(read_landing("vaastav_merged_gw", json_lines=True))


@dlt.table(
    name="vaastav_players",
    comment="Past seasons: player id -> code / names / position, from vaastav players_raw.csv.",
    table_properties={"quality": "bronze"},
)
@dlt.expect_or_fail("snapshot time parsed from file name", "_snapshot_ts IS NOT NULL")
def vaastav_players():
    return add_lineage(read_landing("vaastav_players", json_lines=True))


@dlt.table(
    name="vaastav_teams",
    comment="Past seasons: team id -> name, from vaastav teams.csv.",
    table_properties={"quality": "bronze"},
)
@dlt.expect_or_fail("snapshot time parsed from file name", "_snapshot_ts IS NOT NULL")
def vaastav_teams():
    return add_lineage(read_landing("vaastav_teams", json_lines=True))
