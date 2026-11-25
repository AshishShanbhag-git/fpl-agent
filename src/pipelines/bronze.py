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


def read_landing(endpoint: str) -> DataFrame:
    """Incrementally read every JSON file under landing/<endpoint>/ with Auto Loader."""
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("multiLine", "true")                       # each file is one pretty JSON document
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
