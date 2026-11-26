# Databricks notebook source
# MAGIC %md
# MAGIC # Fetch FPL tip articles → private landing Volume
# MAGIC
# MAGIC Reads `config/tip_sources.yml`, discovers new tip articles, fetches each one politely, extracts the main
# MAGIC text with trafilatura and writes one JSON file per article to `/Volumes/<catalog>/raw/tips_landing/`.
# MAGIC
# MAGIC * **mode=incremental** (scheduled): RSS feeds + snapshot pages.
# MAGIC * **mode=backfill** (manual): sitemaps since `since_date`, capped at `max_articles`.
# MAGIC
# MAGIC Rules enforced here: robots.txt checked for every URL, ≥ `min_delay_seconds` between requests to a host,
# MAGIC descriptive User-Agent, paywalled pages skipped, de-duplication by URL and content hash.
# MAGIC **Copyright:** the raw text lives only in this private Volume and the bronze table built from it.

# COMMAND ----------

# MAGIC %pip install trafilatura==2.0.0 pyyaml --quiet

# COMMAND ----------

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import trafilatura
import yaml

dbutils.widgets.text("catalog", "fpl")
dbutils.widgets.text("bundle_root", "")                 # set by the job: ${workspace.file_path}
dbutils.widgets.dropdown("mode", "incremental", ["incremental", "backfill"])
dbutils.widgets.text("since_date", "2026-07-01")        # backfill only
dbutils.widgets.text("max_articles", "250")             # backfill only: safety cap per run

CATALOG = dbutils.widgets.get("catalog")
ROOT = Path(dbutils.widgets.get("bundle_root") or Path.cwd().parents[1])
MODE = dbutils.widgets.get("mode")
SINCE = datetime.fromisoformat(dbutils.widgets.get("since_date")).replace(tzinfo=timezone.utc)
MAX_ARTICLES = int(dbutils.widgets.get("max_articles"))

sys.path.insert(0, str(ROOT / "src" / "tips"))
import tiplib as t  # noqa: E402

CONFIG = yaml.safe_load((ROOT / "config" / "tip_sources.yml").read_text())
UA, MIN_DELAY = CONFIG["user_agent"], float(CONFIG["min_delay_seconds"])
INCLUDE = t.compile_patterns(CONFIG["default_title_include"])
EXCLUDE = t.compile_patterns(CONFIG["default_title_exclude"])
LANDING = Path(f"/Volumes/{CATALOG}/raw/tips_landing")

SOURCES = [s for s in CONFIG["sources"]
           if s.get("active") and s.get("robots_ok") and s.get("terms_ok") and not s.get("paywalled")]
print(f"mode={MODE}; active sources: {[s['name'] for s in SOURCES]}")

# COMMAND ----------

# ---- HTTP with politeness --------------------------------------------------------------------
session = requests.Session()
session.headers.update({"User-Agent": UA})
limiter = t.HostRateLimiter(MIN_DELAY)


def http_get(url: str) -> tuple[int, str]:
    limiter.wait(url)
    resp = session.get(url, timeout=30)
    return resp.status_code, resp.text


robots = t.Robots(UA, http_get)

# ---- What have we already got? ---------------------------------------------------------------
seen_urls: set[str] = set()
try:
    seen_urls = {r.url for r in spark.table(f"{CATALOG}.bronze.tip_articles_raw").select("url").distinct().collect()}
except Exception:
    pass  # table doesn't exist before the first pipeline run
landing_files = list(LANDING.glob("source=*/dt=*/*.json"))
seen_hashes = {p.name.rsplit("_", 2)[-2] for p in landing_files}  # <source>_<hash8>_<ts>.json
# Also read URLs from landing files: they may not be in bronze yet (pipeline runs after this task).
for p in landing_files:
    try:
        seen_urls.add(json.loads(p.read_text(encoding="utf-8"))["url"])
    except Exception:
        pass
print(f"already have {len(seen_urls)} urls, {len(seen_hashes)} files")

# ---- Gameweek deadlines (for gw_guess; the authoritative assignment happens again in silver) -----
gw_rows = spark.table(f"{CATALOG}.silver.gameweeks").select("season", "gameweek", "deadline_time").collect()
deadlines = [(r.gameweek, r.deadline_time.replace(tzinfo=timezone.utc)) for r in gw_rows]
CURRENT_SEASON = gw_rows[0].season if gw_rows else t.season_of(datetime.now(timezone.utc))

# COMMAND ----------

def discover(source: dict) -> list[t.Candidate]:
    name, disc = source["name"], source["discovery"]
    if MODE == "incremental":
        if disc["method"] == "rss":
            status, text = http_get(disc["url"])
            if status != 200:
                raise RuntimeError(f"feed HTTP {status}")
            items = t.parse_rss(text, name)
            return [c for c in items if t.title_matches(c.title or "", INCLUDE, EXCLUDE)]
        if disc["method"] == "snapshot":
            return [t.Candidate(name, u, None, None, "snapshot") for u in disc["urls"]]
        raise ValueError(f"unknown discovery method {disc['method']}")

    # backfill
    bf = source.get("backfill")
    if not bf:
        return []
    pattern = re.compile(bf["url_pattern"]) if bf.get("url_pattern") else None
    out = []
    for sm_url in bf["urls"]:
        status, text = http_get(sm_url)
        if status != 200:
            print(f"  {name}: sitemap {sm_url} HTTP {status}, skipped")
            continue
        for loc, lastmod in t.parse_sitemap(text):
            if pattern and not pattern.search(loc):
                continue
            if lastmod and lastmod < SINCE:          # lastmod >= publish time, so this filter is safe
                continue
            if t.title_matches(t.slug_to_title(loc), INCLUDE, EXCLUDE):
                out.append(t.Candidate(name, loc, None, lastmod, "sitemap"))
    return out


def fetch_article(c: t.Candidate) -> dict | None:
    """Fetch + extract one article. Returns the landing record, or None if skipped."""
    if not robots.allowed(c.url):
        print(f"  robots.txt disallows {c.url}")
        return None
    status, html = http_get(c.url)
    fetched_at = datetime.now(timezone.utc)
    if status != 200:
        print(f"  HTTP {status} {c.url}")
        return None
    text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_precision=True) or ""
    if t.looks_paywalled(text):
        print(f"  paywalled/too short, skipped: {c.url}")
        return None

    meta = trafilatura.extract_metadata(html)
    title = c.title or (meta.title if meta else None) or t.slug_to_title(c.url)
    page_time, precision = t.published_from_html(html)

    # Publish time, most trustworthy first. Snapshot pages are rewritten weekly -> use our fetch time.
    if c.discovery_method == "snapshot":
        published_at, published_src = fetched_at, "fetched_at(snapshot)"
    elif c.discovery_method == "rss" and c.published_at:
        published_at, published_src = c.published_at, "rss_pubDate"
    elif page_time:
        published_at, published_src = page_time, f"html_meta({precision})"
    elif c.published_at:
        published_at, published_src = c.published_at, "sitemap_lastmod"
    else:
        published_at, published_src = fetched_at, "fetched_at(fallback)"

    chash = t.content_hash(text)
    season = t.season_of(published_at)
    # Deadlines are for the current season only: older articles (sitemaps include past seasons) get no GW here.
    gw_guess = t.assign_gameweek(published_at, deadlines) if deadlines and season == CURRENT_SEASON else None
    gw_in_title = t.gw_from_title(title)
    flags = []
    if c.discovery_method == "snapshot" and gw_in_title and gw_guess and gw_in_title != gw_guess:
        flags.append("stale_snapshot")              # page title names an old gameweek: content not refreshed
    if season != CURRENT_SEASON:
        flags.append("previous_season")
    return {
        "source": c.source,
        "url": c.url,
        "title": title,
        "season": season,
        "published_at": published_at.isoformat(),
        "published_at_source": published_src,
        "fetched_at": fetched_at.isoformat(),
        "gw_guess": gw_guess,
        "gw_in_title": gw_in_title,
        "flags": flags,
        "discovery_method": c.discovery_method,
        "extractor": f"trafilatura=={trafilatura.__version__}",
        "text": text,
        "content_hash": chash,
    }


def write(record: dict) -> Path:
    fetched = datetime.fromisoformat(record["fetched_at"])
    out_dir = LANDING / f"source={record['source']}" / f"dt={fetched.strftime('%Y-%m-%d')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / t.landing_file_name(record["source"], record["content_hash"], fetched)
    tmp = out_dir / f"_tmp_{path.name}"
    tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)                                 # atomic: Auto Loader never sees half a file
    return path

# COMMAND ----------

report, errors, fetched_count = [], [], 0
for source in SOURCES:
    name = source["name"]
    try:
        candidates = discover(source)
    except Exception as exc:
        errors.append(f"{name}: discovery failed: {exc!r}")
        continue
    new = [c for c in candidates if c.discovery_method == "snapshot" or c.url not in seen_urls]
    written = dupes = skipped = 0
    for c in new:
        if MODE == "backfill" and fetched_count >= MAX_ARTICLES:
            break
        try:
            rec = fetch_article(c)
            fetched_count += 1
        except Exception as exc:
            errors.append(f"{name}: {c.url}: {exc!r}")
            continue
        if rec is None:
            skipped += 1
        elif rec["content_hash"][:8] in seen_hashes:
            dupes += 1                               # unchanged snapshot page or syndicated copy
        else:
            write(rec)
            seen_hashes.add(rec["content_hash"][:8])
            seen_urls.add(rec["url"])
            written += 1
    report.append(f"{name}: {len(candidates)} candidates, {len(new)} new, {written} written, "
                  f"{dupes} duplicate, {skipped} skipped")
    print(report[-1])

summary = f"mode={MODE} | " + " | ".join(report) + (f" | {len(errors)} errors" if errors else "")
for e in errors[:10]:
    print("ERROR", e)
if SOURCES and len(report) == 0:
    raise RuntimeError(summary)                      # fail only if discovery failed for every source
dbutils.notebook.exit(summary)
