"""Pure-Python helpers for the tip-article fetcher (no Spark, no network) so they can be unit tested.

Used by src/tips/fetch_tips.py. Tests: tests/test_tiplib.py  (python -m unittest discover -s tests)
"""
from __future__ import annotations

import bisect
import hashlib
import re
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse


# ----------------------------------------------------------------------------- data shapes
@dataclass(frozen=True)
class Candidate:
    """An article URL found during discovery, before we fetch it."""
    source: str
    url: str
    title: str | None
    published_at: datetime | None      # from RSS pubDate / sitemap lastmod (UTC); None if unknown
    discovery_method: str              # rss | sitemap | snapshot


# ----------------------------------------------------------------------------- filtering
def compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def title_matches(title: str, include: list[re.Pattern], exclude: list[re.Pattern]) -> bool:
    """True if the title matches at least one include pattern and no exclude pattern."""
    if not title:
        return False
    if any(p.search(title) for p in exclude):
        return False
    return any(p.search(title) for p in include)


def slug_to_title(url: str) -> str:
    """'https://x.com/2026/11/fpl-gw12-captain-picks/' -> 'fpl gw12 captain picks' (title proxy for sitemaps)."""
    path = urlparse(url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1] if path else ""
    return re.sub(r"[-_]+", " ", slug).strip()


def gw_from_title(title: str | None) -> int | None:
    """Gameweek number mentioned in a title ('GW12', 'Gameweek 12', 'gw 12'), if exactly one is mentioned."""
    if not title:
        return None
    found = {int(n) for n in re.findall(r"\b(?:gw|gameweek)\s?(\d{1,2})\b", title, re.IGNORECASE)}
    found = {n for n in found if 1 <= n <= 38}
    return found.pop() if len(found) == 1 else None


# ----------------------------------------------------------------------------- dates & gameweeks
def to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def parse_any_datetime(value: str | None) -> datetime | None:
    """Parse RFC-822 (RSS) or ISO-8601 (sitemaps, HTML meta) date strings into UTC. None if unparseable."""
    if not value:
        return None
    value = value.strip()
    try:
        return to_utc(parsedate_to_datetime(value))
    except (TypeError, ValueError, IndexError):
        pass
    try:
        return to_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def season_of(dt: datetime) -> str:
    """FPL season label for a timestamp: July-June, e.g. 2026-11-26 -> '2026-27', 2026-03-01 -> '2025-26'."""
    dt = to_utc(dt)
    start = dt.year if dt.month >= 7 else dt.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def assign_gameweek(published_at: datetime, deadlines: list[tuple[int, datetime]]) -> int | None:
    """The gameweek a tip is *for*: the first gameweek whose deadline is strictly after publication.

    No look-ahead: an article published at/after GW n's deadline can only count for GW n+1 onwards.
    Returns None if published after the last deadline of the season.
    """
    ordered = sorted(deadlines, key=lambda d: d[1])
    times = [to_utc(d) for _, d in ordered]
    i = bisect.bisect_right(times, to_utc(published_at))
    return ordered[i][0] if i < len(ordered) else None


_META_PUBLISHED = [
    re.compile(r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)', re.I),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']article:published_time["\']', re.I),
    re.compile(r'"datePublished"\s*:\s*"([^"]+)"', re.I),
]


def published_from_html(html: str) -> tuple[datetime | None, str]:
    """Publish time from page metadata (article:published_time or JSON-LD datePublished).

    Returns (datetime_utc, precision) with precision 'timestamp' or 'date'. A date-only value is moved to
    23:59:59 UTC of that day, so an article can never look *earlier* than it really was (no look-ahead).
    """
    for pattern in _META_PUBLISHED:
        m = pattern.search(html or "")
        if not m:
            continue
        raw = m.group(1).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            d = datetime.fromisoformat(raw)
            return d.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc), "date"
        dt = parse_any_datetime(raw)
        if dt:
            return dt, "timestamp"
    return None, "none"


# ----------------------------------------------------------------------------- feeds & sitemaps
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]  # drop XML namespace


def parse_rss(xml_text: str, source: str) -> list[Candidate]:
    """Items from an RSS 2.0 (or Atom) feed."""
    root = ET.fromstring(xml_text)
    out = []
    for item in root.iter():
        if _local(item.tag) not in ("item", "entry"):
            continue
        fields = {_local(c.tag): c for c in item}
        link_el = fields.get("link")
        link = (link_el.text or link_el.get("href") or "").strip() if link_el is not None else ""
        title = (fields["title"].text or "").strip() if "title" in fields else None
        # Note: an XML element with no children is falsy, so don't chain with `or`.
        date_el = next((fields[k] for k in ("pubDate", "published", "updated") if k in fields), None)
        if link:
            out.append(Candidate(source, link, title, parse_any_datetime(date_el.text if date_el is not None else None), "rss"))
    return out


def parse_sitemap(xml_text: str) -> list[tuple[str, datetime | None]]:
    """(loc, lastmod) pairs from a urlset sitemap. For a sitemap *index*, returns the child sitemap URLs."""
    root = ET.fromstring(xml_text)
    out = []
    for node in root:
        if _local(node.tag) not in ("url", "sitemap"):
            continue
        fields = {_local(c.tag): (c.text or "").strip() for c in node}
        if fields.get("loc"):
            out.append((fields["loc"], parse_any_datetime(fields.get("lastmod"))))
    return out


# ----------------------------------------------------------------------------- content
def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def content_hash(text: str) -> str:
    """Stable fingerprint of the article body (whitespace-insensitive) used for de-duplication."""
    return hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()


PAYWALL_MARKERS = ["subscribe to continue", "subscribe to read", "members only", "premium members",
                   "this content is for members", "log in to continue reading", "become a member to"]


def looks_paywalled(text: str, min_chars: int = 400) -> bool:
    """Heuristic: very short extracted text or typical paywall wording -> treat as paywalled and skip."""
    t = normalise_text(text).lower()
    return len(t) < min_chars or any(m in t for m in PAYWALL_MARKERS)


def landing_file_name(source: str, chash: str, fetched_at: datetime) -> str:
    """'<source>_<hash8>_<YYYYMMDDTHHMMSSZ>.json': the bronze pipeline parses the timestamp suffix."""
    return f"{source}_{chash[:8]}_{to_utc(fetched_at).strftime('%Y%m%dT%H%M%SZ')}.json"


# ----------------------------------------------------------------------------- politeness
class Robots:
    """Caches robots.txt per host. A missing robots.txt (404) means everything is allowed."""

    def __init__(self, user_agent: str, fetch_text):
        self.ua = user_agent
        self._fetch = fetch_text           # callable(url) -> (status_code, text)
        self._parsers: dict[str, urllib.robotparser.RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        host = urlparse(url).netloc
        if host not in self._parsers:
            status, text = self._fetch(f"{urlparse(url).scheme}://{host}/robots.txt")
            rp = urllib.robotparser.RobotFileParser()
            if status == 200:
                rp.parse(text.splitlines())
            elif status in (401, 403):
                rp.disallow_all = True     # robots.txt itself is forbidden -> be conservative
            else:
                rp.parse([])               # 404 etc. -> no rules
            self._parsers[host] = rp
        return self._parsers[host].can_fetch(self.ua, url)


class HostRateLimiter:
    """Ensures at least `min_delay` seconds between two requests to the same host."""

    def __init__(self, min_delay: float, clock=time.monotonic, sleep=time.sleep):
        self.min_delay = min_delay
        self._clock, self._sleep = clock, sleep
        self._last: dict[str, float] = {}

    def wait(self, url: str) -> float:
        host = urlparse(url).netloc
        now = self._clock()
        slept = 0.0
        if host in self._last:
            remaining = self.min_delay - (now - self._last[host])
            if remaining > 0:
                self._sleep(remaining)
                slept = remaining
        self._last[host] = self._clock()
        return slept
