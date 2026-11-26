"""Check robots.txt for candidate tip sources before adding them to config/tip_sources.yml.

For each site it reports:
  * whether our User-Agent may fetch the sample article path (generic `*` rules + our own UA),
  * whether the site blocks AI crawlers (GPTBot, ClaudeBot, CCBot, Google-Extended, ...). We treat that as
    the site saying "don't use our content for AI", and since we'd run an LLM over the text, we don't use those sites,
  * any Crawl-delay, and the Sitemap URLs (used to discover articles).

Usage:  python -I scripts/check_robots.py            (prints a Markdown table)
Only fetches robots.txt files; never fetches article pages.
"""
from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser

USER_AGENT = "fpl-agent-tipster/0.1 (+https://github.com/AshishShanbhag-git/fpl-agent; personal non-commercial research)"
AI_BOTS = ["GPTBot", "ClaudeBot", "anthropic-ai", "Claude-Web", "CCBot", "Google-Extended", "PerplexityBot", "Bytespider"]

# (name, base URL, sample article path to test)
CANDIDATES = [
    ("Fantasy Football Scout", "https://www.fantasyfootballscout.co.uk", "/2026/11/24/fpl-gameweek-12-tips/"),
    ("Fantasy Football Hub", "https://www.fantasyfootballhub.co.uk", "/blog/fpl-gameweek-12-tips"),
    ("Premier League (The Scout)", "https://www.premierleague.com", "/news/4321000"),
    ("Fantasy Football Fix", "https://www.fantasyfootballfix.com", "/blog-index/fpl-gameweek-12-tips/"),
    ("Fantasy Football Pundit", "https://www.fantasyfootballpundit.com", "/fpl-gameweek-12-tips/"),
    ("All About FPL", "https://allaboutfpl.com", "/2026/11/fpl-gameweek-12-tips/"),
    ("Fantasy Football Community", "https://fantasyfootballcommunity.com", "/fpl-gameweek-12-tips/"),
    ("Planet FPL", "https://www.planetfpl.com", "/fpl-gameweek-12-tips/"),
    ("FPL Wire", "https://www.fplwire.com", "/fpl-gameweek-12-tips/"),
    ("Sportskeeda (FPL)", "https://www.sportskeeda.com", "/football/fpl-gameweek-12-tips"),
    ("Sky Sports (FPL)", "https://www.skysports.com", "/football/news/11095/13000000/fpl-tips"),
    ("Evening Standard (FPL)", "https://www.standard.co.uk", "/sport/football/fpl-tips-gameweek-12-b1000000.html"),
    ("The Independent (FPL)", "https://www.independent.co.uk", "/sport/football/fpl-tips-gameweek-12-b2000000.html"),
    ("Goal.com (FPL)", "https://www.goal.com", "/en-gb/news/fpl-tips-gameweek-12/blt000"),
]


def fetch_robots(base: str) -> tuple[int | None, str]:
    req = urllib.request.Request(base + "/robots.txt", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:  # DNS, TLS, timeout...
        return None, repr(e)


def check(name: str, base: str, sample: str) -> dict:
    status, text = fetch_robots(base)
    row = {"name": name, "base": base, "http": status}
    if status is None:
        row.update(ok="?", note=f"robots.txt unreachable: {text[:60]}", ai_block="?", delay="", sitemaps=0)
        return row
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(text.splitlines() if status == 200 else [])  # 404 = no restrictions
    sample_url = base + sample
    ok = rp.can_fetch(USER_AGENT, sample_url) and rp.can_fetch("*", sample_url)
    blocked_ai = [b for b in AI_BOTS if not rp.can_fetch(b, base + "/")]
    row.update(
        ok="yes" if ok else "no",
        ai_block=", ".join(blocked_ai) if blocked_ai else "none",
        delay=str(rp.crawl_delay(USER_AGENT) or rp.crawl_delay("*") or ""),
        sitemaps=len(rp.site_maps() or []),
        note="no robots.txt (404)" if status == 404 else ("" if status == 200 else f"HTTP {status}"),
    )
    return row


def main() -> int:
    rows = []
    for name, base, sample in CANDIDATES:
        rows.append(check(name, base, sample))
        time.sleep(1)
    print("| Source | robots allows our UA | Blocks AI crawlers | Crawl-delay | Sitemaps | Note |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['ok']} | {r['ai_block']} | {r['delay']} | {r['sitemaps']} | {r['note']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
