"""Unit tests for src/tips/tiplib.py.  Run:  python -m unittest discover -s tests -v"""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "tips"))
import tiplib as t  # noqa: E402

UTC = timezone.utc
INCLUDE = t.compile_patterns([r"\bgw ?\d+\b", r"gameweek ?\d+", "captain", "transfer", "differential", "tips"])
EXCLUDE = t.compile_patterns([r"\breview\b", "team name"])


class TitleFilter(unittest.TestCase):
    def test_tip_titles_match(self):
        self.assertTrue(t.title_matches("FPL Gameweek 12 Transfer Tips: Players to Buy", INCLUDE, EXCLUDE))
        self.assertTrue(t.title_matches("Top FPL GW12 Differential Picks", INCLUDE, EXCLUDE))

    def test_reviews_and_junk_excluded(self):
        self.assertFalse(t.title_matches("FPL GW11 review: what happened", INCLUDE, EXCLUDE))
        self.assertFalse(t.title_matches("Best FPL team names 2026", INCLUDE, EXCLUDE))
        self.assertFalse(t.title_matches("Top FPL goalscorers this season", INCLUDE, EXCLUDE))
        self.assertFalse(t.title_matches("", INCLUDE, EXCLUDE))

    def test_slug_to_title(self):
        self.assertEqual(t.slug_to_title("https://allaboutfpl.com/2026/11/fpl-gw12-captain-picks/"), "fpl gw12 captain picks")
        self.assertEqual(t.slug_to_title("https://x.com/"), "")


class GameweekFromTitle(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(t.gw_from_title("FPL GW12 Scout Picks"), 12)
        self.assertEqual(t.gw_from_title("Gameweek 7 captain"), 7)
        self.assertEqual(t.gw_from_title("gw 3 tips"), 3)

    def test_ambiguous_or_missing(self):
        self.assertIsNone(t.gw_from_title("GW12 and GW13 planner"))
        self.assertIsNone(t.gw_from_title("Best midfielders"))
        self.assertIsNone(t.gw_from_title("GW99"))
        self.assertIsNone(t.gw_from_title(None))


class AssignGameweek(unittest.TestCase):
    DEADLINES = [
        (11, datetime(2026, 11, 21, 11, 0, tzinfo=UTC)),
        (12, datetime(2026, 11, 27, 18, 30, tzinfo=UTC)),
        (13, datetime(2026, 12, 1, 18, 30, tzinfo=UTC)),
    ]

    def test_before_deadline_counts_for_that_gw(self):
        self.assertEqual(t.assign_gameweek(datetime(2026, 11, 26, 9, 0, tzinfo=UTC), self.DEADLINES), 12)

    def test_no_look_ahead_at_or_after_deadline(self):
        # published exactly at the GW12 deadline -> too late for GW12
        self.assertEqual(t.assign_gameweek(datetime(2026, 11, 27, 18, 30, tzinfo=UTC), self.DEADLINES), 13)
        self.assertEqual(t.assign_gameweek(datetime(2026, 11, 27, 18, 31, tzinfo=UTC), self.DEADLINES), 13)

    def test_one_second_before_deadline(self):
        self.assertEqual(t.assign_gameweek(datetime(2026, 11, 27, 18, 29, 59, tzinfo=UTC), self.DEADLINES), 12)

    def test_after_last_deadline_and_unsorted_input(self):
        self.assertIsNone(t.assign_gameweek(datetime(2026, 12, 2, tzinfo=UTC), self.DEADLINES))
        self.assertEqual(t.assign_gameweek(datetime(2026, 11, 1, tzinfo=UTC), list(reversed(self.DEADLINES))), 11)

    def test_naive_datetime_treated_as_utc(self):
        self.assertEqual(t.assign_gameweek(datetime(2026, 11, 26, 9, 0), self.DEADLINES), 12)


class Season(unittest.TestCase):
    def test_season_boundaries(self):
        self.assertEqual(t.season_of(datetime(2026, 11, 26, tzinfo=UTC)), "2026-27")
        self.assertEqual(t.season_of(datetime(2026, 7, 1, tzinfo=UTC)), "2026-27")
        self.assertEqual(t.season_of(datetime(2026, 6, 30, 23, 59, tzinfo=UTC)), "2025-26")
        self.assertEqual(t.season_of(datetime(2099, 8, 1, tzinfo=UTC)), "2099-00")


class Dates(unittest.TestCase):
    def test_rss_and_iso(self):
        self.assertEqual(t.parse_any_datetime("Thu, 26 Nov 2026 11:33:54 +0000"), datetime(2026, 11, 26, 11, 33, 54, tzinfo=UTC))
        self.assertEqual(t.parse_any_datetime("2026-11-20T04:21:34.490Z").date().isoformat(), "2026-11-20")
        self.assertEqual(t.parse_any_datetime("2026-11-26T12:00:00+01:00"), datetime(2026, 11, 26, 11, 0, tzinfo=UTC))
        self.assertIsNone(t.parse_any_datetime("not a date"))
        self.assertIsNone(t.parse_any_datetime(None))


class PublishedFromHtml(unittest.TestCase):
    def test_meta_timestamp_either_attribute_order(self):
        a = '<meta property="article:published_time" content="2026-11-26T20:15:00+00:00" />'
        b = "<meta content='2026-11-26T20:15:00Z' property='article:published_time'>"
        for html in (a, b):
            self.assertEqual(t.published_from_html(html), (datetime(2026, 11, 26, 20, 15, tzinfo=UTC), "timestamp"))

    def test_json_ld(self):
        html = '<script type="application/ld+json">{"@type":"Article","datePublished":"2026-11-25T08:00:00+00:00"}</script>'
        self.assertEqual(t.published_from_html(html)[0], datetime(2026, 11, 25, 8, 0, tzinfo=UTC))

    def test_date_only_is_end_of_day(self):
        dt, precision = t.published_from_html('{"datePublished": "2026-11-27"}')
        self.assertEqual((dt, precision), (datetime(2026, 11, 27, 23, 59, 59, tzinfo=UTC), "date"))

    def test_missing(self):
        self.assertEqual(t.published_from_html("<html></html>"), (None, "none"))


class FeedsAndSitemaps(unittest.TestCase):
    RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>Site</title><link>https://s.com</link>
      <item><title>FPL GW12 Tips</title><link>https://s.com/gw12-tips/</link><pubDate>Thu, 26 Nov 2026 11:33:54 +0000</pubDate></item>
      <item><title>No date</title><link>https://s.com/x/</link></item>
    </channel></rss>"""
    SITEMAP = """<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://s.com/2026/11/gw12-tips/</loc><lastmod>2026-11-26T10:00:00+00:00</lastmod></url>
      <url><loc>https://s.com/about/</loc></url></urlset>"""
    INDEX = """<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://s.com/post-sitemap.xml</loc></sitemap></sitemapindex>"""

    def test_rss_items_only(self):
        items = t.parse_rss(self.RSS, "s")
        self.assertEqual([c.url for c in items], ["https://s.com/gw12-tips/", "https://s.com/x/"])
        self.assertEqual(items[0].title, "FPL GW12 Tips")
        self.assertEqual(items[0].published_at, datetime(2026, 11, 26, 11, 33, 54, tzinfo=UTC))
        self.assertIsNone(items[1].published_at)
        self.assertEqual(items[0].discovery_method, "rss")

    def test_sitemap_and_index(self):
        self.assertEqual(t.parse_sitemap(self.SITEMAP)[0], ("https://s.com/2026/11/gw12-tips/", datetime(2026, 11, 26, 10, tzinfo=UTC)))
        self.assertIsNone(t.parse_sitemap(self.SITEMAP)[1][1])
        self.assertEqual(t.parse_sitemap(self.INDEX), [("https://s.com/post-sitemap.xml", None)])


class Content(unittest.TestCase):
    def test_hash_ignores_whitespace(self):
        self.assertEqual(t.content_hash("Buy  Saka\n now"), t.content_hash("Buy Saka now"))
        self.assertNotEqual(t.content_hash("Buy Saka"), t.content_hash("Sell Saka"))

    def test_paywall_heuristic(self):
        self.assertTrue(t.looks_paywalled("short"))
        self.assertTrue(t.looks_paywalled("x " * 300 + "Subscribe to continue reading"))
        self.assertFalse(t.looks_paywalled("Captain Haaland this week. " * 30))

    def test_landing_file_name(self):
        name = t.landing_file_name("fplwire", "abcdef1234", datetime(2026, 11, 26, 9, 5, 7, tzinfo=UTC))
        self.assertEqual(name, "fplwire_abcdef12_20261126T090507Z.json")


class Politeness(unittest.TestCase):
    def test_robots_rules_and_missing_file(self):
        robots = {"https://a.com/robots.txt": (200, "User-agent: *\nDisallow: /private/\n"),
                  "https://b.com/robots.txt": (404, "")}
        r = t.Robots("fpl-agent-tipster/0.1", lambda url: robots[url])
        self.assertTrue(r.allowed("https://a.com/gw12-tips/"))
        self.assertFalse(r.allowed("https://a.com/private/x"))
        self.assertTrue(r.allowed("https://b.com/anything"))

    def test_robots_forbidden_means_disallow(self):
        r = t.Robots("ua", lambda url: (403, ""))
        self.assertFalse(r.allowed("https://c.com/page"))

    def test_rate_limiter_per_host(self):
        clock = {"now": 100.0}
        slept = []

        def fake_sleep(s):
            slept.append(s)
            clock["now"] += s

        rl = t.HostRateLimiter(5, clock=lambda: clock["now"], sleep=fake_sleep)
        self.assertEqual(rl.wait("https://a.com/1"), 0.0)      # first request: no wait
        clock["now"] += 2
        self.assertAlmostEqual(rl.wait("https://a.com/2"), 3.0)  # 2 s elapsed -> wait 3 more
        self.assertEqual(rl.wait("https://b.com/1"), 0.0)      # different host: no wait
        clock["now"] += 10
        self.assertEqual(rl.wait("https://a.com/3"), 0.0)      # plenty of time passed
        self.assertEqual(slept, [3.0])


if __name__ == "__main__":
    unittest.main()
