# Databricks notebook source
# MAGIC %md
# MAGIC # Can Databricks reach the tip sources?
# MAGIC Free Edition limits outbound internet to trusted domains. This fetches **only robots.txt and each source's
# MAGIC discovery URL** (feed / sitemap / snapshot page head), 5 s apart, and reports the HTTP result.
# MAGIC No article text is stored.

# COMMAND ----------

import time

import requests

UA = "fpl-agent-tipster/0.1 (+https://github.com/AshishShanbhag-git/fpl-agent; personal non-commercial research)"
CHECKS = [
    ("allaboutfpl", "https://allaboutfpl.com/robots.txt"),
    ("allaboutfpl", "https://allaboutfpl.com/feed/"),
    ("fplwire", "https://www.fplwire.com/robots.txt"),
    ("fplwire", "https://www.fplwire.com/feed/"),
    ("ffpundit", "https://www.fantasyfootballpundit.com/robots.txt"),
    ("ffpundit", "https://www.fantasyfootballpundit.com/sitemap.xml"),
]

results = []
for source, url in CHECKS:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        results.append(f"{source:12} {r.status_code} {len(r.content):>8} bytes  {url}")
    except Exception as exc:
        results.append(f"{source:12} FAIL {type(exc).__name__}: {str(exc)[:80]}  {url}")
    print(results[-1])
    time.sleep(5)

dbutils.notebook.exit("\n".join(results))
