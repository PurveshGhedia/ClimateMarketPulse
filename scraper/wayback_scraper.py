"""
Wayback Machine scraper for Indian newspaper archives (2020-2024).
Works with plain requests + trafilatura 2.0.0. No Playwright needed.

Usage:
    cd scraper
    python wayback_scraper.py

Target: ~1,250 articles across 5 outlets x 5 years

CDX design (validated via curl):
  - Pattern : www.domain/section/ + matchType=prefix + collapse=urlkey
  - Timestamp regex filter on CDX (filter=timestamp:YYYY.*) ensures we only
    pull URLs first archived in the target year — best available proxy for
    publish year when no date is in the URL slug.
  - Trafilatura 2.0.0 with with_metadata=True extracts real publish date from
    article HTML — used to store accurate publish_date and do a final
    year-range sanity check (discard if outside 2020-2024).
  - AMP / lite / UTM / image URLs filtered by per-outlet regex before fetching.
  - Keyword filtering (climate AND commodity) done on full text via
    keyword_prefilter() — unchanged from original design.

Outlets (mint dropped — livemint.com not reliably indexed by Wayback):
  hindu_bl          thehindubusinessline.com
  financial_express financialexpress.com
  indian_express    indianexpress.com
  toi               timesofindia.indiatimes.com
  down_to_earth     downtoearth.org.in
"""

import re
import json
import requests
import time
import trafilatura
from db import get_conn, init_db, insert_article
from keywords import keyword_prefilter, URL_HINTS

# ---------------------------------------------------------------------------
# Per-outlet CDX configuration (every entry validated via curl)
#
# domain      : exact www. prefix as indexed by Wayback
# sections    : queried independently, results merged + deduped
# article_re  : must match URL — filters section index pages
# skip_re     : drops AMP, lite, UTM, image, gallery variants
# ---------------------------------------------------------------------------
OUTLETS = {
    "hindu_bl": {
        "domain": "www.thehindubusinessline.com",
        "sections": [
            "economy/agri-business/",
            "economy/macro-economy/",
            "markets/commodities/",
        ],
        "article_re": re.compile(r"/article\d+\.ece$"),
        "skip_re":    re.compile(
            r"/amp/|/gallery/|/video/|/ALTERNATES/|\.jpg|\.png"),
    },
    "financial_express": {
        "domain": "www.financialexpress.com",
        "sections": [
            "economy/",
            "market/",
            "agriculture/",
        ],
        "article_re": re.compile(r"/\d{5,}/$"),
        "skip_re":    re.compile(r"/lite/|utm_|ia_markup|\?"),
    },
    "indian_express": {
        "domain": "indianexpress.com",
        "sections": [
            "article/india/",
            "article/explained/",
            "article/business/",
        ],
        "article_re": re.compile(r"-\d{6,}/$"),
        "skip_re":    re.compile(r"/%20|/amp/|/lite/|\?"),
    },
    "toi": {
        "domain": "timesofindia.indiatimes.com",
        "sections": [
            "business/india-business/",
            "business/economy/",
            "india/",
        ],
        "article_re": re.compile(r"/articleshow/\d+\.cms$"),
        "skip_re":    re.compile(r"from=mdr|\?|/photo/|/video/"),
    },
    "down_to_earth": {
        "domain": "www.downtoearth.org.in",
        "sections": [
            "news/agriculture/",
            "news/climate-change/",
            "news/food/",
            "news/water/",
        ],
        "article_re": re.compile(r"-\d{4,}$"),   # ends with numeric slug ID
        "skip_re":    re.compile(r"/amp/|\?|/gallery/"),
    },
}

YEARS = range(2020, 2025)
CDX_API = "https://web.archive.org/cdx/search/cdx"
MAX_RETRIES = 4
CDX_TIMEOUT = 60
FETCH_TIMEOUT = 60                  # raised from 40 — Wayback can be slow
CDX_LIMIT = 500
TARGET_PER_OUTLET_YEAR = 55
VALID_YEARS = set(range(2020, 2025))

FETCH_DELAY = 10   # seconds between every article fetch
COOLDOWN_EVERY = 10   # pause after every N fetches
COOLDOWN_DURATION = 45   # seconds — lets Wayback rate limit window reset


# ---------------------------------------------------------------------------
# CDX querying
# ---------------------------------------------------------------------------

def get_archived_urls(domain, section, year):
    """
    Query CDX for one domain/section/year.
    Uses timestamp regex filter so we only get URLs first archived in
    that year — best available proxy for publish year.
    Returns list of [original_url, timestamp] rows (header stripped).
    """
    params = {
        "url":        f"{domain}/{section}",
        "matchType":  "prefix",
        "output":     "json",
        "fl":         "original,timestamp",
        "filter":    [
            "statuscode:200",
            f"timestamp:{year}.*",   # only URLs first crawled this year
        ],
        "from":       f"{year}0101",
        "to":         f"{year}1231",
        "limit":      CDX_LIMIT,
        "collapse":   "urlkey",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                CDX_API, params=params,
                timeout=CDX_TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0 (academic research)"}
            )
            if resp.status_code == 200 and resp.text.strip():
                rows = resp.json()
                return rows[1:]   # strip ["original","timestamp"] header
            return []

        except requests.exceptions.Timeout:
            print(f"    CDX timeout (attempt {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                wait = attempt * 10
                print(f"    Waiting {wait}s before retry...")
                time.sleep(wait)
        except Exception as e:
            print(f"    CDX error: {e}")
            return []

    return []


def collect_candidates(outlet_cfg, year):
    """
    Query all sections, merge, deduplicate, apply article_re + skip_re
    and URL-level topic hint gate.
    Returns list of (url, timestamp) tuples ready for fetching.
    """
    article_re = outlet_cfg["article_re"]
    skip_re = outlet_cfg["skip_re"]
    domain = outlet_cfg["domain"]

    seen = set()
    results = []

    for section in outlet_cfg["sections"]:
        rows = get_archived_urls(domain, section, year)
        print(f"    /{section} → {len(rows)} raw CDX rows")

        for row in rows:
            if len(row) < 2:
                continue
            url, ts = row[0], row[1]

            if url in seen:
                continue
            seen.add(url)

            if not article_re.search(url):
                continue

            if skip_re.search(url):
                continue

            # URL-level topic gate — at least one hint in the URL
            if not any(h in url.lower() for h in URL_HINTS):
                continue

            results.append((url, ts))

        time.sleep(2)

    return results


# ---------------------------------------------------------------------------
# Article fetching + metadata extraction
# ---------------------------------------------------------------------------

def make_session():
    """Create a requests session with retry-on-connection-error."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (academic research)"})
    return session


def fetch_article(session, original_url, timestamp):
    """
    Fetch archived article from Wayback and extract text + metadata.
    Uses trafilatura 2.0.0 with with_metadata=True.
    Returns (text, publish_date, title, author, archive_url) or Nones on failure.
    """
    archive_url = f"https://web.archive.org/web/{timestamp}/{original_url}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(archive_url, timeout=FETCH_TIMEOUT)

            if resp.status_code == 429:
                wait = 60 * attempt
                print(f"    Rate limited (429) — waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                return None, None, None, None, None

            raw = trafilatura.extract(
                resp.text,
                output_format="json",
                with_metadata=True,
                url=archive_url,
            )

            if not raw:
                return None, None, None, None, None

            data = json.loads(raw)
            text = data.get("text", "")
            publish_date = data.get("date")
            title = data.get("title")
            author = data.get("author")

            return text, publish_date, title, author, archive_url

        except requests.exceptions.ConnectionError as e:
            print(
                f"    Connection error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                wait = 30 * attempt   # 30s, 60s, 90s backoff
                print(f"    Waiting {wait}s before retry...")
                time.sleep(wait)
        except requests.exceptions.Timeout:
            print(f"    Fetch timeout (attempt {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(attempt * 10)
        except Exception as e:
            print(f"    Fetch error: {e}")
            return None, None, None, None, None

    return None, None, None, None, None


# ---------------------------------------------------------------------------
# Per-outlet-year scraping
# ---------------------------------------------------------------------------

def scrape_outlet_year(outlet_name, outlet_cfg, year, conn,
                       target=TARGET_PER_OUTLET_YEAR):
    print(f"\n{'─'*55}")
    print(f"  {outlet_name.upper()} | {year}")
    print(f"{'─'*55}")

    candidates = collect_candidates(outlet_cfg, year)
    print(f"  {len(candidates)} candidates after URL filtering")

    if not candidates:
        print("  No candidates — skipping")
        return 0

    session = make_session()
    saved = skipped_short = skipped_date = skipped_keywords = skipped_dup = 0
    fetch_count = 0   # total fetches attempted (for cooldown pacing)

    for url, timestamp in candidates:
        if saved >= target:
            break

        # Cooldown pause every N fetches to avoid Wayback rate limiting
        if fetch_count > 0 and fetch_count % COOLDOWN_EVERY == 0:
            print(
                f"\n  --- cooldown {COOLDOWN_DURATION}s after {fetch_count} fetches ---\n")
            time.sleep(COOLDOWN_DURATION)

        print(f"  → {url[:78]}...")
        text, publish_date, title, author, archive_url = fetch_article(
            session, url, timestamp)
        fetch_count += 1

        if not text or len(text.split()) < 80:
            skipped_short += 1
            print(f"     skip: too short / empty")
            time.sleep(FETCH_DELAY)
            continue

        # Date sanity check
        if publish_date:
            try:
                pub_year = int(publish_date[:4])
                if pub_year not in VALID_YEARS:
                    skipped_date += 1
                    print(
                        f"     skip: publish_date {publish_date} outside 2020-2024")
                    time.sleep(FETCH_DELAY)
                    continue
            except (ValueError, TypeError):
                pass
        else:
            publish_date = (f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
                            if len(timestamp) >= 8 else None)

        is_relevant, climate_hits, commodity_hits = keyword_prefilter(text)
        if not is_relevant:
            skipped_keywords += 1
            print(f"     skip: no climate+commodity keyword overlap")
            time.sleep(FETCH_DELAY)
            continue

        inserted = insert_article(conn, {
            "url":                   url,
            "source_type":           "wayback",
            "outlet":                outlet_name,
            "headline":              title,
            "author":                author,
            "publish_date":          publish_date,
            "full_text":             text,
            "climate_terms_found":   "|".join(climate_hits),
            "commodity_terms_found": "|".join(commodity_hits),
            "archive_url":           archive_url,
            "extraction_method":     "trafilatura",
        })

        if inserted:
            saved += 1
            print(f"     SAVED #{saved} [{publish_date}] — "
                  f"climate:{len(climate_hits)} commodity:{len(commodity_hits)}")
        else:
            skipped_dup += 1
            print(f"     skip: duplicate")

        time.sleep(FETCH_DELAY)   # polite delay between every fetch

    print(f"\n  Result : {saved} saved | "
          f"{skipped_short} too-short | "
          f"{skipped_date} wrong-year | "
          f"{skipped_keywords} no-keywords | "
          f"{skipped_dup} duplicates")
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("Initialising database...")
    init_db()
    conn = get_conn()

    grand_total = 0
    for year in YEARS:
        year_total = 0
        for outlet_name, outlet_cfg in OUTLETS.items():
            count = scrape_outlet_year(
                outlet_name, outlet_cfg, year, conn
            )
            year_total += count
            grand_total += count
            print(f"\n  Running total: {grand_total} articles")
            time.sleep(5)

        print(f"\n  ── Year {year} complete: {year_total} articles ──")

    conn.close()
    print(f"\n{'='*55}")
    print(f"  Wayback scraping complete — {grand_total} total articles")
    print(f"{'='*55}")


if __name__ == "__main__":
    run()
