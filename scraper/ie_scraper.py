"""
Indian Express daily archive scraper (2020-2024).
Uses date-based archive URLs: indianexpress.com/archive/YYYY/MM/DD/
No JS rendering needed — archive pages are plain HTML.

Usage:
    cd scraper
    python ie_scraper.py

Target: ~2,000 articles across 2020-2024
Strategy: scrape every day to cover full date range efficiently
          365 days / 3 = ~122 pages per year x 5 years = 610 pages
          ~44 links per page x 20% keyword hit rate = ~5,368 candidates
"""

import re
import json
import requests
import time
from datetime import date, timedelta
import trafilatura
from db import get_conn, init_db, insert_article
from keywords import keyword_prefilter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
START_DATE = date(2020, 1, 1)
END_DATE = date(2024, 12, 31)
DATE_STEP = 1          # scrape every 2nd day for good coverage
# TARGET = float('inf')       # stop after this many saved articles
VALID_YEARS = set(range(2020, 2025))
MAX_RETRIES = 3
PAGE_DELAY = 3          # seconds between archive page fetches
ARTICLE_DELAY = 2
COOLDOWN_EVERY = 100
COOLDOWN_DURATION = 20

BASE_URL = "https://indianexpress.com/archive/{year}/{month}/{day}/"
BASE_URL_PAGE = "https://indianexpress.com/archive/{year}/{month}/{day}/page/{page}/"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
})

ARTICLE_RE = re.compile(
    r'href=["\'](https://indianexpress\.com/article/[^"\'\s<>]+)["\']')


# ---------------------------------------------------------------------------
# Archive page fetching
# ---------------------------------------------------------------------------

def fetch_page(url):
    """Fetch a single URL and return response text or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                return None   # page doesn't exist
            if resp.status_code == 429:
                wait = 60 * attempt
                print(f"    Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return None
            return resp.text
        except requests.exceptions.ConnectionError as e:
            print(f"    Connection error (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(30 * attempt)
        except requests.exceptions.Timeout:
            print(f"    Timeout (attempt {attempt})")
            if attempt < MAX_RETRIES:
                time.sleep(15)
        except Exception as e:
            print(f"    Error: {e}")
            return None
    return None


def get_archive_links(archive_date):
    """
    Fetch all pages of a daily archive and return deduplicated article URLs.
    IE paginates as /archive/YYYY/MM/DD/page/N/
    """
    seen = set()
    all_links = []
    page = 1

    while True:
        if page == 1:
            url = BASE_URL.format(
                year=archive_date.year,
                month=str(archive_date.month).zfill(2),
                day=str(archive_date.day).zfill(2),
            )
        else:
            url = BASE_URL_PAGE.format(
                year=archive_date.year,
                month=str(archive_date.month).zfill(2),
                day=str(archive_date.day).zfill(2),
                page=page,
            )

        html = fetch_page(url)
        if not html:
            break   # 404 or error — no more pages

        links = ARTICLE_RE.findall(html)
        new_links = [l for l in links if l not in seen]

        if not new_links:
            break   # no new links on this page — stop paginating

        for l in new_links:
            seen.add(l)
            all_links.append(l)

        page += 1
        time.sleep(1)   # small delay between pagination requests

    return all_links


# ---------------------------------------------------------------------------
# Article fetching
# ---------------------------------------------------------------------------

def fetch_article(url):
    """Fetch article and extract text + metadata via trafilatura 2.0.0."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=30)

            if resp.status_code == 429:
                wait = 60 * attempt
                print(f"    Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return None, None, None, None

            raw = trafilatura.extract(
                resp.text,
                output_format="json",
                with_metadata=True,
                url=url,
            )
            if not raw:
                return None, None, None, None

            data = json.loads(raw)
            return (
                data.get("text", ""),
                data.get("date"),
                data.get("title"),
                data.get("author"),
            )

        except requests.exceptions.ConnectionError as e:
            print(f"    Connection error (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(30 * attempt)
        except requests.exceptions.Timeout:
            print(f"    Timeout (attempt {attempt})")
            if attempt < MAX_RETRIES:
                time.sleep(15)
        except Exception as e:
            print(f"    Fetch error: {e}")
            return None, None, None, None

    return None, None, None, None


# ---------------------------------------------------------------------------
# Main scrape loop
# ---------------------------------------------------------------------------

def run():
    print("Initialising database...")
    init_db()
    conn = get_conn()

    saved = 0
    skipped_short = skipped_date = skipped_keywords = skipped_dup = 0
    fetch_count = 0
    seen_urls = set()

    current_date = START_DATE
    total_days = (END_DATE - START_DATE).days // DATE_STEP

    print(f"\nScraping IE archive {START_DATE} → {END_DATE} "
          f"(every {DATE_STEP} days = ~{total_days} pages)")
    # print(f"Target: {TARGET} articles\n")

    while current_date <= END_DATE:
        # if saved >= TARGET:
        #     break

        archive_links = get_archive_links(current_date)
        print(f"  {current_date} — {len(archive_links)} links", end="")

        if not archive_links:
            print(" (empty)")
            current_date += timedelta(days=DATE_STEP)
            time.sleep(PAGE_DELAY)
            continue

        # Filter to topic-relevant URLs only using URL_HINTS
        from keywords import URL_HINTS
        relevant = [l for l in archive_links
                    if any(h in l.lower() for h in URL_HINTS)]
        print(f" → {len(relevant)} topic-relevant")

        for url in relevant:
            # if saved >= TARGET:
            #     break
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Cooldown every N fetches
            if fetch_count > 0 and fetch_count % COOLDOWN_EVERY == 0:
                print(f"\n  --- cooldown {COOLDOWN_DURATION}s "
                      f"after {fetch_count} fetches ---\n")
                time.sleep(COOLDOWN_DURATION)

            print(f"    → {url[:75]}...")
            text, publish_date, title, author = fetch_article(url)
            fetch_count += 1

            if not text or len(text.split()) < 80:
                skipped_short += 1
                time.sleep(ARTICLE_DELAY)
                continue

            # Date filter
            if publish_date:
                try:
                    pub_year = int(publish_date[:4])
                    if pub_year not in VALID_YEARS:
                        skipped_date += 1
                        time.sleep(ARTICLE_DELAY)
                        continue
                except (ValueError, TypeError):
                    pass
            else:
                # Use archive date as fallback
                publish_date = current_date.strftime("%Y-%m-%d")

            is_relevant, climate_hits, commodity_hits = keyword_prefilter(text)
            if not is_relevant:
                skipped_keywords += 1
                time.sleep(ARTICLE_DELAY)
                continue

            inserted = insert_article(conn, {
                "url":                   url,
                "source_type":           "direct",
                "outlet":                "indian_express",
                "headline":              title,
                "author":                author,
                "publish_date":          publish_date,
                "full_text":             text,
                "climate_terms_found":   "|".join(climate_hits),
                "commodity_terms_found": "|".join(commodity_hits),
                "extraction_method":     "trafilatura",
            })

            if inserted:
                saved += 1
                print(f"       SAVED #{saved} [{publish_date}] — "
                      f"climate:{len(climate_hits)} "
                      f"commodity:{len(commodity_hits)}")
            else:
                skipped_dup += 1

            time.sleep(ARTICLE_DELAY)

        current_date += timedelta(days=DATE_STEP)
        time.sleep(PAGE_DELAY)

    conn.close()
    print(f"\n{'='*55}")
    print(f"  IE scraping complete")
    print(f"  Saved     : {saved}")
    print(f"  Too short : {skipped_short}")
    print(f"  Wrong year: {skipped_date}")
    print(f"  No keywords: {skipped_keywords}")
    print(f"  Duplicates: {skipped_dup}")
    print(f"{'='*55}")


if __name__ == "__main__":
    run()
