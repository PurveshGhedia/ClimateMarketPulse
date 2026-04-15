"""
et_scraper.py

Economic Times agriculture scraper — 5 year historical run.
Collects article URLs from ET's archive pages, fetches full text,
and stores directly into articles.db using the shared insert_article()
function — same format as all other scrapers in this project.

Architecture:
    Phase 1 (Playwright) — navigate archive page, wait 12s for
                           ad-timer to clear, extract article URLs
    Phase 2 (requests)   — fetch full article text from each URL
                           using trafilatura (no JS needed for article pages)

Usage:
    python et_scraper.py

Target: ~2,000 articles over 5 years (2021-2026)
"""

import re
import time
import os
import requests
import trafilatura
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

from db import get_conn, init_db, insert_article
from keywords import keyword_prefilter

# ── Configuration ─────────────────────────────────────────────────

END_DATE = datetime(2026, 4, 12)
START_DATE = END_DATE - timedelta(days=4 * 365)

# Temporarily change at top of et_scraper.py:
# START_DATE = datetime(2026, 4, 1)
# END_DATE = datetime(2026, 4, 2)

ET_BASE_DATE = datetime(1899, 12, 30)

# Matches ET agriculture section article URLs
AG_PATTERN = re.compile(r'[^"]*agriculture[^"]*articleshow/[0-9]+\.cms')

# Additional topic patterns relevant to your project
TOPIC_PATTERNS = [
    re.compile(r'[^"]*economy/agriculture[^"]*articleshow/[0-9]+\.cms'),
    re.compile(r'[^"]*news/economy/agriculture[^"]*articleshow/[0-9]+\.cms'),
    # REMOVED: re.compile(r'[^"]*markets/commodities[^"]*articleshow/[0-9]+\.cms'),
]

IRRELEVANT_URL_TERMS = [
    "crude-oil", "brent", "gold-price", "silver-price",
    "natural-gas", "copper", "zinc", "aluminium",
    "sensex", "nifty", "stock-market", "equity",
]
OUTLET = "economic_times"
SOURCE_TYPE = "playwright"


# ── Helpers ───────────────────────────────────────────────────────

def get_starttime(target_date):
    """ET archive uses days-since-1899-12-30 as starttime param."""
    return (target_date - ET_BASE_DATE).days


def extract_headline(soup):
    """Extract article headline from ET article page."""
    for selector in [
        soup.find("h1", class_=re.compile(r"artTitle|heading")),
        soup.find("h1"),
        soup.find("title"),
    ]:
        if selector:
            return selector.get_text(strip=True)
    return None


def fetch_article_text(url):
    """
    Fetch full article text using trafilatura.
    ET article pages are static - no Playwright needed.
    Returns (headline, full_text) or (None, None) on failure.
    """
    try:
        resp = requests.get(
            url, timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (academic research)",
                "Accept-Encoding": "gzip, deflate",
            }
        )
        if resp.status_code != 200:
            return None, None

        # trafilatura for clean body text
        full_text = trafilatura.extract(resp.text)

        # BeautifulSoup for headline
        soup = BeautifulSoup(resp.text, "html.parser")
        headline = extract_headline(soup)

        return headline, full_text

    except Exception as e:
        print(f"      Fetch error: {e}")
        return None, None


def extract_date_from_url(url, fallback_date):
    """
    ET article URLs sometimes contain dates.
    e.g. /articleshow/129935461.cms — no date in URL, use archive date.
    """
    return fallback_date.strftime("%Y-%m-%d")


# ── Main scrape loop ──────────────────────────────────────────────

def run():
    print("Initialising database...")
    init_db()
    conn = get_conn()

    print(f"\nScraping Economic Times agriculture section")
    print(
        f"Period: {START_DATE.strftime('%Y-%m-%d')} → {END_DATE.strftime('%Y-%m-%d')}")
    print(f"{'='*55}")

    grand_total = 0
    skipped_short = 0
    skipped_kw = 0
    skipped_dup = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"
        )

        curr_date = START_DATE

        while curr_date <= END_DATE:
            st = get_starttime(curr_date)
            year = curr_date.year
            month = curr_date.month
            date_str = curr_date.strftime("%Y-%m-%d")
            archive_url = (
                f"https://economictimes.indiatimes.com/"
                f"archivelist/year-{year},month-{month},starttime-{st}.cms"
            )

            try:
                # ── Phase 1: collect article URLs from archive page ──
                page.goto(archive_url, wait_until="domcontentloaded",
                          timeout=60000)
                page.wait_for_timeout(12000)   # wait for ad-timer to clear

                html = page.content()
                soup = BeautifulSoup(html, "html.parser")
                links = soup.find_all("a", href=True)
                article_urls = set()

                for link in links:
                    href = link["href"]
                    for pattern in TOPIC_PATTERNS:
                        if pattern.search(href):
                            if href.startswith("/"):
                                href = f"https://economictimes.indiatimes.com{href}"
                            # Skip financial commodities — not relevant to project
                            if not any(term in href.lower() for term in IRRELEVANT_URL_TERMS):
                                article_urls.add(href)
                            break

                print(f"\n[{date_str}] Found {len(article_urls)} article URLs")

                # ── Phase 2: fetch full text for each URL ────────────
                day_saved = 0
                for url in article_urls:
                    headline, full_text = fetch_article_text(url)

                    if not full_text or len(full_text.split()) < 80:
                        skipped_short += 1
                        continue

                    is_rel, climate_hits, commodity_hits = keyword_prefilter(
                        full_text)
                    if not is_rel:
                        skipped_kw += 1
                        continue

                    publish_date = extract_date_from_url(url, curr_date)

                    inserted = insert_article(conn, {
                        "url":                   url,
                        "source_type":           SOURCE_TYPE,
                        "outlet":                OUTLET,
                        "publish_date":          publish_date,
                        "headline":              headline,
                        "full_text":             full_text,
                        "climate_terms_found":   "|".join(climate_hits),
                        "commodity_terms_found": "|".join(commodity_hits),
                        "extraction_method":     "playwright+trafilatura",
                    })

                    if inserted:
                        grand_total += 1
                        day_saved += 1
                        print(
                            f"  SAVED #{grand_total}: {(headline or url[:60])[:65]}")
                    else:
                        skipped_dup += 1

                    time.sleep(1)   # polite delay between article fetches

                print(f"  [{date_str}] Saved {day_saved} | "
                      f"Total so far: {grand_total}")

            except Exception as e:
                print(f"  [{date_str}] ERROR: {e}")

            curr_date += timedelta(days=1)

        browser.close()

    conn.close()

    print(f"\n{'='*55}")
    print(f"  ET scraping complete")
    print(f"  Saved:           {grand_total}")
    print(f"  Skipped short:   {skipped_short}")
    print(f"  Skipped no-kw:   {skipped_kw}")
    print(f"  Skipped dupes:   {skipped_dup}")
    print(f"{'='*55}")


if __name__ == "__main__":
    run()
