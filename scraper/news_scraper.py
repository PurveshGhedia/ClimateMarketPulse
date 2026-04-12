"""
Direct news scraper for Economic Times Agriculture and Hindu Business Line.
Uses requests + BeautifulSoup for listing pages + trafilatura for article text.
No Playwright needed — both sites serve article links in plain HTML.

Usage:
    cd scraper
    python news_scraper.py

Target: ~2,500 articles across 2 outlets x 5 years

Outlets:
  et_agriculture   economictimes.indiatimes.com/news/economy/agriculture
  hindu_bl_agri    thehindubusinessline.com/economy/agri-business
  hindu_bl_market  thehindubusinessline.com/markets/commodities
"""

import re
import json
import hashlib
import requests
import time
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import trafilatura
from db import get_conn, init_db, insert_article
from keywords import keyword_prefilter

# ---------------------------------------------------------------------------
# Outlet configuration
# ---------------------------------------------------------------------------
OUTLETS = {
    "et_agriculture": {
        "base_url":    "https://economictimes.indiatimes.com",
        "section_url": "https://economictimes.indiatimes.com/news/economy/agriculture",
        # ET paginates via ?page=N
        "page_param":  "page",
        "start_page":  1,
        "max_pages":   120,   # 120 pages x ~10 articles = ~1,200 articles
        # Article link pattern
        "link_re":     re.compile(
            r'href=["\'](/news/economy/agriculture/[^"\']+/articleshow/\d+\.cms)["\']'),
        "article_url_prefix": "https://economictimes.indiatimes.com",
        # Date filter — ET URLs don't contain dates, use trafilatura
        "date_in_url": False,
    },
    "hindu_bl_agri": {
        "base_url":    "https://www.thehindubusinessline.com",
        "section_url": "https://www.thehindubusinessline.com/economy/agri-business",
        # Hindu BL paginates via ?start=N (increments of 20)
        "page_param":  "start",
        "start_page":  0,
        "page_size":   20,
        "max_pages":   80,    # 80 pages x 20 articles = ~1,600 articles
        "link_re":     re.compile(
            r'href=["\']((https://www\.thehindubusinessline\.com)?'
            r'/economy/agri-business/[^"\']+/article\d+\.ece)["\']'),
        "article_url_prefix": "",
        "date_in_url": False,
    },
    "hindu_bl_market": {
        "base_url":    "https://www.thehindubusinessline.com",
        "section_url": "https://www.thehindubusinessline.com/markets/commodities",
        "page_param":  "start",
        "start_page":  0,
        "page_size":   20,
        "max_pages":   40,
        "link_re":     re.compile(
            r'href=["\']((https://www\.thehindubusinessline\.com)?'
            r'/markets/commodities/[^"\']+/article\d+\.ece)["\']'),
        "article_url_prefix": "",
        "date_in_url": False,
    },
}

VALID_YEARS = set(range(2020, 2025))
MAX_RETRIES = 3
FETCH_TIMEOUT = 30
PAGE_DELAY = 4    # seconds between listing page fetches
ARTICLE_DELAY = 6    # seconds between article fetches
TARGET_PER_OUTLET = 900   # 3 outlets x 900 = 2,700 ceiling

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",   # explicitly exclude brotli
})


# ---------------------------------------------------------------------------
# Listing page fetching
# ---------------------------------------------------------------------------

def get_listing_page(url, attempt=1):
    """Fetch a section listing page and return BeautifulSoup."""
    try:
        resp = session.get(url, timeout=FETCH_TIMEOUT)
        if resp.status_code == 200:
            return BeautifulSoup(resp.text, "html.parser")
        elif resp.status_code == 429:
            wait = 60 * attempt
            print(f"    Rate limited — waiting {wait}s...")
            time.sleep(wait)
            if attempt < MAX_RETRIES:
                return get_listing_page(url, attempt + 1)
        else:
            print(f"    Listing page {resp.status_code}: {url[:80]}")
        return None
    except Exception as e:
        print(f"    Listing fetch error: {e}")
        return None


def extract_article_links(soup, outlet_cfg):
    """Extract article URLs from a listing page."""
    html = str(soup)
    matches = outlet_cfg["link_re"].findall(html)
    urls = []
    prefix = outlet_cfg["article_url_prefix"]
    for m in matches:
        url = m[0] if isinstance(m, tuple) else m
        # Strip AMP, UTM, query params
        url = url.split("?")[0].split("#")[0]
        if url.endswith("/amp") or "/amp/" in url:
            continue
        if not url.startswith("http"):
            url = prefix + url
        if url not in urls:
            urls.append(url)
    return urls


def get_page_url(outlet_cfg, page_num):
    """Build paginated URL for a given page number."""
    base = outlet_cfg["section_url"]
    param = outlet_cfg["page_param"]
    if param == "page":
        if page_num == outlet_cfg["start_page"]:
            return base
        return f"{base}/{page_num}"
    elif param == "start":
        offset = page_num * outlet_cfg.get("page_size", 20)
        if offset == 0:
            return base
        return f"{base}?start={offset}"
    return base


# ---------------------------------------------------------------------------
# Article fetching
# ---------------------------------------------------------------------------

def fetch_article(url):
    """
    Fetch article and extract text + metadata via trafilatura 2.0.0.
    Returns (text, publish_date, title, author) or Nones on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=FETCH_TIMEOUT)

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
                time.sleep(15 * attempt)
        except Exception as e:
            print(f"    Fetch error: {e}")
            return None, None, None, None

    return None, None, None, None


# ---------------------------------------------------------------------------
# Per-outlet scraping
# ---------------------------------------------------------------------------

def scrape_outlet(outlet_name, outlet_cfg, conn, target=TARGET_PER_OUTLET):
    print(f"\n{'='*55}")
    print(f"  {outlet_name.upper()}")
    print(f"{'='*55}")

    saved = skipped_short = skipped_date = skipped_keywords = skipped_dup = 0
    seen_urls = set()

    for page_num in range(outlet_cfg["start_page"],
                          outlet_cfg["start_page"] + outlet_cfg["max_pages"]):
        if saved >= target:
            break

        page_url = get_page_url(outlet_cfg, page_num)
        print(f"\n  Page {page_num} — {page_url[:80]}")

        soup = get_listing_page(page_url)
        if not soup:
            print(f"  Failed to fetch listing page — stopping this outlet")
            break

        article_urls = extract_article_links(soup, outlet_cfg)
        print(f"  Found {len(article_urls)} article links on page")

        if not article_urls:
            print(f"  No articles found — likely end of pagination")
            break

        for url in article_urls:
            if saved >= target:
                break
            if url in seen_urls:
                continue
            seen_urls.add(url)

            print(f"  → {url[:78]}...")
            text, publish_date, title, author = fetch_article(url)

            if not text or len(text.split()) < 80:
                skipped_short += 1
                print(f"     skip: too short / empty")
                time.sleep(ARTICLE_DELAY)
                continue

            # Date filter — only keep 2020-2024
            if publish_date:
                try:
                    pub_year = int(publish_date[:4])
                    if pub_year not in VALID_YEARS:
                        skipped_date += 1
                        print(f"     skip: {publish_date} outside 2020-2024")
                        time.sleep(ARTICLE_DELAY)
                        continue
                except (ValueError, TypeError):
                    pass
            else:
                # No date extracted — skip, don't store undated articles
                skipped_date += 1
                print(f"     skip: no date extracted")
                time.sleep(ARTICLE_DELAY)
                continue

            is_relevant, climate_hits, commodity_hits = keyword_prefilter(text)
            if not is_relevant:
                skipped_keywords += 1
                print(f"     skip: no climate+commodity keyword overlap")
                time.sleep(ARTICLE_DELAY)
                continue

            inserted = insert_article(conn, {
                "url":                   url,
                "source_type":           "direct",
                "outlet":                outlet_name,
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
                print(f"     SAVED #{saved} [{publish_date}] — "
                      f"climate:{len(climate_hits)} commodity:{len(commodity_hits)}")
            else:
                skipped_dup += 1
                print(f"     skip: duplicate")

            time.sleep(ARTICLE_DELAY)

        time.sleep(PAGE_DELAY)

    print(f"\n  Result : {saved} saved | "
          f"{skipped_short} too-short | "
          f"{skipped_date} wrong-year/no-date | "
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
    for outlet_name, outlet_cfg in OUTLETS.items():
        count = scrape_outlet(outlet_name, outlet_cfg, conn)
        grand_total += count
        print(f"\n  Running total: {grand_total} articles")
        time.sleep(10)   # pause between outlets

    conn.close()
    print(f"\n{'='*55}")
    print(f"  Direct scraping complete — {grand_total} total articles")
    print(f"{'='*55}")


if __name__ == "__main__":
    run()
