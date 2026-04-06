"""
Wayback Machine scraper for Indian newspaper archives (2020-2024).
No Playwright needed — works with plain requests + trafilatura.

Usage:
    python wayback_scraper.py

Target: ~1,250 articles (50 per outlet per year, 5 outlets x 5 years)
"""

import requests
import time
import trafilatura
from db import get_conn, init_db, insert_article
from keywords import keyword_prefilter, URL_HINTS

OUTLETS = {
    "hindu_bl":          "thehindubusinessline.com",
    "mint":              "livemint.com",
    "financial_express": "financialexpress.com",
    "indian_express":    "indianexpress.com",
    "toi":               "timesofindia.indiatimes.com",
}

CDX_API = "http://web.archive.org/cdx/search/cdx"


def get_archived_urls(domain, year, max_per_hint=30):
    """
    Query the Wayback CDX API for relevant archived article URLs.
    We search per URL hint to bias results toward relevant content.
    """
    all_records = []
    seen_urls = set()

    for hint in URL_HINTS[:10]:   # top 10 hints — stay polite
        params = {
            "url":      f"{domain}/*{hint}*",
            "output":   "json",
            "fl":       "original,timestamp,statuscode",
            "filter":   "statuscode:200",
            "from":     f"{year}0101",
            "to":       f"{year}1231",
            "limit":    max_per_hint,
            "collapse": "urlkey",
        }
        try:
            resp = requests.get(CDX_API, params=params, timeout=30)
            if resp.status_code == 200 and resp.text.strip():
                rows = resp.json()
                for row in rows[1:]:  # skip header
                    if len(row) >= 2 and row[0] not in seen_urls:
                        seen_urls.add(row[0])
                        all_records.append(row)
        except Exception as e:
            print(f"    CDX query error for hint '{hint}': {e}")

        time.sleep(1.5)   # polite delay between CDX queries

    return all_records


def fetch_article(original_url, timestamp):
    """Fetch an archived article and extract clean text using trafilatura."""
    archive_url = f"http://web.archive.org/web/{timestamp}/{original_url}"
    try:
        resp = requests.get(
            archive_url, timeout=25,
            headers={"User-Agent": "Mozilla/5.0 (academic research)"}
        )
        if resp.status_code != 200:
            return None, None
        text = trafilatura.extract(resp.text, include_metadata=False,
                                   no_fallback=False)
        return text, archive_url
    except Exception as e:
        print(f"    Fetch error: {e}")
        return None, None


def scrape_outlet_year(outlet_name, domain, year, conn, target=50):
    """Scrape one outlet for one calendar year."""
    print(f"\n{'─'*55}")
    print(f"  {outlet_name.upper()} | {year}")
    print(f"{'─'*55}")

    records = get_archived_urls(domain, year)
    print(f"  CDX returned {len(records)} candidate URLs")

    saved, skipped_short, skipped_keywords, skipped_dup = 0, 0, 0, 0

    for record in records:
        if saved >= target:
            break
        if len(record) < 2:
            continue

        original_url, timestamp = record[0], record[1]

        # Quick URL-level gate before spending a network request
        url_ok = any(h in original_url.lower() for h in URL_HINTS)
        if not url_ok:
            continue

        print(f"  → {original_url[:75]}...")
        text, archive_url = fetch_article(original_url, timestamp)

        if not text or len(text.split()) < 80:
            skipped_short += 1
            print(f"     skip: too short / empty")
            continue

        is_relevant, climate_hits, commodity_hits = keyword_prefilter(text)
        if not is_relevant:
            skipped_keywords += 1
            print(f"     skip: no climate+commodity keyword overlap")
            continue

        publish_date = (f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
                        if len(timestamp) >= 8 else None)

        inserted = insert_article(conn, {
            "url":                  original_url,
            "source_type":          "wayback",
            "outlet":               outlet_name,
            "publish_date":         publish_date,
            "full_text":            text,
            "climate_terms_found":  "|".join(climate_hits),
            "commodity_terms_found": "|".join(commodity_hits),
            "archive_url":          archive_url,
            "extraction_method":    "trafilatura",
        })

        if inserted:
            saved += 1
            print(f"     SAVED #{saved} — climate:{len(climate_hits)} commodity:{len(commodity_hits)}")
        else:
            skipped_dup += 1
            print(f"     skip: duplicate")

        time.sleep(2)   # polite delay between article fetches

    print(f"\n  Result: {saved} saved | "
          f"{skipped_short} too-short | "
          f"{skipped_keywords} no-keywords | "
          f"{skipped_dup} duplicates")
    return saved


def run():
    print("Initialising database...")
    init_db()
    conn = get_conn()

    grand_total = 0
    for year in range(2020, 2025):
        year_total = 0
        for outlet_name, domain in OUTLETS.items():
            count = scrape_outlet_year(
                outlet_name, domain, year, conn, target=50
            )
            year_total  += count
            grand_total += count
            print(f"\n  Running total: {grand_total} articles")

        print(f"\n  Year {year} complete: {year_total} articles")

    conn.close()
    print(f"\n{'='*55}")
    print(f"  Wayback scraping complete — {grand_total} total articles")
    print(f"{'='*55}")


if __name__ == "__main__":
    run()
