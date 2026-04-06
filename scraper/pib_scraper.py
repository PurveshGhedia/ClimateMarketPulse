"""
PIB (Press Information Bureau) scraper.
Scrapes Agriculture Ministry press releases — clean HTML, no JS rendering needed.

Usage:
    python pib_scraper.py

Target: ~400-600 relevant articles (2020-2024)
"""

import requests
from bs4 import BeautifulSoup
import time
import re
from db import get_conn, init_db, insert_article
from keywords import keyword_prefilter

PIB_BASE     = "https://pib.gov.in"
PIB_LIST_URL = "https://pib.gov.in/allRel.aspx"

# Ministry codes relevant to your project
MINISTRIES = {
    "agriculture": "7",    # Agriculture & Farmers Welfare
    "food":        "29",   # Food & Consumer Affairs
    "environment": "13",   # Environment, Forest & Climate Change
}


def get_press_release_links(ministry_code, year, month):
    """Fetch list of press release links for a ministry/month."""
    params = {
        "mncode": ministry_code,
        "yr":     str(year),
        "mn":     str(month).zfill(2),
    }
    try:
        resp = requests.get(PIB_LIST_URL, params=params, timeout=20,
                            headers={"User-Agent": "Mozilla/5.0 (academic research)"})
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # PIB article links contain 'PressReleasePage' or 'prid='
            if "PressReleasePage" in href or "prid=" in href:
                full_url = (PIB_BASE + href) if href.startswith("/") else href
                headline = a.get_text(strip=True)
                if headline:
                    links.append((full_url, headline))
        return links
    except Exception as e:
        print(f"  PIB list fetch failed ({year}-{month:02d}): {e}")
        return []


def fetch_press_release_text(url):
    """Extract clean text from a PIB press release page."""
    try:
        resp = requests.get(url, timeout=20,
                            headers={"User-Agent": "Mozilla/5.0 (academic research)"})
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try known PIB content containers in order of reliability
        content = (
            soup.find("div", {"id": "WriteReadData"}) or
            soup.find("div", class_="innerpageheading") or
            soup.find("div", class_="content") or
            soup.find("article")
        )

        if content:
            # Remove script/style noise
            for tag in content(["script", "style", "nav"]):
                tag.decompose()
            return content.get_text(separator=" ", strip=True)

        # Fallback: grab all paragraph text
        paras = soup.find_all("p")
        return " ".join(p.get_text(strip=True) for p in paras if len(p.get_text()) > 40)

    except Exception as e:
        print(f"  PIB article fetch failed: {e}")
        return None


def extract_date_from_url(url, year, month):
    """Try to parse exact date from URL, fall back to year-month."""
    m = re.search(r"(\d{4})(\d{2})(\d{2})", url)
    if m and m.group(1) == str(year):
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return f"{year}-{month:02d}-01"


def run():
    print("Initialising database...")
    init_db()
    conn = get_conn()

    grand_total = 0

    for ministry_name, ministry_code in MINISTRIES.items():
        print(f"\n{'='*55}")
        print(f"  Ministry: {ministry_name.upper()}")
        print(f"{'='*55}")

        for year in range(2020, 2025):
            for month in range(1, 13):
                print(f"\n  PIB {ministry_name} | {year}-{month:02d}")
                links = get_press_release_links(ministry_code, year, month)
                print(f"  Found {len(links)} press releases")

                for url, headline in links:
                    # Quick headline-level filter first (fast, no network call)
                    hl_relevant, _, _ = keyword_prefilter(headline)
                    # Fetch full text regardless — headlines often don't mention all terms
                    text = fetch_press_release_text(url)

                    if not text or len(text.split()) < 50:
                        print(f"    skip: too short — {headline[:50]}")
                        continue

                    is_relevant, climate_hits, commodity_hits = keyword_prefilter(text)
                    if not is_relevant:
                        print(f"    skip: no keyword match — {headline[:50]}")
                        continue

                    publish_date = extract_date_from_url(url, year, month)

                    inserted = insert_article(conn, {
                        "url":                   url,
                        "source_type":           "pib",
                        "outlet":                "pib",
                        "publish_date":          publish_date,
                        "headline":              headline,
                        "full_text":             text,
                        "climate_terms_found":   "|".join(climate_hits),
                        "commodity_terms_found": "|".join(commodity_hits),
                        "extraction_method":     "beautifulsoup",
                    })

                    if inserted:
                        grand_total += 1
                        print(f"    SAVED #{grand_total}: {headline[:60]}...")

                    time.sleep(1)

    conn.close()
    print(f"\n{'='*55}")
    print(f"  PIB scraping complete — {grand_total} articles saved")
    print(f"{'='*55}")


if __name__ == "__main__":
    run()
