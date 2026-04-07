"""
pib_playwright_scraper.py  (v4 - fixed content extraction)

Content page structure (confirmed from debug):
    <div id="ministry" class="mddiv content-ministry">  — ministry + date header
    <div class="contentdiv">                             — full article body

Full text IS accessible via plain requests on erelcontent.aspx?relid=XXXXX
Fix: target div.contentdiv specifically, strip nav noise.
"""

import asyncio
import re
import requests
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from db import get_conn, init_db, insert_article
from keywords import keyword_prefilter, pib_filter

PIB_URL = "https://archive.pib.gov.in/archive2/erelease.aspx?reg=3&lang=1"
CONTENT_URL = "https://archive.pib.gov.in/archive2/erelcontent.aspx?relid={relid}"

MINISTRIES = {
    "27":   "agriculture",
    "39":   "food_consumer",
    "30":   "environment",
    "38":   "water_resources",
    "67":   "earth_science",
    "1340": "fisheries_animal",
}

YEARS = ["2020", "2021", "2022", "2023", "2024"]
MONTHS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

MONTH_PAD = {str(i): str(i).zfill(2) for i in range(1, 13)}


# ── Fetch full article text ───────────────────────────────────────

def fetch_content(relid):
    """
    Fetch article from erelcontent.aspx — plain requests, no JS needed.
    Targets div.contentdiv which holds the full article body.
    Also captures ministry name and date from div#ministry.
    """
    url = CONTENT_URL.format(relid=relid)
    try:
        resp = requests.get(
            url, timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (academic research)"}
        )
        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Ministry + date from header div ──────────────────────
        ministry_div = soup.find("div", id="ministry")
        header_text = ministry_div.get_text(
            " ", strip=True) if ministry_div else ""

        # ── Article body from contentdiv ─────────────────────────
        content_div = soup.find("div", class_="contentdiv")
        if not content_div:
            # fallback — grab everything after the ministry header
            content_div = soup.find("body")

        if not content_div:
            return "", url

        # Remove script/style noise
        for tag in content_div(["script", "style"]):
            tag.decompose()

        body_text = content_div.get_text(separator=" ", strip=True)

        # Combine header + body so date/ministry context is preserved
        full_text = f"{header_text}  {body_text}".strip()

        # Clean excessive whitespace
        full_text = re.sub(r"\s{3,}", "  ", full_text)

        return full_text, url

    except Exception as e:
        print(f"      Content fetch error (relid={relid}): {e}")
        return "", url


# ── Extract relids from #lreleaseID ──────────────────────────────

async def get_relids(page):
    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")

    release_div = soup.find("div", id="lreleaseID")
    if not release_div:
        return []

    results = []
    for btn in release_div.find_all("button", class_="btn-release"):
        relid = btn.get("id", "").strip()
        headline = btn.get_text(strip=True)
        if relid.isdigit() and len(headline) > 10:
            results.append((relid, headline))
    return results


# ── Date parsing ──────────────────────────────────────────────────

def parse_date(text, year, month_str):
    mm = MONTH_PAD.get(month_str, "01")
    month_names = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12"
    }
    m = re.search(
        r"(\d{1,2})[- ](january|february|march|april|may|june|july|august|"
        r"september|october|november|december)[,\- ]+(\d{4})",
        text.lower()
    )
    if m and m.group(3) == year:
        return f"{year}-{month_names[m.group(2)]}-{m.group(1).zfill(2)}"
    return f"{year}-{mm}-01"


# ── Core scrape loop ──────────────────────────────────────────────

async def scrape_one(page, ministry_val, ministry_name, year, month, conn):
    try:
        await page.goto(PIB_URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1500)

        await page.select_option("#ryearID",  value=year)
        await page.wait_for_timeout(800)
        await page.select_option("#rmonthID", value=month)
        await page.wait_for_timeout(800)
        await page.select_option("#rdateID",  value="0")
        await page.wait_for_timeout(800)
        await page.select_option("#minID",    value=ministry_val)

        try:
            await page.wait_for_function(
                """() => {
                    const div = document.getElementById('lreleaseID');
                    return div && div.querySelectorAll('button.btn-release').length > 0;
                }""",
                timeout=8000
            )
        except Exception:
            return 0

        relids = await get_relids(page)
        if not relids:
            return 0

        print(
            f"    {len(relids)} releases — {ministry_name} {year}-{month.zfill(2)}")

        saved = 0
        for relid, headline in relids:
            text, content_url = fetch_content(relid)

            if not text or len(text.split()) < 50:
                continue

            is_rel, climate_hits, commodity_hits = pib_filter(text)
            if not is_rel:
                continue

            publish_date = parse_date(text, year, month)

            inserted = insert_article(conn, {
                "url":                   content_url,
                "source_type":           "pib",
                "outlet":                "pib",
                "publish_date":          publish_date,
                "headline":              headline,
                "full_text":             text,
                "climate_terms_found":   "|".join(climate_hits),
                "commodity_terms_found": "|".join(commodity_hits),
                "extraction_method":     "playwright+requests",
            })

            if inserted:
                saved += 1
                print(f"      SAVED #{saved}: {headline[:65]}")

        return saved

    except Exception as e:
        print(f"    Error ({ministry_name} {year}-{month}): {e}")
        return 0


# ── Main ──────────────────────────────────────────────────────────

async def run():
    print("Initialising database...")
    init_db()
    conn = get_conn()
    grand_total = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        for year in YEARS:
            year_total = 0
            print(f"\n{'='*55}")
            print(f"  YEAR {year}")
            print(f"{'='*55}")

            for month in MONTHS:
                for ministry_val, ministry_name in MINISTRIES.items():
                    count = await scrape_one(
                        page, ministry_val, ministry_name, year, month, conn
                    )
                    year_total += count
                    grand_total += count
                    await asyncio.sleep(1)

            print(f"\n  Year {year} complete: {year_total} articles")
            print(f"  Running total: {grand_total} articles")

        await browser.close()

    conn.close()
    print(f"\n{'='*55}")
    print(f"  PIB scraping complete — {grand_total} total articles")
    print(f"{'='*55}")


if __name__ == "__main__":
    asyncio.run(run())
