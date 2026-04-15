"""
collect_cfpi_mospi.py
ClimateMarketPulse -- MoSPI CFPI Item-Level Data Collection
===========================================================
Uses the public GET API that powers the esankhyiki.mospi.gov.in portal.
No authentication required.

Endpoint (from browser network tab):
  GET https://api.mospi.gov.in/api/cpi/getCPIData
  ?base_year=2012&level=Item&series=Current
  &year=YYYY&month_code=M&state_code=99
  &item_code=CODE&isView=table&page=1&limit=500

127 food items x 5 years (2020-2024) x 12 months = 7,620 requests.
At 0.5s delay: ~63 min. Fully resumable -- completed calls are skipped.

Usage:
  conda activate nlp
  pip install requests pandas
  python collect_cfpi_mospi.py

Outputs (in data/raw/):
  price_data.db          SQLite with table cfpi_item
  cfpi_item_long.csv     Long format: ready for NLP alignment
  cfpi_item_wide.csv     Pivot: rows=year-month, cols=item names (index values)
"""

from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
import urllib3
import time
import sqlite3
import logging
import requests
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL = "https://api.mospi.gov.in/api/cpi/getCPIData"
BASE_YEAR = "2012"
LEVEL = "Item"
SERIES = "Current"
# STATE_CODE = "99"           # 99 = All India
# Replace STATE_CODE = "99" with this:
STATES = {
    "1": "Jammu & Kashmir", "2": "Himachal Pradesh", "3": "Punjab", "4": "Chandigarh", "5": "Uttarakhand",
    "6": "Haryana", "7": "Delhi", "8": "Rajasthan", "9": "Uttar Pradesh", "10": "Bihar", "11": "Sikkim",
    "12": "Arunachal Pradesh", "13": "Nagaland", "14": "Manipur", "15": "Mizoram", "16": "Tripura",
    "17": "Meghalaya", "18": "Assam", "19": "West Bengal", "20": "Jharkhand", "21": "Odisha",
    "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat", "25": "Daman & Diu",
    "26": "Dadra & Nagar Haveli", "27": "Maharashtra", "28": "Andhra Pradesh", "29": "Karnataka",
    "30": "Goa", "31": "Lakshadweep", "32": "Kerala", "33": "Tamil Nadu", "34": "Puducherry",
    "35": "Andaman & Nicobar Islands", "36": "Telangana", "99": "All India"
}

YEARS = list(range(2020, 2027))   # change to range(2020, 2027) for 2025-26
MONTHS = list(range(1, 13))
REQUEST_DELAY = 0.5                        # seconds between requests

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DB_PATH = DATA_DIR / "price_data.db"
LOG_PATH = DATA_DIR / "collect_cfpi.log"

# ── Food items from CPI_Metadata.xlsx (base_year=2012, groups 1.1.xx/1.2.xx)
# All 127 items from Food & Beverages group. Non-food groups excluded.

FOOD_ITEMS = {
    # 1.1.01  Cereals and Products
    "1.1.01.1.1.01.P": "Rice - PDS",
    "1.1.01.1.1.02.X": "Rice - Other Sources",
    "1.1.01.1.1.03.0": "Chira",
    "1.1.01.1.1.05.0": "Muri",
    "1.1.01.1.1.06.0": "Other Rice Products",
    "1.1.01.1.1.07.P": "Wheat/Atta - PDS",
    "1.1.01.1.1.08.X": "Wheat/Atta - Other Sources",
    "1.1.01.1.1.09.0": "Maida",
    "1.1.01.1.1.10.0": "Suji, Rawa",
    "1.1.01.1.1.11.X": "Sewai, Noodles",
    "1.1.01.1.1.12.0": "Bread (bakery)",
    "1.1.01.1.1.13.X": "Biscuits, Chocolates, etc.",
    "1.1.01.1.1.15.0": "Other Cereals",
    "1.1.01.1.1.16.0": "Cereal Substitutes: Tapioca, etc.",
    "1.1.01.2.1.01.X": "Jowar and its Products",
    "1.1.01.2.1.02.X": "Bajra and its Products",
    "1.1.01.2.1.03.X": "Maize and Products",
    "1.1.01.2.1.05.X": "Small Millets and their Products",
    "1.1.01.2.1.06.X": "Ragi and its Products",
    "1.1.01.3.2.01.0": "Grinding Charges",
    # 1.1.02  Meat and Fish
    "1.1.02.1.1.01.0": "Goat Meat/Mutton",
    "1.1.02.1.1.02.X": "Beef/Buffalo Meat",
    "1.1.02.1.1.03.0": "Pork",
    "1.1.02.1.1.04.0": "Chicken",
    "1.1.02.1.1.05.0": "Other Meat (Birds, Crab, etc.)",
    "1.1.02.2.1.01.X": "Fish, Prawn",
    # 1.1.03  Egg
    "1.1.03.1.1.01.0": "Eggs",
    # 1.1.04  Milk and Products
    "1.1.04.1.1.01.X": "Milk: Liquid",
    "1.1.04.2.1.01.0": "Baby Food",
    "1.1.04.2.1.02.X": "Milk: Condensed/Powder",
    "1.1.04.2.1.03.0": "Curd",
    "1.1.04.2.1.04.0": "Other Milk Products",
    # 1.1.05  Oils and Fats
    "1.1.05.1.1.01.0": "Mustard Oil",
    "1.1.05.1.1.02.0": "Groundnut Oil",
    "1.1.05.1.1.03.0": "Coconut Oil",
    "1.1.05.1.1.04.0": "Refined Oil (Sunflower/Soyabean)",
    "1.1.05.2.1.01.0": "Ghee",
    "1.1.05.2.1.02.0": "Butter",
    "1.1.05.2.1.03.0": "Vanaspati, Margarine",
    # 1.1.06  Fruits
    "1.1.06.1.1.01.0": "Banana",
    "1.1.06.1.1.02.0": "Jackfruit",
    "1.1.06.1.1.03.0": "Watermelon",
    "1.1.06.1.1.04.0": "Pineapple",
    "1.1.06.1.1.05.0": "Coconut",
    "1.1.06.1.1.06.0": "Green Coconut",
    "1.1.06.1.1.07.0": "Guava",
    "1.1.06.1.1.08.0": "Singara",
    "1.1.06.1.1.09.X": "Orange, Mausami",
    "1.1.06.1.1.10.0": "Papaya",
    "1.1.06.1.1.11.0": "Mango",
    "1.1.06.1.1.12.0": "Kharbooza",
    "1.1.06.1.1.13.X": "Pears/Nashpati",
    "1.1.06.1.1.14.0": "Berries",
    "1.1.06.1.1.15.0": "Leechi",
    "1.1.06.1.1.16.0": "Apple",
    "1.1.06.1.1.17.0": "Grapes",
    "1.1.06.1.1.18.0": "Other Fresh Fruits",
    "1.1.06.2.1.01.0": "Coconut: Copra",
    "1.1.06.2.1.02.0": "Groundnut",
    "1.1.06.2.1.03.0": "Dates",
    "1.1.06.2.1.04.0": "Cashewnut",
    "1.1.06.2.1.05.0": "Walnut",
    "1.1.06.2.1.06.0": "Other Nuts",
    "1.1.06.2.1.07.X": "Raisin, Kishmish, Monacca",
    "1.1.06.2.1.08.0": "Other Dry Fruits",
    # 1.1.07  Vegetables
    "1.1.07.1.1.01.0": "Potato",
    "1.1.07.1.1.02.0": "Onion",
    "1.1.07.1.1.03.0": "Radish",
    "1.1.07.1.1.04.0": "Carrot",
    "1.1.07.1.1.05.0": "Garlic",
    "1.1.07.1.1.06.0": "Ginger",
    "1.1.07.2.1.01.X": "Palak/Leafy Vegetables",
    "1.1.07.3.1.01.0": "Tomato",
    "1.1.07.3.1.02.0": "Brinjal",
    "1.1.07.3.1.03.0": "Cauliflower",
    "1.1.07.3.1.04.0": "Cabbage",
    "1.1.07.3.1.05.0": "Green Chillies",
    "1.1.07.3.1.06.0": "Lady's Finger",
    "1.1.07.3.1.07.X": "Parwal/Patal, Kundru",
    "1.1.07.3.1.08.X": "Gourd, Pumpkin",
    "1.1.07.3.1.09.0": "Peas (vegetables)",
    "1.1.07.3.1.10.X": "Beans, Barbati",
    "1.1.07.3.1.11.0": "Lemon",
    "1.1.07.3.1.12.X": "Other Vegetables",
    "1.1.07.4.1.01.0": "Pickles",
    "1.1.07.4.1.02.0": "Chips",
    # 1.1.08  Pulses and Products
    "1.1.08.1.1.01.0": "Arhar/Tur Dal",
    "1.1.08.1.1.02.0": "Gram: Split (Chana Dal)",
    "1.1.08.1.1.03.0": "Gram: Whole",
    "1.1.08.1.1.04.0": "Moong Dal",
    "1.1.08.1.1.05.0": "Masur Dal",
    "1.1.08.1.1.06.0": "Urd Dal",
    "1.1.08.1.1.07.0": "Peas (pulses)",
    "1.1.08.1.1.08.0": "Khesari",
    "1.1.08.1.1.09.X": "Other Pulses",
    "1.1.08.2.1.01.0": "Gram Products (Sattu)",
    "1.1.08.2.1.02.0": "Besan",
    "1.1.08.2.1.03.0": "Other Pulse Products",
    # 1.1.09  Sugar and Confectionery
    "1.1.09.1.1.01.P": "Sugar - PDS",
    "1.1.09.1.1.02.0": "Sugar - Other Sources",
    "1.1.09.1.1.03.0": "Gur (Jaggery)",
    "1.1.09.2.1.01.0": "Candy, Misri",
    "1.1.09.2.1.02.0": "Honey",
    "1.1.09.2.1.03.X": "Sauce, Jam, Jelly",
    "1.1.09.3.1.01.0": "Ice-cream",
    # 1.1.10  Spices
    "1.1.10.1.1.01.0": "Salt",
    "1.1.10.1.1.02.0": "Jeera",
    "1.1.10.1.1.03.0": "Dhania (Coriander)",
    "1.1.10.1.1.04.0": "Turmeric",
    "1.1.10.1.1.05.0": "Black Pepper",
    "1.1.10.1.1.06.0": "Dry Chillies",
    "1.1.10.1.1.07.0": "Tamarind",
    "1.1.10.1.1.08.0": "Curry Powder",
    "1.1.10.1.1.09.0": "Oilseeds",
    # 1.1.12  Prepared Meals and Snacks
    "1.1.12.1.1.01.0": "Tea: Cups",
    "1.1.12.1.1.02.0": "Coffee: Cups",
    "1.1.12.2.1.01.0": "Cooked Meals Purchased",
    "1.1.12.3.1.01.X": "Cooked Snacks Purchased",
    "1.1.12.3.1.03.X": "Prepared Sweets, Cake, Pastry",
    "1.1.12.3.1.04.0": "Papad, Bhujia, Namkeen",
    "1.1.12.3.1.05.0": "Other Packaged Processed Food",
    # 1.2.11  Non-alcoholic Beverages
    "1.2.11.1.1.01.0": "Tea: Leaf",
    "1.2.11.1.1.02.0": "Coffee: Powder",
    "1.2.11.2.1.01.0": "Mineral Water",
    "1.2.11.2.1.02.X": "Cold Beverages: Bottled/canned",
    "1.2.11.2.1.03.X": "Fruit Juice and Shake",
    "1.2.11.2.1.04.X": "Other Beverages: Cocoa, Chocolate",
}

SUBGROUP_MAP = {
    "1.1.01": "Cereals and Products",
    "1.1.02": "Meat and Fish",
    "1.1.03": "Egg",
    "1.1.04": "Milk and Products",
    "1.1.05": "Oils and Fats",
    "1.1.06": "Fruits",
    "1.1.07": "Vegetables",
    "1.1.08": "Pulses and Products",
    "1.1.09": "Sugar and Confectionery",
    "1.1.10": "Spices",
    "1.1.12": "Prepared Meals/Snacks",
    "1.2.11": "Non-alcoholic Beverages",
}


def subgroup(code):
    return SUBGROUP_MAP.get(".".join(code.split(".")[:3]), "Other")


# ── DB ────────────────────────────────────────────────────────────────────────

def init_db(db_path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE IF NOT EXISTS cfpi_item (
        item_code TEXT NOT NULL, item_name TEXT, subgroup TEXT,
        state_code TEXT NOT NULL,
        year INTEGER NOT NULL, month INTEGER NOT NULL,
        index_value REAL, inflation_yoy REAL,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (item_code, state_code, year, month))""")

    # We must recreate fetch_log to include state_code in the primary key
    conn.execute("DROP TABLE IF EXISTS fetch_log")
    conn.execute("""CREATE TABLE IF NOT EXISTS fetch_log (
        item_code TEXT NOT NULL, state_code TEXT NOT NULL, year INTEGER NOT NULL, month INTEGER NOT NULL,
        PRIMARY KEY (item_code, state_code, year, month))""")
    conn.commit()
    return conn


def done(conn, code, state_code, y, m):
    return conn.execute(
        "SELECT 1 FROM fetch_log WHERE item_code=? AND state_code=? AND year=? AND month=?",
        (code, state_code, y, m)).fetchone() is not None


def mark(conn, code, state_code, y, m):
    conn.execute("INSERT OR IGNORE INTO fetch_log VALUES (?,?,?,?)",
                 (code, state_code, y, m))
    conn.commit()

# ── HTTP ──────────────────────────────────────────────────────────────────────

# Create a custom adapter to allow legacy SSL renegotiation


class LegacySSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        # 0x4 is the hex value for ssl.OP_LEGACY_SERVER_CONNECT
        context.options |= 0x4
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)


HEADERS = {
    "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br", "Connection": "keep-alive",
    "Origin": "https://esankhyiki.mospi.gov.in",
    "Referer": "https://esankhyiki.mospi.gov.in/",
    "Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                   "Version/17.0 Safari/605.1.15"),
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
# Mount the adapter to HTTPS requests
SESSION.mount("https://", LegacySSLAdapter())


def fetch(code, state_code, year, month, retries=3):
    # params = {"base_year": BASE_YEAR, "level": LEVEL, "series": SERIES,
    #           "year": str(year), "month_code": str(month),
    #           "state_code": STATE_CODE, "item_code": code,
    #           "isView": "table", "page": "1", "limit": "500"}
    params = {"base_year": BASE_YEAR, "level": LEVEL, "series": SERIES,
              "year": str(year), "month_code": str(month),
              "state_code": state_code, "item_code": code,
              "isView": "table", "page": "1", "limit": "500"}
    for attempt in range(retries):
        try:
            r = SESSION.get(BASE_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for k in ("data", "records", "result", "Data", "items"):
                    if k in data and isinstance(data[k], list):
                        return data[k]
            return []
        except requests.exceptions.HTTPError:
            wait = 30 if r.status_code == 429 else 10 * (attempt + 1)
            logging.warning(
                f"  HTTP {r.status_code} {code} {year}-{month} waiting {wait}s")
            time.sleep(wait)
        except requests.exceptions.Timeout:
            logging.warning(
                f"  Timeout {code} {year}-{month} attempt {attempt+1}")
            time.sleep(5)
        except Exception as e:
            logging.error(f"  Error {code} {year}-{month}: {e}")
            return []
    return []


def upsert(records, code, name, state_code, year, month, conn):
    sg = subgroup(code)
    n = 0
    for rec in records:
        r = {k.lower().strip(): v for k, v in rec.items()}
        sector = str(r.get("sector", r.get("sector_code", "3")))
        if sector not in ("3", "Combined", "combined", ""):
            continue
        idx = r.get("index") or r.get(
            "index_value") or r.get("cpi") or r.get("value")
        infl = r.get("inflation") or r.get(
            "inflation_yoy") or r.get("inflationrate")
        try:
            idx = float(idx) if idx not in (None, "") else None
            infl = float(infl) if infl not in (None, "") else None
        except (ValueError, TypeError):
            idx, infl = None, None
        try:
            # Note: Now inserting the dynamic state_code
            conn.execute(
                "INSERT OR IGNORE INTO cfpi_item "
                "(item_code,item_name,subgroup,state_code,year,month,index_value,inflation_yoy) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (code, name, sg, state_code, year, month, idx, infl))
            n += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return n

# ── Collection loop ───────────────────────────────────────────────────────────


def collect(conn):
    codes = list(FOOD_ITEMS.keys())
    state_codes = list(STATES.keys())
    total = len(codes) * len(state_codes) * len(YEARS) * len(MONTHS)
    done_n = conn.execute("SELECT COUNT(*) FROM fetch_log").fetchone()[0]
    logging.info(
        f"Total calls: {total} | Done: {done_n} | Remaining: {total - done_n}")

    inserted = 0
    call_n = done_n
    for code in codes:
        name = FOOD_ITEMS[code]
        for state_code in state_codes:
            state_name = STATES[state_code]
            for year in YEARS:
                for month in MONTHS:
                    if done(conn, code, state_code, year, month):
                        continue
                    recs = fetch(code, state_code, year, month)
                    inserted += upsert(recs, code, name,
                                       state_code, year, month, conn)
                    mark(conn, code, state_code, year, month)
                    call_n += 1
                    if call_n % 200 == 0:
                        pct = 100 * call_n / total
                        logging.info(f"  [{pct:5.1f}%] {call_n}/{total} | "
                                     f"rows={inserted} | {name[:15]} ({state_name[:10]}) {year}-{month:02d}")
                    time.sleep(REQUEST_DELAY)
    logging.info(f"Collection complete. New rows: {inserted}")
# ── Export & summary ──────────────────────────────────────────────────────────


def export(conn):
    # Removed the WHERE state_code='99' filter
    df = pd.read_sql(
        "SELECT item_code,item_name,subgroup,state_code,year,month,index_value,inflation_yoy "
        "FROM cfpi_item ORDER BY subgroup,item_code,state_code,year,month",
        conn)
    if df.empty:
        logging.warning("No data to export yet.")
        return

    df['state_name'] = df['state_code'].astype(str).map(STATES)
    long_p = DATA_DIR / "cfpi_item_long.csv"
    df.to_csv(long_p, index=False)
    logging.info(f"  Long CSV -> {long_p} ({len(df)} rows)")

    # Wide export now groups by both state and year_month
    df["state_year_month"] = df["state_name"] + "_" + \
        df["year"].astype(str) + "-" + df["month"].apply(lambda m: f"{m:02d}")
    wide = df.pivot_table(
        index="state_year_month", columns="item_name",
        values="index_value", aggfunc="first").reset_index()
    wide_p = DATA_DIR / "cfpi_item_wide.csv"
    wide.to_csv(wide_p, index=False)
    logging.info(
        f"  Wide CSV -> {wide_p} ({wide.shape[0]}r x {wide.shape[1]}c)")


def summary(conn):
    total = conn.execute("SELECT COUNT(*) FROM cfpi_item").fetchone()[0]
    print("\n" + "="*65)
    print("  ClimateMarketPulse - MoSPI CFPI Summary")
    print("="*65)
    print(f"  Total rows (All States + All India): {total}")
    rows = conn.execute("""
        SELECT subgroup, COUNT(DISTINCT item_code) as items,
               MIN(year)||'-'||printf('%02d',MIN(month)),
               MAX(year)||'-'||printf('%02d',MAX(month)),
               COUNT(*) as obs
        FROM cfpi_item 
        GROUP BY subgroup ORDER BY subgroup""").fetchall()
    print(f"\n  {'Subgroup':<35} Items  Period           Obs")
    print(f"  {'-'*60}")
    for r in rows:
        print(f"  {r[0]:<35} {r[1]:>5}  {r[2]}->{r[3]}  {r[4]}")
    done_n = conn.execute("SELECT COUNT(*) FROM fetch_log").fetchone()[0]

    # Updated to multiply by the number of states!
    ttl = len(FOOD_ITEMS) * len(STATES) * len(YEARS) * len(MONTHS)
    print(f"\n  Fetch progress: {done_n}/{ttl} ({100*done_n/ttl:.1f}%)")
    print(f"  DB  : {DB_PATH}")
    print(f"  CSVs: {DATA_DIR}/cfpi_item_long.csv  /  cfpi_item_wide.csv")
    print("="*65 + "\n")

# ── Entry ─────────────────────────────────────────────────────────────────────


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(str(LOG_PATH))],
    )
    logging.info("MoSPI CFPI collection starting")
    logging.info(f"Endpoint : {BASE_URL}")
    logging.info(f"Items: {len(FOOD_ITEMS)}  Years: {YEARS[0]}-{YEARS[-1]}  "
                 f"Delay: {REQUEST_DELAY}s  States: {len(STATES)}")
    conn = init_db(DB_PATH)
    collect(conn)
    export(conn)
    summary(conn)
    conn.close()


if __name__ == "__main__":
    main()
