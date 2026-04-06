import sqlite3
import hashlib
import os
import re
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "articles.db")

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS articles (
    article_id            TEXT PRIMARY KEY,
    url                   TEXT NOT NULL,
    source_type           TEXT NOT NULL,
    scraped_at            TEXT NOT NULL,
    outlet                TEXT NOT NULL,
    publish_date          TEXT,
    publish_year          INTEGER,
    publish_month         INTEGER,
    headline              TEXT,
    author                TEXT,
    full_text             TEXT,
    word_count            INTEGER,
    content_hash          TEXT,
    states_mentioned      TEXT,
    climate_terms_found   TEXT,
    commodity_terms_found TEXT,
    relevance_score       REAL,
    is_duplicate          INTEGER DEFAULT 0,
    archive_url           TEXT,
    crawl_id              TEXT,
    extraction_method     TEXT
);
CREATE INDEX IF NOT EXISTS idx_date   ON articles(publish_date);
CREATE INDEX IF NOT EXISTS idx_outlet ON articles(outlet);
CREATE INDEX IF NOT EXISTS idx_hash   ON articles(content_hash);
"""

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    conn.executescript(CREATE_TABLE)
    conn.commit()
    conn.close()
    print(f"Database ready at {DB_PATH}")

def make_article_id(url, publish_date):
    key = f"{url}|{publish_date or 'unknown'}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]

def make_content_hash(text):
    return hashlib.md5((text or "")[:300].lower().strip().encode()).hexdigest()

def insert_article(conn, record: dict):
    """
    Call this from every scraper. Only url, source_type, outlet are mandatory.
    All other fields can be missing — they default to None.
    Returns True if inserted, False if duplicate.
    """
    defaults = {
        "publish_date": None, "publish_year": None, "publish_month": None,
        "headline": None, "author": None, "full_text": None, "word_count": 0,
        "states_mentioned": None, "climate_terms_found": None,
        "commodity_terms_found": None, "relevance_score": None,
        "is_duplicate": 0, "archive_url": None, "crawl_id": None,
        "extraction_method": None,
    }
    row = {**defaults, **record}

    row["article_id"]   = make_article_id(row["url"], row.get("publish_date"))
    row["content_hash"] = make_content_hash(row.get("full_text") or "")
    row["word_count"]   = len((row.get("full_text") or "").split())
    row["scraped_at"]   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    if row.get("publish_date"):
        try:
            m = re.match(r"(\d{4})-(\d{2})", str(row["publish_date"]))
            if m:
                row["publish_year"]  = int(m.group(1))
                row["publish_month"] = int(m.group(2))
        except Exception:
            pass

    cursor = conn.execute("""
        INSERT OR IGNORE INTO articles (
            article_id, url, source_type, scraped_at,
            outlet, publish_date, publish_year, publish_month,
            headline, author, full_text, word_count, content_hash,
            states_mentioned, climate_terms_found, commodity_terms_found,
            relevance_score, is_duplicate, archive_url, crawl_id, extraction_method
        ) VALUES (
            :article_id, :url, :source_type, :scraped_at,
            :outlet, :publish_date, :publish_year, :publish_month,
            :headline, :author, :full_text, :word_count, :content_hash,
            :states_mentioned, :climate_terms_found, :commodity_terms_found,
            :relevance_score, :is_duplicate, :archive_url, :crawl_id, :extraction_method
        )
    """, row)
    conn.commit()
    return cursor.rowcount == 1
