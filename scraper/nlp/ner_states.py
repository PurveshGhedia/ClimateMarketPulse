"""
scraper/nlp/ner_states.py

spaCy NER pipeline to populate the `states_mentioned` column in articles.db.
Extracts GPE/LOC entities from headline + full_text, matches against the
canonical list of all 28 Indian states and 8 Union Territories.

Usage (from project root, nlp conda env):
    python scraper/nlp/ner_states.py

Requirements:
    pip install spacy
    python -m spacy download en_core_web_sm
    # OR (preferred on M-series if available):
    pip install spacy[transformers]
    python -m spacy download en_core_web_trf
"""

import os
import sys
import sqlite3
from collections import Counter

import spacy

# ── Paths ────────────────────────────────────────────────────────────────────
# Script lives at scraper/nlp/ner_states.py → project root is two levels up
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "articles.db")

# ── Processing constants ─────────────────────────────────────────────────────
BATCH_SIZE = 500          # rows per DB commit
SPACY_PIPE_BATCH = 64     # texts passed to nlp.pipe() at once
# Truncate text fed to spaCy to avoid timeout on very long articles.
# State names appear early; 6000 chars ≈ first ~900 words.
MAX_CHARS = 6000

# ── Canonical states and UTs ─────────────────────────────────────────────────
CANONICAL_STATES = [
    # 28 States
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal",
    # 8 Union Territories
    "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]

# ── Alias table ───────────────────────────────────────────────────────────────
# Maps lowercase surface form → canonical name.
# Short abbreviations (AP, TN, WB) are intentionally excluded — too ambiguous
# in news text (AP = Associated Press; TN/WB used in other contexts).
ALIASES = {
    # Uttar Pradesh
    "up":                                    "Uttar Pradesh",
    "u.p.":                                  "Uttar Pradesh",
    # Madhya Pradesh
    "mp":                                    "Madhya Pradesh",
    "m.p.":                                  "Madhya Pradesh",
    # Jammu & Kashmir
    "j&k":                                   "Jammu and Kashmir",
    "j & k":                                 "Jammu and Kashmir",
    "jammu & kashmir":                       "Jammu and Kashmir",
    "jammu kashmir":                         "Jammu and Kashmir",
    "kashmir":                               "Jammu and Kashmir",
    # Odisha
    "orissa":                                "Odisha",
    # Puducherry
    "pondicherry":                           "Puducherry",
    # Uttarakhand
    "uttaranchal":                           "Uttarakhand",
    # Himachal Pradesh
    "himachal":                              "Himachal Pradesh",
    "hp":                                    "Himachal Pradesh",
    # Chhattisgarh (common misspellings)
    "chattisgarh":                           "Chhattisgarh",
    "chhatisgarh":                           "Chhattisgarh",
    # Arunachal Pradesh
    "arunachal":                             "Arunachal Pradesh",
    # Telangana
    "ts":                                    "Telangana",
    # Delhi
    "new delhi":                             "Delhi",
    "nct":                                   "Delhi",
    "ncr":                                   "Delhi",
    # Andaman and Nicobar
    "andaman":                               "Andaman and Nicobar Islands",
    "andaman and nicobar":                   "Andaman and Nicobar Islands",
    "andaman & nicobar":                     "Andaman and Nicobar Islands",
    "nicobar":                               "Andaman and Nicobar Islands",
    # Dadra / Daman / Diu
    "daman":                                 "Dadra and Nagar Haveli and Daman and Diu",
    "diu":                                   "Dadra and Nagar Haveli and Daman and Diu",
    "dadra":                                 "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli":                "Dadra and Nagar Haveli and Daman and Diu",
    # City → State mappings (prominent cities used as proxies in journalism)
    "mumbai":                                "Maharashtra",
    "pune":                                  "Maharashtra",
    "nagpur":                                "Maharashtra",
    "chennai":                               "Tamil Nadu",
    "bengaluru":                             "Karnataka",
    "bangalore":                             "Karnataka",
    "hyderabad":                             "Telangana",
    "kolkata":                               "West Bengal",
    "calcutta":                              "West Bengal",
    "ahmedabad":                             "Gujarat",
    "surat":                                 "Gujarat",
    "lucknow":                               "Uttar Pradesh",
    "kanpur":                                "Uttar Pradesh",
    "varanasi":                              "Uttar Pradesh",
    "agra":                                  "Uttar Pradesh",
    "jaipur":                                "Rajasthan",
    "jodhpur":                               "Rajasthan",
    "patna":                                 "Bihar",
    "bhopal":                                "Madhya Pradesh",
    "indore":                                "Madhya Pradesh",
    "raipur":                                "Chhattisgarh",
    "ranchi":                                "Jharkhand",
    "bhubaneswar":                           "Odisha",
    "thiruvananthapuram":                    "Kerala",
    "trivandrum":                            "Kerala",
    "kochi":                                 "Kerala",
    "amritsar":                              "Punjab",
    "ludhiana":                              "Punjab",
    "dehradun":                              "Uttarakhand",
    "shimla":                                "Himachal Pradesh",
    "guwahati":                              "Assam",
    "imphal":                                "Manipur",
    "shillong":                              "Meghalaya",
    "aizawl":                                "Mizoram",
    "kohima":                                "Nagaland",
    "agartala":                              "Tripura",
    "gangtok":                               "Sikkim",
    "itanagar":                              "Arunachal Pradesh",
    "dispur":                                "Assam",
    "panaji":                                "Goa",
    "srinagar":                              "Jammu and Kashmir",
    "jammu":                                 "Jammu and Kashmir",
    "leh":                                   "Ladakh",
    "kargil":                                "Ladakh",
}

# Build the final lookup: canonical names + aliases, all lowercased
STATE_LOOKUP: dict[str, str] = {}
for name in CANONICAL_STATES:
    STATE_LOOKUP[name.lower()] = name
for alias, canonical in ALIASES.items():
    STATE_LOOKUP[alias.lower()] = canonical


# ── spaCy model loader ───────────────────────────────────────────────────────
def build_entity_ruler(nlp: spacy.language.Language) -> spacy.language.Language:
    """
    Inject a rule-based EntityRuler BEFORE the NER component.
    This guarantees every canonical state name and alias is tagged as GPE
    regardless of model confidence — fixing the ~526 literal-match misses
    where spaCy's statistical NER failed to tag obvious Indian place names.

    The ruler uses overwrite_ents=False so it only fills gaps; it does not
    override entities the NER model already found correctly.
    """
    ruler = nlp.add_pipe("entity_ruler", before="ner",
                         config={"overwrite_ents": False})

    patterns = []

    # All canonical state/UT names
    for name in CANONICAL_STATES:
        patterns.append({"label": "GPE", "pattern": name})
        # Also add lowercase variant for safety
        patterns.append({"label": "GPE", "pattern": name.lower()})

    # All aliases → we tag the alias text as GPE; extract_states() then
    # maps it to the canonical name via STATE_LOOKUP
    for alias in ALIASES:
        patterns.append({"label": "GPE", "pattern": alias})
        patterns.append({"label": "GPE", "pattern": alias.capitalize()})

    ruler.add_patterns(patterns)
    return nlp


def load_spacy_model() -> spacy.language.Language:
    """
    Try en_core_web_trf first (more accurate NER), fall back to en_core_web_sm.
    Disables pipeline components we don't need to save memory and time.
    Injects an EntityRuler to catch literal state names the statistical
    NER model misses.
    """
    # trf does not have 'parser' or 'lemmatizer' by default, so we only
    # disable what exists in each model.
    candidates = [
        ("en_core_web_trf", ["tagger", "parser",
         "lemmatizer", "attribute_ruler"]),
        ("en_core_web_sm",  ["tagger", "parser",
         "lemmatizer", "attribute_ruler"]),
    ]
    for model_name, disable_components in candidates:
        try:
            nlp = spacy.load(model_name)
            existing = set(nlp.pipe_names)
            to_disable = [c for c in disable_components if c in existing]
            if to_disable:
                nlp = spacy.load(model_name, disable=to_disable)

            # Inject EntityRuler for deterministic Indian place name coverage
            nlp = build_entity_ruler(nlp)

            print(f"Loaded spaCy model  : {model_name}")
            print(f"Active components   : {nlp.pipe_names}")
            return nlp
        except OSError:
            continue

    print(
        "ERROR: No spaCy model found.\n"
        "  Run: python -m spacy download en_core_web_sm\n"
        "  Or:  pip install spacy[transformers] && python -m spacy download en_core_web_trf"
    )
    sys.exit(1)


# ── Entity → state matcher ───────────────────────────────────────────────────
def extract_states(doc) -> list[str]:
    """
    Walk spaCy GPE/LOC entities and map to canonical state names.
    Returns a sorted, deduplicated list.
    """
    found: set[str] = set()
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            key = ent.text.strip().lower()
            if key in STATE_LOOKUP:
                found.add(STATE_LOOKUP[key])
    return sorted(found)


# ── Main pipeline ────────────────────────────────────────────────────────────
def run_ner(db_path: str) -> None:
    nlp = load_spacy_model()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT article_id, headline, full_text
        FROM   articles
        WHERE  is_duplicate = 0
          AND  (states_mentioned IS NULL OR states_mentioned = '')
    """).fetchall()

    total = len(rows)
    if total == 0:
        print("No unprocessed articles found. states_mentioned is already populated.")
        conn.close()
        return

    print(f"Articles to process : {total}")
    print(f"DB commit every     : {BATCH_SIZE} articles")
    print("-" * 50)

    processed = 0
    with_state = 0
    state_counter: Counter = Counter()
    batch_updates: list[tuple[str, str]] = []

    # Build (article_id, text) pairs for nlp.pipe()
    # We iterate in chunks of SPACY_PIPE_BATCH to keep memory bounded
    def iter_chunks(seq, size):
        for i in range(0, len(seq), size):
            yield seq[i: i + size]

    for chunk in iter_chunks(rows, SPACY_PIPE_BATCH):
        ids = [r["article_id"] for r in chunk]
        texts = [
            f"{r['headline'] or ''} {r['full_text'] or ''}"[:MAX_CHARS].strip()
            for r in chunk
        ]

        for article_id, doc in zip(ids, nlp.pipe(texts, batch_size=SPACY_PIPE_BATCH)):
            states = extract_states(doc)
            # empty string "" if no states found
            states_val = "|".join(states)

            batch_updates.append((states_val, article_id))

            if states:
                with_state += 1
                for s in states:
                    state_counter[s] += 1

            processed += 1

            if len(batch_updates) >= BATCH_SIZE:
                conn.executemany(
                    "UPDATE articles SET states_mentioned = ? WHERE article_id = ?",
                    batch_updates,
                )
                conn.commit()
                batch_updates.clear()

            if processed % 1000 == 0:
                pct = processed / total * 100
                print(f"  Progress: {processed:>5}/{total}  ({pct:.1f}%)")

    # Flush remaining updates
    if batch_updates:
        conn.executemany(
            "UPDATE articles SET states_mentioned = ? WHERE article_id = ?",
            batch_updates,
        )
        conn.commit()

    conn.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    pct_with = (with_state / processed * 100) if processed else 0
    print(f"\n{'=' * 50}")
    print(f"NER complete.")
    print(f"Total processed      : {processed}")
    print(f"With ≥1 state        : {with_state}  ({pct_with:.1f}%)")
    print(f"No state found       : {processed - with_state}")
    print(f"\nTop 10 most mentioned states:")
    print(f"  {'State':<45} {'Count':>6}")
    print(f"  {'-'*45}  {'-'*6}")
    for state, count in state_counter.most_common(10):
        print(f"  {state:<45} {count:>6}")


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"ERROR: DB not found at {DB_PATH}")
        sys.exit(1)
    run_ner(DB_PATH)
