"""
Delete all Wayback Machine articles from articles.db.
Run this before re-running wayback_scraper.py to start fresh.

Usage:
    cd scraper
    python delete_wayback.py
"""

from db import get_conn


def main():
    conn = get_conn()

    # --- NEW: Display Database Summary ---
    print("\n--- Current Database Summary ---")

    # 1. Total number of articles
    total_articles = conn.execute(
        "SELECT COUNT(*) FROM articles").fetchone()[0]
    print(f"Total articles in database: {total_articles}")

    # 2. Total number of articles per source
    print("Articles per source:")
    source_counts = conn.execute(
        "SELECT source_type, COUNT(*) FROM articles GROUP BY source_type"
    ).fetchall()

    if not source_counts:
        print("  (Database is empty)")
    else:
        for source, s_count in source_counts:
            # Fallback to 'Unknown' just in case there are null source_types
            source_name = source if source else "Unknown"
            print(f"  - {source_name}: {s_count}")

    print("--------------------------------\n")
    # -------------------------------------

    # Count before deletion
    count = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE source_type = 'wayback'"
    ).fetchone()[0]

    if count == 0:
        print("No wayback articles found in database. Nothing to delete.")
        conn.close()
        return

    print(f"Found {count} wayback articles in database ready for deletion.")
    confirm = input("Type 'yes' to delete them: ").strip().lower()

    if confirm != "yes":
        print("Aborted — nothing deleted.")
        conn.close()
        return

    conn.execute("DELETE FROM articles WHERE source_type = 'wayback'")
    conn.commit()

    remaining = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    conn.close()

    print(f"Deleted {count} wayback articles.")
    print(f"{remaining} articles remain in database.")


if __name__ == "__main__":
    main()
