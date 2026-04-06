# Git workflow cheatsheet
# Reference for day-to-day use on this project

# ── Daily workflow ──────────────────────────────────────────────────

# Before starting any work — check where you are
git status
git branch

# Save your work (do this often, at least once per session)
git add .
git commit -m "scraping: add 200 PIB articles from 2021"

# Push to GitHub (backs up your work remotely)
git push origin scraping


# ── Commit message format ───────────────────────────────────────────
# Use a short prefix so the history is readable:
#
#   init:      first setup
#   scraping:  changes to scraper code or data collection
#   nlp:       NLP feature engineering
#   analysis:  time-series, Granger, VAR work
#   fix:       bug fix
#   docs:      README or comment changes
#
# Examples:
#   git commit -m "scraping: fix trafilatura extraction for Mint articles"
#   git commit -m "nlp: add spaCy NER pass for state tagging"
#   git commit -m "fix: handle null publish_date in insert_article"


# ── Moving between stages ───────────────────────────────────────────

# When scraping is done and you start NLP work:
git checkout main
git merge scraping          # bring scraping changes into main
git push origin main

git checkout nlp            # switch to NLP branch
# ... do NLP work ...
git add .
git commit -m "nlp: add climate risk index per commodity"


# ── Useful inspection commands ──────────────────────────────────────

# See full history
git log --oneline

# See what changed in a specific file
git diff scraper/keywords.py

# See changes since last commit
git diff HEAD

# Undo uncommitted changes to a file (careful — cannot be undone)
git checkout -- scraper/keywords.py

# ── If something goes wrong ─────────────────────────────────────────

# See the last 5 commits
git log --oneline -5

# Go back to a previous commit temporarily (safe — read only)
git checkout <commit-hash>

# Come back to the latest
git checkout scraping
