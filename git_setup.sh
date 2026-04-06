#!/bin/bash
# ================================================================
# git_setup.sh
# Run this ONCE inside your project folder to initialise git.
# Works on Mac, Linux, and Git Bash on Windows.
#
# Usage:
#   chmod +x git_setup.sh
#   ./git_setup.sh
# ================================================================

set -e  # stop on any error

echo ""
echo "=== Step 1: Initialise git repo ==="
git init
echo "Git repo initialised."

echo ""
echo "=== Step 2: Set your identity (edit these!) ==="
# Change to your actual name and university email
git config user.name  "PurveshGhedia"
git config user.email "purveshghedia@gmail.com"

echo ""
echo "=== Step 3: Stage all project files ==="
git add .

echo ""
echo "=== Step 4: First commit ==="
git commit -m "init: ClimateMarketPulse project scaffold"  

echo ""
echo "=== Step 5: Rename default branch to main ==="
git branch -M main

echo ""
echo "=== Step 6: Create working branches ==="
git branch scraping   # you'll work here during data collection
git branch nlp        # for NLP feature engineering later
git branch analysis   # for VAR/Granger analysis later

echo ""
echo "=== Step 7: Switch to scraping branch ==="
git checkout scraping

echo ""
echo "=== Done! ==="
echo ""
echo "Your branches:"
git branch -v
echo ""
echo "Next steps:"
echo "  1. Create a repo on GitHub (github.com → New repository)"
echo "  2. Copy the remote URL (looks like: https://github.com/yourname/ClimateMarketPulse.git)"
echo "  3. Run these two commands:"
echo "       git remote add origin <paste-url-here>"
echo "       git push -u origin main"
echo "       git push origin scraping nlp analysis"
echo ""
echo "  4. Start scraping:"
echo "       cd scraper"
echo "       python pib_scraper.py"
