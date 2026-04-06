# Analyzing the Impact of Climate Change on Commodity Prices


---

## Project overview

This project studies how climate events (floods, droughts, heat waves, monsoon
variability) influence food commodity prices across Indian states, using NLP
techniques applied to Indian news articles and government press releases.

**Time period:** 2020–2024  
**Geography:** India (state-level)  
**Commodities:** Vegetables, fruits, cereals, pulses, edible oils

---

## Folder structure

```
nlp_project/
│
├── scraper/                   # Data collection scripts
│   ├── db.py                  #   Database schema & insert helper
│   ├── keywords.py            #   Climate & commodity keyword lists
│   ├── pib_scraper.py         #   PIB press release scraper (Step 1)
│   ├── wayback_scraper.py     #   Wayback Machine scraper (Step 2)
│   └── export_csv.py          #   Export DB → CSV + print stats
│
├── data/
│   ├── raw/                   # Gitignored — raw scraped files live here
│   └── processed/             # Gitignored — cleaned NLP-ready files
│
├── notebooks/                 # Jupyter notebooks for exploration
├── analysis/                  # Final analysis scripts (VAR, Granger etc.)
├── tests/                     # Unit tests for scraper & NLP functions
├── logs/                      # Scraper run logs (gitignored)
│
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd nlp_project

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# (Only needed for Playwright scraper later)
playwright install chromium
```

---

## Data collection — run order

```bash
cd scraper

# Step 1: PIB press releases (~500 articles, ~30 min)
python pib_scraper.py

# Step 2: Wayback Machine archives (~1,250 articles, run overnight)
python wayback_scraper.py

# Check progress anytime
python export_csv.py
```

Data is saved to `data/raw/articles.db` (SQLite).  
Export CSVs appear in `data/raw/` — open in Excel or pandas.

---

## Branch strategy

| Branch      | Purpose                              |
|-------------|--------------------------------------|
| `main`      | Stable, working code only            |
| `scraping`  | Data collection work                 |
| `nlp`       | NLP feature engineering              |
| `analysis`  | Time-series & causal analysis        |

---

## Important: data is not in git

Raw data files (`.db`, `.csv`) are gitignored.  
To share data with a collaborator, use Google Drive or a shared folder.  
Never commit API keys, credentials, or full article text to git.
