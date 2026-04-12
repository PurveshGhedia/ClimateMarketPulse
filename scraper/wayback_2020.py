"""Run: cd scraper && python wayback_2020.py"""
import wayback_scraper as w

# hindu_bl 2020 already scraped (49 articles in db) — skip it
del w.OUTLETS["hindu_bl"]
del w.OUTLETS["financial_express"]

w.YEARS = range(2020, 2021)
w.run()
