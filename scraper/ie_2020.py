"""Run: cd scraper && python ie_2020.py"""
import ie_scraper
from datetime import date
ie_scraper.START_DATE = date(2020, 1, 1)
ie_scraper.END_DATE   = date(2020, 12, 31)
ie_scraper.run()
