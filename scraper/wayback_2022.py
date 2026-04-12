"""Run: cd scraper && python wayback_2022.py"""
import wayback_scraper as w
w.YEARS = range(2022, 2023)
w.run()
