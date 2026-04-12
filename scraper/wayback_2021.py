"""Run: cd scraper && python wayback_2021.py"""
import wayback_scraper as w
w.YEARS = range(2021, 2022)
w.run()
