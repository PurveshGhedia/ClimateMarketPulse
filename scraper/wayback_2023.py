"""Run: cd scraper && python wayback_2023.py"""
import wayback_scraper as w
w.YEARS = range(2023, 2024)
w.run()
