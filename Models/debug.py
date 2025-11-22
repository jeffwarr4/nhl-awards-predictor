import requests
import pandas as pd
from io import StringIO

season = 2026  # same as SEASON_END_YEAR in your predictor
url = f"https://www.hockey-reference.com/leagues/NHL_{season}_skaters.html"

print("Fetching:", url)
html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
tables = pd.read_html(StringIO(html))

print("Number of tables:", len(tables))
for i, t in enumerate(tables):
    print(f"\n--- TABLE {i} ---")
    print(t.head(20))
