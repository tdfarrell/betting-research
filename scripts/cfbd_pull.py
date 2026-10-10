#!/usr/bin/env python3
"""Pull CollegeFootballData.com stats into data/cfbd/ (needs CFBD_KEY env / repo secret)."""
import datetime as dt, json, os, sys, urllib.request

KEY = os.environ.get("CFBD_KEY")
if not KEY:
    sys.exit("CFBD_KEY is not set (add it as a repo secret named CFBD_KEY).")
now = dt.date.today()
YEAR = now.year if now.month >= 2 else now.year - 1
BASE = "https://api.collegefootballdata.com"
ENDPOINTS = {
    "sp_ratings": f"/ratings/sp?year={YEAR}",
    "advanced_season": f"/stats/season/advanced?year={YEAR}&excludeGarbageTime=true",
    "ppa_teams": f"/ppa/teams?year={YEAR}&excludeGarbageTime=true",
    "season_stats": f"/stats/season?year={YEAR}",
    "games": f"/games?year={YEAR}&seasonType=regular",
    "talent": f"/talent?year={YEAR}",
}
os.makedirs("data/cfbd", exist_ok=True)
failed = 0
for name, path in ENDPOINTS.items():
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {KEY}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.load(r)
        with open(f"data/cfbd/{name}.json", "w") as f:
            json.dump({"pulled_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
                       "year": YEAR, "rows": data}, f)
        print(f"ok   {name}: {len(data)} rows")
    except Exception as e:
        failed += 1
        print(f"FAIL {name}: {e}")
sys.exit(1 if failed == len(ENDPOINTS) else 0)
