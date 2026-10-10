#!/usr/bin/env python3
"""Snapshot DraftKings lines from ESPN's public scoreboard (CFB FBS + NFL).

Writes:
  data/odds/latest_<league>.json   compact current board (open + current lines)
  data/odds/history_<league>.csv   one row per game per pull, for line-movement / CLV
"""
import csv, datetime as dt, json, os, urllib.request

LEAGUES = {
    "cfb": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=80&limit=300&dates={d}",
    "nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={d}",
}
OUT = "data/odds"
now = dt.datetime.now(dt.timezone.utc)
dates = "-".join(d.strftime("%Y%m%d") for d in (now - dt.timedelta(days=1), now + dt.timedelta(days=7)))

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d

def parse(ev):
    comp = ev["competitions"][0]
    teams = {c["homeAway"]: c for c in comp["competitors"]}
    dk = next((o for o in comp.get("odds", []) if "draft" in g(o, "provider", "name").lower()), None)
    row = {
        "game_id": ev["id"], "kickoff_utc": ev["date"], "status": g(ev, "status", "type", "state"),
        "away": g(teams["away"], "team", "abbreviation"), "home": g(teams["home"], "team", "abbreviation"),
        "away_name": g(teams["away"], "team", "displayName"), "home_name": g(teams["home"], "team", "displayName"),
        "away_score": teams["away"].get("score"), "home_score": teams["home"].get("score"),
        "neutral": comp.get("neutralSite"), "conference_game": comp.get("conferenceCompetition"),
        "venue": g(comp, "venue", "fullName"), "weather": g(ev, "weather", "displayValue"),
        "temp_f": g(ev, "weather", "temperature"),
    }
    if dk:
        for side in ("home", "away"):
            row[f"{side}_spread"] = g(dk, "pointSpread", side, "close", "line")
            row[f"{side}_spread_odds"] = g(dk, "pointSpread", side, "close", "odds")
            row[f"{side}_spread_open"] = g(dk, "pointSpread", side, "open", "line")
            row[f"{side}_ml"] = g(dk, "moneyline", side, "close", "odds")
            row[f"{side}_ml_open"] = g(dk, "moneyline", side, "open", "odds")
        row["total"] = dk.get("overUnder")
        row["over_odds"] = g(dk, "total", "over", "close", "odds")
        row["under_odds"] = g(dk, "total", "under", "close", "odds")
        row["total_open"] = g(dk, "total", "over", "open", "line")
    return row

os.makedirs(OUT, exist_ok=True)
stamp = now.strftime("%Y-%m-%dT%H:%MZ")
for lg, url in LEAGUES.items():
    try:
        rows = [parse(e) for e in fetch(url.format(d=dates)).get("events", [])]
    except Exception as e:
        print(f"FAIL {lg}: {e}")
        continue
    with open(f"{OUT}/latest_{lg}.json", "w") as f:
        json.dump({"pulled_utc": stamp, "source": "ESPN scoreboard, provider DraftKings", "games": rows}, f, indent=1)
    hist = f"{OUT}/history_{lg}.csv"
    cols = ["pulled_utc", "game_id", "kickoff_utc", "status", "away", "home", "away_spread", "home_spread",
            "home_spread_odds", "away_spread_odds", "total", "over_odds", "under_odds", "away_ml", "home_ml",
            "away_score", "home_score"]
    new = not os.path.exists(hist)
    with open(hist, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            if r.get("status") != "post" or r.get("home_score"):
                w.writerow({"pulled_utc": stamp, **r})
    print(f"ok   {lg}: {len(rows)} games")
