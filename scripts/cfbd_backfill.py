#!/usr/bin/env python3
"""One-time CFBD backfill for the CFB calibration backtest (needs CFBD_KEY; run from Actions).

  python scripts/cfbd_backfill.py 2022 2025

Writes data/cfbd/history_games.csv: every FBS regular-season game with pregame Elo, score and the
DraftKings line (open + close) when CFBD has it, else the consensus line (provider column says which).
About 2 API calls per season.
"""
import csv, json, os, sys, urllib.request

KEY = os.environ.get("CFBD_KEY")
if not KEY:
    sys.exit("CFBD_KEY is not set")
BASE = "https://api.collegefootballdata.com"

def get(path):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {KEY}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def main(first, last):
    out = []
    for y in range(first, last + 1):
        games = {g["id"]: g for g in get(f"/games?year={y}&seasonType=regular&division=fbs")}
        lines = get(f"/lines?year={y}&seasonType=regular")
        L = {}
        for l in lines:
            pick = None
            for want in ("DraftKings", "consensus", "ESPN Bet", "Bovada"):
                pick = next((x for x in l.get("lines", []) if (x.get("provider") or "").lower() == want.lower()), None)
                if pick:
                    break
            if pick:
                L[l["id"]] = pick
        n = 0
        for gid, g in games.items():
            if not g.get("completed") or g.get("homePoints") is None:
                continue
            ln = L.get(gid, {})
            out.append({"season": y, "week": g["week"], "id": gid, "start": g["startDate"], "neutral": int(bool(g.get("neutralSite"))),
                        "conference_game": int(bool(g.get("conferenceGame"))),
                        "home": g["homeTeam"], "away": g["awayTeam"], "home_conf": g.get("homeConference"), "away_conf": g.get("awayConference"),
                        "home_div": g.get("homeClassification"), "away_div": g.get("awayClassification"),
                        "home_points": g["homePoints"], "away_points": g["awayPoints"],
                        "home_elo": g.get("homePregameElo"), "away_elo": g.get("awayPregameElo"),
                        "provider": ln.get("provider"), "spread": f(ln.get("spread")), "spread_open": f(ln.get("spreadOpen")),
                        "total": f(ln.get("overUnder")), "total_open": f(ln.get("overUnderOpen"))})
            n += 1
        print(f"ok {y}: {n} games, {sum(1 for r in out if r['season'] == y and r['spread'] is not None)} with a line, "
              f"{sum(1 for r in out if r['season'] == y and (r['provider'] or '').lower() == 'draftkings')} DraftKings")
    os.makedirs("data/cfbd", exist_ok=True)
    with open("data/cfbd/history_games.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(a[0] if a else 2022, a[1] if len(a) > 1 else 2025)
