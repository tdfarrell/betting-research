#!/usr/bin/env python3
"""Opening and closing game lines (spread, total, prices) for finished games, from ESPN's core API.

The model rebuild tests every model against the line it would actually bet into, so it needs openers as well as
closers. ESPN keeps DraftKings' (provider 100) or ESPN BET's (58) odds item on finished games, with open / close /
current blocks for each side.

  python scripts/lines_history_pull.py nba 2025 2026 2027
  python scripts/lines_history_pull.py cbb 2025 2026
  python scripts/lines_history_pull.py nfl 2022 2023 2024 2025
  python scripts/lines_history_pull.py cfb 2022 2023 2024 2025

Writes data/lines_hist/<league>_<season>.jsonl.gz: one record per game with the trimmed odds item (raw ESPN fields;
scripts/lines_hist.py parses them).
"""
import csv, gzip, json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

OUT = "data/lines_hist"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
PATH = {"nba": "basketball/leagues/nba", "cbb": "basketball/leagues/mens-college-basketball",
        "nfl": "football/leagues/nfl", "cfb": "football/leagues/college-football"}
KEEP = ("provider", "details", "overUnder", "spread", "overOdds", "underOdds", "awayTeamOdds", "homeTeamOdds",
        "open", "close", "current", "moneylineWinner", "spreadWinner")

def get(url):
    for k in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1 + 2 * k)
        except Exception:
            time.sleep(1 + 2 * k)
    return None

def trim(x, depth=0):
    """Drop $ref links and team objects so records stay small."""
    if isinstance(x, dict):
        return {k: trim(v, depth + 1) for k, v in x.items() if k not in ("$ref", "team", "links", "propBets") and not k.startswith("$")}
    if isinstance(x, list):
        return [trim(v, depth + 1) for v in x]
    return x

def one(lg, ev):
    d = get(f"https://sports.core.api.espn.com/v2/sports/{PATH[lg]}/events/{ev}/competitions/{ev}/odds?lang=en&region=us")
    items = (d or {}).get("items", [])
    by = {str((it.get("provider") or {}).get("id")): it for it in items}
    for pid in ("100", "58", "1002", "40"):
        it = by.get(pid)
        if it:
            rec = {k: trim(it[k]) for k in KEEP if k in it}
            rec["provider_id"] = pid
            rec["game_id"] = ev
            rec["providers"] = sorted(by)
            return rec
    return {"game_id": ev, "providers": sorted(by), "missing": True}

def games(lg, season):
    s = str(season)
    if lg == "nba":
        return sorted({r["game_id"] for r in csv.DictReader(open("data/nba/team_games.csv")) if r["season"] == s})
    if lg == "cbb":
        return sorted({r["game_id"] for r in csv.DictReader(open("data/cbb/games.csv")) if r["season"] == s})
    if lg == "nfl":
        return sorted({r["espn"].split(".")[0] for r in csv.DictReader(open("data/nfl/schedule.csv"))
                       if r["season"] == s and r.get("espn") and r["home_score"] not in ("", None)})
    return sorted({r["id"] for r in csv.DictReader(open("data/cfbd/history_games.csv")) if r["season"] == s})

def main():
    lg = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    for season in sys.argv[2:]:
        path = f"{OUT}/{lg}_{season}.jsonl.gz"
        done = set()
        if os.path.exists(path):
            for line in gzip.open(path, "rt"):
                r = json.loads(line)
                if not r.get("missing"):
                    done.add(r["game_id"])
        todo = [g for g in games(lg, season) if g not in done]
        t0 = time.time()
        recs = []
        with ThreadPoolExecutor(8) as ex:
            for r in ex.map(lambda g: one(lg, g), todo):
                recs.append(r)
        old = [json.loads(l) for l in gzip.open(path, "rt")] if os.path.exists(path) else []
        old = [r for r in old if r["game_id"] in done]
        with gzip.open(path, "wt") as f:
            for r in old + recs:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        have = sum(1 for r in recs if not r.get("missing"))
        print(f"{lg} {season}: {len(todo)} fetched, {have} with odds, {len(old)} kept ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    main()
