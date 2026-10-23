#!/usr/bin/env python3
"""Historical player prop lines for finished games, from ESPN's core API, for backtesting prop thresholds.

ESPN keeps the pre-game prop board on finished games: DraftKings (provider 100) where it still serves it, else
ESPN BET (provider 58, ESPN's book until it switched to DraftKings in late 2025). ESPN BET's lines are a stand-in
for DraftKings' (books' prop lines sit within a half point or so of each other); every row says which book.

  python scripts/props_history_pull.py nba 2025 2026     NBA seasons (ESPN years: 2026 = 2025-26)
  python scripts/props_history_pull.py nfl 2024 2025     NFL seasons

Writes data/props_hist/<league>_<season>.csv.gz, one row per board item (alternate lines included):
  game_id, date, athlete_id (ESPN), market, line, side (over/under/blank), price (American), open_line, provider, updated
The backtest picks each player's MAIN line from these (scripts/props_backtest.py).
NBA markets: pts reb ast tpm pra pr pa ra stl blk.  NFL: pass_yds pass_td rush_yds rec_yds rec rr_yds (lines only).
"""
import csv, gzip, json, os, re, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

OUT = "data/props_hist"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
NBA_T = {"total points": "pts", "total rebounds": "reb", "total assists": "ast", "total 3-point field goals": "tpm",
         "total points, rebounds, and assists": "pra", "total points and rebounds": "pr", "total points and assists": "pa",
         "total assists and rebounds": "ra", "total steals": "stl", "total blocks": "blk"}
NFL_T = {"total passing yards": "pass_yds", "total passing touchdowns": "pass_td", "total rushing yards": "rush_yds",
         "total receiving yards": "rec_yds", "total receptions": "rec", "total rushing plus receiving yards": "rr_yds"}
SPORT = {"nba": "basketball", "nfl": "football"}

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

def market(lg, name):
    t = re.sub(r"\s*\(incl\. overtime\)\s*$", "", (name or "").strip().lower())
    return (NBA_T if lg == "nba" else NFL_T).get(t)

def one_game(lg, ev, date):
    base = f"https://sports.core.api.espn.com/v2/sports/{SPORT[lg]}/leagues/{lg}/events/{ev}/competitions/{ev}/odds"
    d = get(base)
    provs = [str((it.get("provider") or {}).get("id")) for it in (d or {}).get("items", [])]
    for pid in [p for p in ("100", "58") if p in provs]:
        items, page = [], 1
        while True:
            pb = get(f"{base}/{pid}/propBets?lang=en&region=us&limit=1000&page={page}")
            if not pb:
                break
            items += pb.get("items", [])
            if page >= int(pb.get("pageCount", 1) or 1):
                break
            page += 1
        if not items:
            continue
        rows = []
        for it in items:
            mk = market(lg, (it.get("type") or {}).get("name"))
            m = re.search(r"/athletes/(\d+)", (it.get("athlete") or {}).get("$ref", ""))
            cur = it.get("current") or {}
            tgt = (cur.get("target") or {}).get("value")
            if tgt is None:
                tgt = ((it.get("open") or {}).get("target") or {}).get("value")
            if not mk or not m or tgt is None:
                continue
            side = "over" if "over" in cur else "under" if "under" in cur else ""
            am = ((it.get("odds") or {}).get("american") or {})
            rows.append([ev, date, m.group(1), mk, float(tgt), side, am.get("value", ""),
                         ((it.get("open") or {}).get("target") or {}).get("value", ""), pid, it.get("lastUpdated", "")])
        return rows
    return []

def games_for(lg, season):
    if lg == "nba":
        seen = {}
        for r in csv.DictReader(open("data/nba/team_games.csv")):
            if r["season"] == str(season):
                seen[r["game_id"]] = r["date"]
        return sorted(seen.items(), key=lambda x: x[1])
    out = []
    for r in csv.DictReader(open("data/nfl/schedule.csv")):
        if r["season"] == str(season) and r["game_type"] == "REG" and r["espn"] and r["home_score"] not in ("", None):
            out.append((r["espn"].split(".")[0], r["gameday"]))
    return out

def main():
    lg = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    for season in [a for a in sys.argv[2:] if not a.startswith("--")]:
        games = games_for(lg, season)
        path = f"{OUT}/{lg}_{season}.csv.gz"
        if os.path.exists(path) and "--fresh" in sys.argv:
            os.remove(path)
        done = set()
        rows = []
        if os.path.exists(path):
            for r in csv.reader(gzip.open(path, "rt")):
                if r and r[0] != "game_id":
                    rows.append(r); done.add(r[0])
        todo = [(g, d) for g, d in games if g not in done]
        t0, n_with = time.time(), 0
        with ThreadPoolExecutor(8) as ex:
            for res in ex.map(lambda x: one_game(lg, x[0], x[1]), todo):
                if res:
                    rows += res; n_with += 1
        with gzip.open(path, "wt", newline="") as f:
            w = csv.writer(f); w.writerow(["game_id", "date", "athlete_id", "market", "line", "side", "price", "open_line", "provider", "updated"]); w.writerows(rows)
        print(f"{lg} {season}: {len(games)} games, {n_with} new with props, {len(rows)} lines total ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    main()
