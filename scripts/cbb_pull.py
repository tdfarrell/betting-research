#!/usr/bin/env python3
"""College basketball (D-I men's) scores and closing lines from ESPN's public API.

  python scripts/cbb_pull.py                    incremental: the last few days of the current season
  python scripts/cbb_pull.py backfill 2026      one ESPN season (2026 = 2025-26), Nov-Apr
  python scripts/cbb_pull.py lines [max]        fill closing lines from ESPN's core odds endpoint

Writes data/cbb/
  games.csv        one row per finished game: date, ids, abbreviations, names, scores, neutral site, conference game,
                   overtime periods, closing spread/total when the scoreboard carried them
  lines_close.csv  closing spread/total per game (core odds endpoint; DraftKings when listed)
  _status.json
Box scores aren't pulled (360 teams x ~30 games); the ratings use scores only. Team keys are ESPN team ids.
"""
import csv, datetime as dt, json, os, sys, time, urllib.request

OUT = "data/cbb"
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball"
ET = dt.timezone(dt.timedelta(hours=-4))
GCOLS = ["season", "date", "game_id", "home_id", "away_id", "home", "away", "home_name", "away_name", "home_score", "away_score",
         "neutral", "conf", "ot", "home_spread", "total", "provider"]
LCOLS = ["season", "date", "game_id", "home_spread", "total", "provider"]
errors = []

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"})
    for k in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if k == 2:
                raise
            time.sleep(2 + 3 * k)

def num(x):
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return None

def read(name, key):
    p = f"{OUT}/{name}"
    return {key(r): r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}

def write(name, rows, cols, sort):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(sorted(rows, key=sort))

def pull_days(days, want_season=None):
    G = read("games.csv", lambda r: r["game_id"])
    new = 0
    for day in days:
        try:
            sb = get(f"{SITE}/scoreboard?dates={day.strftime('%Y%m%d')}&groups=50&limit=400")
        except Exception as e:
            errors.append(f"scoreboard {day}: {type(e).__name__}: {e}"); continue
        for ev in sb.get("events", []):
            sn = ev.get("season") or {}
            if want_season and sn.get("year") != want_season:
                continue
            if sn.get("type") not in (2, 3):    # regular season + postseason
                continue
            c = ev["competitions"][0]; st = c["status"]["type"]
            if not (st.get("completed") and st.get("state") == "post"):
                continue
            side = {x["homeAway"]: x for x in c["competitors"]}
            h, a = side["home"], side["away"]
            o = next((x for x in c.get("odds") or [] if "draft" in ((x.get("provider") or {}).get("name") or "").lower()), None) or ((c.get("odds") or [None])[0])
            date = dt.datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).astimezone(ET).date().isoformat()
            if ev["id"] not in G:
                new += 1
            G[ev["id"]] = {"season": sn.get("year"), "date": date, "game_id": ev["id"],
                           "home_id": h["team"]["id"], "away_id": a["team"]["id"],
                           "home": h["team"].get("abbreviation"), "away": a["team"].get("abbreviation"),
                           "home_name": h["team"].get("shortDisplayName") or h["team"].get("displayName"),
                           "away_name": a["team"].get("shortDisplayName") or a["team"].get("displayName"),
                           "home_score": h.get("score"), "away_score": a.get("score"),
                           "neutral": int(bool(c.get("neutralSite"))), "conf": int(bool(c.get("conferenceCompetition"))),
                           "ot": max(0, len(h.get("linescores") or []) - 2),
                           "home_spread": num(o.get("spread")) if o and o.get("spread") is not None else "",
                           "total": o.get("overUnder", "") if o else "", "provider": ((o or {}).get("provider") or {}).get("name", "")}
        time.sleep(0.15)
    write("games.csv", G.values(), GCOLS, lambda r: (str(r["date"]), str(r["game_id"])))
    return new

def core_lines(max_games=8000):
    G = read("games.csv", lambda r: r["game_id"])
    L = read("lines_close.csv", lambda r: r["game_id"])
    todo = [g for g in sorted(G.values(), key=lambda g: g["date"], reverse=True)
            if g["game_id"] not in L and g.get("home_spread") in ("", None)][:max_games]
    got = 0
    for g in todo:
        gid = g["game_id"]
        try:
            d = get(f"{CORE}/events/{gid}/competitions/{gid}/odds?limit=20")
        except Exception as e:
            L[gid] = {"season": g["season"], "date": g["date"], "game_id": gid, "home_spread": "", "total": "", "provider": "none"}
            continue
        items = d.get("items", [])
        it = next((x for x in items if "draft" in ((x.get("provider") or {}).get("name") or "").lower()), None) or (items[0] if items else None)
        L[gid] = {"season": g["season"], "date": g["date"], "game_id": gid,
                  "home_spread": num(it.get("spread")) if it and it.get("spread") is not None else "",
                  "total": (it or {}).get("overUnder", ""), "provider": ((it or {}).get("provider") or {}).get("name", "none")}
        got += 1
        if got % 500 == 0:
            write("lines_close.csv", L.values(), LCOLS, lambda r: (str(r["date"]), str(r["game_id"])))
        time.sleep(0.05)
    write("lines_close.csv", L.values(), LCOLS, lambda r: (str(r["date"]), str(r["game_id"])))
    return got

def main():
    today = dt.datetime.now(ET).date()
    if len(sys.argv) > 2 and sys.argv[1] == "backfill":
        s = int(sys.argv[2])
        start, end = dt.date(s - 1, 11, 1), min(dt.date(s, 4, 10), today)
        n = pull_days([start + dt.timedelta(days=i) for i in range((end - start).days + 1)], want_season=s)
        print(f"cbb backfill {s}: {n} games")
    elif len(sys.argv) > 1 and sys.argv[1] == "lines":
        print(f"cbb closing lines: {core_lines(int(sys.argv[2]) if len(sys.argv) > 2 else 8000)}")
    else:
        if not (today.month >= 11 or today.month <= 4):
            print("cbb: off-season, nothing to pull"); return
        n = pull_days([today - dt.timedelta(days=i) for i in range(3, -1, -1)])
        print(f"cbb: {n} new games")
    json.dump({"pulled_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "errors": errors[:40]},
              open(f"{OUT}/_status.json", "w"), indent=1)
    for e in errors[:10]:
        print("  ", e)

if __name__ == "__main__":
    main()
