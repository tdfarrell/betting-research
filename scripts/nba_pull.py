#!/usr/bin/env python3
"""NBA game logs from ESPN's public API (free; stats.nba.com and cdn.nba.com block GitHub Actions).

  python scripts/nba_pull.py                      incremental: last few days of the current season + injuries
  python scripts/nba_pull.py backfill 2026        one ESPN season (2026 = 2025-26), regular season only

Writes data/nba/
  player_games.csv  one row per player per regular-season game (DNPs included with min 0 and dnp=1)
  team_games.csv    one row per team per game: box totals, possessions, points, opponent, home, rest days
  lines_close.csv   DraftKings closing spread / total per finished game (from the scoreboard's odds)
  injuries.csv      current ESPN injury report (athlete id, name, team, status, detail)
  _status.json
Seasons use ESPN's year (2027 = the 2026-27 season). Player ids are ESPN athlete ids, the same ids ESPN's
prop lines use.
"""
import csv, datetime as dt, json, os, re, sys, time, urllib.request

OUT = "data/nba"
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ET = dt.timezone(dt.timedelta(hours=-4))
PCOLS = ["season", "date", "game_id", "team", "opp", "home", "player_id", "name", "pos", "starter", "dnp", "dnp_reason",
         "min", "pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta", "reb", "oreb", "dreb", "ast", "tov", "stl", "blk", "pf", "pm"]
TCOLS = ["season", "date", "game_id", "team", "opp", "home", "pts", "opp_pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta",
         "oreb", "dreb", "reb", "ast", "tov", "stl", "blk", "poss", "min", "ot", "rest"]
LCOLS = ["season", "date", "game_id", "home", "away", "home_spread", "total", "home_score", "away_score", "provider"]
NBA_TEAMS = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
             "NO", "NY", "OKC", "ORL", "PHI", "PHX", "POR", "SA", "SAC", "TOR", "UTAH", "WSH"}   # skips All-Star games
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
        return float(str(x).replace(",", "").replace("+", ""))
    except (TypeError, ValueError):
        return 0.0

def pair(x):
    a, _, b = str(x).partition("-")
    return num(a), num(b)

def read(name, key):
    p = f"{OUT}/{name}"
    if not os.path.exists(p):
        return {}
    return {key(r): r for r in csv.DictReader(open(p))}

def write(name, rows, cols, sort):
    os.makedirs(OUT, exist_ok=True)
    rows = sorted(rows, key=sort)
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def box_rows(sm, season, date, gid):
    """Player rows + team rows from one ESPN summary."""
    comp = (sm.get("header") or {}).get("competitions", [{}])[0]
    side = {c["homeAway"]: c for c in comp.get("competitors", [])}
    abbr = {s: side[s]["team"]["abbreviation"] for s in side}
    score = {s: num(side[s].get("score")) for s in side}
    home_of = {abbr["home"]: 1, abbr["away"]: 0}
    opp_of = {abbr["home"]: abbr["away"], abbr["away"]: abbr["home"]}
    periods = max(len(side["home"].get("linescores") or []), 4)
    prow, trow = [], {}
    for t in (sm.get("boxscore") or {}).get("players", []):
        tm = t["team"]["abbreviation"]
        for grp in t.get("statistics", []):
            keys = grp.get("keys") or []
            for a in grp.get("athletes", []):
                ath = a.get("athlete") or {}
                st = dict(zip(keys, a.get("stats") or []))
                dnp = bool(a.get("didNotPlay")) or not st
                fgm, fga = pair(st.get("fieldGoalsMade-fieldGoalsAttempted"))
                tpm, tpa = pair(st.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
                ftm, fta = pair(st.get("freeThrowsMade-freeThrowsAttempted"))
                r = {"season": season, "date": date, "game_id": gid, "team": tm, "opp": opp_of.get(tm), "home": home_of.get(tm),
                     "player_id": ath.get("id"), "name": ath.get("displayName"), "pos": (ath.get("position") or {}).get("abbreviation", ""),
                     "starter": int(bool(a.get("starter"))), "dnp": int(dnp), "dnp_reason": a.get("reason", "") if dnp else "",
                     "min": num(st.get("minutes")), "pts": num(st.get("points")), "fgm": fgm, "fga": fga, "tpm": tpm, "tpa": tpa,
                     "ftm": ftm, "fta": fta, "reb": num(st.get("rebounds")), "oreb": num(st.get("offensiveRebounds")),
                     "dreb": num(st.get("defensiveRebounds")), "ast": num(st.get("assists")), "tov": num(st.get("turnovers")),
                     "stl": num(st.get("steals")), "blk": num(st.get("blocks")), "pf": num(st.get("fouls")), "pm": num(st.get("plusMinus"))}
                if dnp:
                    for k in PCOLS[12:]:
                        r[k] = 0
                prow.append(r)
    for tm in abbr.values():
        ps = [r for r in prow if r["team"] == tm]
        tot = {k: sum(r[k] for r in ps) for k in ("fgm", "fga", "tpm", "tpa", "ftm", "fta", "oreb", "dreb", "reb", "ast", "tov", "stl", "blk", "min")}
        s = "home" if home_of[tm] else "away"
        trow[tm] = {"season": season, "date": date, "game_id": gid, "team": tm, "opp": opp_of[tm], "home": home_of[tm],
                    "pts": score[s], "opp_pts": score["away" if s == "home" else "home"], **tot,
                    "poss": round(tot["fga"] - tot["oreb"] + tot["tov"] + 0.44 * tot["fta"], 1), "ot": periods - 4}
    for r in prow:
        for k in PCOLS[12:]:
            r[k] = int(r[k]) if float(r[k]).is_integer() else r[k]
    return prow, list(trow.values())

def dk_close(ev):
    c = ev["competitions"][0]
    dk = next((o for o in c.get("odds", []) or [] if "draft" in ((o.get("provider") or {}).get("name") or "").lower()), None)
    o = dk or ((c.get("odds") or [None])[0])
    if not o:
        return None
    side = {x["homeAway"]: x for x in c["competitors"]}
    hs = ((o.get("pointSpread") or {}).get("home") or {}).get("close", {}).get("line")
    if hs is None and o.get("spread") is not None:
        hs = o.get("spread")   # ESPN's "spread" is the home line
    return {"home": side["home"]["team"]["abbreviation"], "away": side["away"]["team"]["abbreviation"],
            "home_spread": num(hs) if hs is not None else "", "total": o.get("overUnder", ""),
            "home_score": side["home"].get("score"), "away_score": side["away"].get("score"),
            "provider": (o.get("provider") or {}).get("name", "")}

def pull_days(days, want_season=None):
    P = read("player_games.csv", lambda r: (r["game_id"], r["player_id"]))
    T = read("team_games.csv", lambda r: (r["game_id"], r["team"]))
    L = read("lines_close.csv", lambda r: r["game_id"])
    have = {k[0] for k in T}
    new_games = 0
    for day in days:
        try:
            sb = get(f"{SITE}/scoreboard?dates={day.strftime('%Y%m%d')}")
        except Exception as e:
            errors.append(f"scoreboard {day}: {type(e).__name__}: {e}"); continue
        for ev in sb.get("events", []):
            sn = ev.get("season") or {}
            if sn.get("type") != 2 or (want_season and sn.get("year") != want_season):
                continue
            st = ev["competitions"][0]["status"]["type"]
            if not (st.get("completed") and st.get("state") == "post"):
                continue
            gid, season = ev["id"], sn.get("year")
            if {c["team"]["abbreviation"] for c in ev["competitions"][0]["competitors"]} - NBA_TEAMS:
                continue
            date = dt.datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).astimezone(ET).date().isoformat()
            try:
                lc = dk_close(ev)
                if lc:
                    L[gid] = {"season": season, "date": date, "game_id": gid, **lc}
            except Exception as e:
                errors.append(f"odds {gid}: {e}")
            if gid in have:
                continue
            try:
                sm = get(f"{SITE}/summary?event={gid}")
                prow, trow = box_rows(sm, season, date, gid)
                for r in prow:
                    P[(gid, r["player_id"])] = r
                for r in trow:
                    T[(gid, r["team"])] = r
                have.add(gid); new_games += 1
                time.sleep(0.15)
            except Exception as e:
                errors.append(f"summary {gid}: {type(e).__name__}: {e}")
        time.sleep(0.15)
    # rest days per team (days since its previous game; 0 never happens, 1 = back-to-back)
    by_team = {}
    for r in T.values():
        by_team.setdefault((str(r["season"]), r["team"]), []).append(r)
    for rows in by_team.values():
        rows.sort(key=lambda r: r["date"])
        prev = None
        for r in rows:
            d = dt.date.fromisoformat(r["date"])
            r["rest"] = (d - prev).days if prev else ""
            prev = d
    write("player_games.csv", P.values(), PCOLS, lambda r: (str(r["date"]), str(r["game_id"]), str(r["team"]), -float(r["min"] or 0)))
    write("team_games.csv", T.values(), TCOLS, lambda r: (str(r["date"]), str(r["game_id"]), str(r["team"])))
    write("lines_close.csv", L.values(), LCOLS, lambda r: (str(r["date"]), str(r["game_id"])))
    return new_games

CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"

def core_lines(max_games=4000):
    """Closing DK spread/total for finished games that the scoreboard didn't carry odds for (ESPN keeps them on the
    core API's odds endpoint, listed per provider)."""
    T = read("team_games.csv", lambda r: (r["game_id"], r["team"]))
    L = read("lines_close.csv", lambda r: r["game_id"])
    games = {}
    for r in T.values():
        g = games.setdefault(r["game_id"], {"season": r["season"], "date": r["date"]})
        g["home" if str(r["home"]) == "1" else "away"] = r["team"]
        g["home_score" if str(r["home"]) == "1" else "away_score"] = r["pts"]
    todo = [gid for gid in sorted(games, key=lambda k: games[k]["date"], reverse=True) if gid not in L][:max_games]
    got = 0
    for gid in todo:
        try:
            d = get(f"{CORE}/events/{gid}/competitions/{gid}/odds?limit=20")
        except Exception as e:
            errors.append(f"core odds {gid}: {type(e).__name__}: {e}"); continue
        items = d.get("items", [])
        it = next((x for x in items if "draft" in ((x.get("provider") or {}).get("name") or "").lower()), None) or (items[0] if items else None)
        if not it:
            continue
        hs = it.get("spread")
        hto = it.get("homeTeamOdds") or {}
        if hs is None:
            hs = ((hto.get("close") or {}).get("pointSpread") or {}).get("value")
        g = games[gid]
        L[gid] = {"season": g["season"], "date": g["date"], "game_id": gid, "home": g.get("home"), "away": g.get("away"),
                  "home_spread": num(hs) if hs is not None else "", "total": it.get("overUnder", ""),
                  "home_score": g.get("home_score"), "away_score": g.get("away_score"), "provider": (it.get("provider") or {}).get("name", "")}
        got += 1
        time.sleep(0.1)
    write("lines_close.csv", L.values(), LCOLS, lambda r: (str(r["date"]), str(r["game_id"])))
    return got

def injuries():
    try:
        d = get(f"{SITE}/injuries")
    except Exception as e:
        errors.append(f"injuries: {type(e).__name__}: {e}"); return 0
    rows = []
    for t in d.get("injuries", []):
        tm = (t.get("team") or {}).get("abbreviation") or t.get("displayName", "")
        for i in t.get("injuries", []):
            a = i.get("athlete") or {}
            team = ((a.get("team") or {}).get("abbreviation")) or tm
            pid = a.get("id") or next((m.group(1) for l in a.get("links", []) for m in [re.search(r"/id/(\d+)", l.get("href", ""))] if m), "")
            rows.append({"player_id": pid,
                         "name": a.get("displayName"), "team": team, "status": i.get("status"),
                         "detail": (i.get("shortComment") or i.get("details", {}).get("type") or "")[:160], "date": i.get("date", "")[:10]})
    write("injuries.csv", rows, ["player_id", "name", "team", "status", "detail", "date"], lambda r: (str(r["team"]), str(r["name"])))
    return len(rows)

def main():
    today = dt.datetime.now(ET).date()
    if len(sys.argv) > 2 and sys.argv[1] == "backfill":
        season = int(sys.argv[2])   # ESPN year: 2026 = 2025-26
        start, end = dt.date(season - 1, 10, 15), min(dt.date(season, 4, 20), today)
        days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
        n = pull_days(days, want_season=season)
        print(f"backfill {season}: {n} new games")
    elif len(sys.argv) > 1 and sys.argv[1] == "lines":
        print(f"closing lines: {core_lines()} games added")
    else:
        T = read("team_games.csv", lambda r: (r["game_id"], r["team"]))
        last = max((dt.date.fromisoformat(r["date"]) for r in T.values()), default=today - dt.timedelta(days=4))
        start = min(last - dt.timedelta(days=1), today - dt.timedelta(days=2))
        days = [start + dt.timedelta(days=i) for i in range((today - start).days + 1)]
        n = pull_days(days)
        k = injuries()
        if n:
            core_lines(60)
        print(f"nba: {n} new games, {k} injury rows")
    json.dump({"pulled_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "errors": errors[:40]},
              open(f"{OUT}/_status.json", "w"), indent=1)
    for e in errors[:10]:
        print("  ", e)

if __name__ == "__main__":
    main()
