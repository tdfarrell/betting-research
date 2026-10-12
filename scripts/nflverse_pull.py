#!/usr/bin/env python3
"""Pull nflverse data and boil it down to small per-game files in data/nfl/.

Raw play-by-play is ~20 MB a season, so it is downloaded, summarized and thrown away:
  data/nfl/schedule.csv     every game (results, closing consensus lines, projected QBs, rest, roof, weather)
  data/nfl/team_games.csv   one row per team per game: EPA, success, explosives, pace, red zone, etc.
  data/nfl/qb_games.csv     one row per QB per game: dropbacks and EPA (for QB adjustments)
  data/nfl/injuries.csv     current season injury reports (latest week)

Usage:
  python scripts/nflverse_pull.py                 current + previous season (what the Action runs)
  python scripts/nflverse_pull.py 2021 2026       a range of seasons (for backtests)
  NFLVERSE_LOCAL=/dir python scripts/...          read pbp/schedule files from a local folder instead
"""
import csv, datetime as dt, gzip, io, json, os, sys, urllib.request

REL = "https://github.com/nflverse/nflverse-data/releases/download"
OUT = "data/nfl"
LOCAL = os.environ.get("NFLVERSE_LOCAL")
today = dt.date.today()
CUR = today.year if today.month >= 8 else today.year - 1  # NFL season year

def fetch(name, url):
    if LOCAL and os.path.exists(os.path.join(LOCAL, name)):
        return open(os.path.join(LOCAL, name), "rb").read()
    req = urllib.request.Request(url, headers={"User-Agent": "betting-research"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()

def rows_of(raw, gz=False):
    if gz:
        raw = gzip.decompress(raw)
    return csv.DictReader(io.StringIO(raw.decode("utf-8")))

def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def i(x):
    v = f(x)
    return int(v) if v is not None else 0

def summarize(season):
    """Team-game and QB-game summaries from one season of play-by-play."""
    raw = fetch(f"play_by_play_{season}.csv.gz", f"{REL}/pbp/play_by_play_{season}.csv.gz")
    T, Q, last, RZ = {}, {}, {}, {}
    for p in rows_of(raw, gz=True):
        off, de = p["posteam"], p["defteam"]
        if not off or not de:
            continue
        key = (p["game_id"], off)
        t = T.get(key)
        if t is None:
            t = T[key] = {"season": season, "week": i(p["week"]), "game_id": p["game_id"], "season_type": p["season_type"],
                          "team": off, "opp": de, "home": 1 if off == p["home_team"] else 0,
                          "plays": 0, "epa": 0.0, "success": 0, "pass_plays": 0, "pass_epa": 0.0, "pass_success": 0,
                          "rush_plays": 0, "rush_epa": 0.0, "rush_success": 0, "explosive": 0, "sacks": 0, "qb_hits": 0,
                          "early_plays": 0, "early_pass": 0, "neutral_plays": 0, "neutral_secs": 0.0,
                          "third_att": 0, "third_conv": 0, "rz_drives": 0, "rz_td_drives": 0, "giveaways": 0,
                          "all_plays": 0, "all_epa": 0.0, "pass_oe_sum": 0.0, "pass_oe_n": 0}
        is_pass, is_rush = p["pass"] == "1", p["rush"] == "1"
        if p["play_type"] in ("qb_kneel", "qb_spike") or not (is_pass or is_rush):
            continue
        yl = f(p["yardline_100"])
        if yl is not None and yl <= 20:
            pid = p["receiver_player_id"] if is_pass else p["rusher_player_id"]
            if pid:
                z = RZ.setdefault((p["game_id"], pid), {"game_id": p["game_id"], "player_id": pid, "rz_tgt": 0, "rz_car": 0, "gl_car": 0, "rz_pass_att": 0})
                if is_pass:
                    z["rz_tgt"] += 1
                else:
                    z["rz_car"] += 1; z["gl_car"] += yl <= 5
            if is_pass and p["passer_player_id"]:
                z = RZ.setdefault((p["game_id"], p["passer_player_id"]), {"game_id": p["game_id"], "player_id": p["passer_player_id"], "rz_tgt": 0, "rz_car": 0, "gl_car": 0, "rz_pass_att": 0})
                z["rz_pass_att"] += 1
        epa, wp = f(p["epa"]), f(p["wp"])
        if epa is None:
            continue
        # every scrimmage play (for raw totals) ...
        t["all_plays"] += 1; t["all_epa"] += epa
        if p["interception"] == "1" or p["fumble_lost"] == "1":
            t["giveaways"] += 1
        down = i(p["down"])
        if down == 3:
            t["third_att"] += 1; t["third_conv"] += p["third_down_converted"] == "1"
        dk = (p["game_id"], off, p["fixed_drive"])
        if f(p["yardline_100"]) is not None and f(p["yardline_100"]) <= 20 and dk not in last:
            last[dk] = True; t["rz_drives"] += 1
            t["_rz"] = t.get("_rz", set()); t["_rz"].add(p["fixed_drive"])
        if p["touchdown"] == "1" and p["td_team"] == off and p["fixed_drive"] in t.get("_rz", ()):
            t["_rztd"] = t.get("_rztd", set()); t["_rztd"].add(p["fixed_drive"])
        # ... but efficiency only when the game is still competitive (garbage time removed)
        if wp is None or not (0.10 <= wp <= 0.90):
            continue
        succ = p["success"] == "1"
        t["plays"] += 1; t["epa"] += epa; t["success"] += succ
        yds = i(p["yards_gained"])
        if is_pass:
            t["pass_plays"] += 1; t["pass_epa"] += epa; t["pass_success"] += succ
            t["sacks"] += p["sack"] == "1"; t["qb_hits"] += (p["qb_hit"] == "1" or p["sack"] == "1")
            t["explosive"] += yds >= 20
        else:
            t["rush_plays"] += 1; t["rush_epa"] += epa; t["rush_success"] += succ
            t["explosive"] += yds >= 12
        if down in (1, 2) and 0.20 <= wp <= 0.80 and i(p["qtr"]) <= 3:
            t["early_plays"] += 1; t["early_pass"] += is_pass
            if f(p["pass_oe"]) is not None:
                t["pass_oe_sum"] += f(p["pass_oe"]); t["pass_oe_n"] += 1
        # neutral pace: seconds between consecutive offensive snaps in the same drive, close game, Q1-3
        gs = f(p["game_seconds_remaining"])
        prev = last.get(("pace",) + dk)
        if prev is not None and gs is not None and 0.20 <= wp <= 0.80 and i(p["qtr"]) <= 3 and 0 < prev - gs <= 60:
            t["neutral_plays"] += 1; t["neutral_secs"] += prev - gs
        if gs is not None:
            last[("pace",) + dk] = gs
        # QB-level
        if is_pass and p["passer_player_id"]:
            qk = (p["game_id"], off, p["passer_player_id"])
            q = Q.setdefault(qk, {"season": season, "week": i(p["week"]), "game_id": p["game_id"], "team": off,
                                  "qb_id": p["passer_player_id"], "qb": p["passer_player_name"], "dropbacks": 0, "epa": 0.0})
            q["dropbacks"] += 1; q["epa"] += f(p["qb_epa"]) if f(p["qb_epa"]) is not None else epa
    for t in T.values():
        t["rz_td_drives"] = len(t.pop("_rztd", set()))
        t.pop("_rz", None)
        for k in ("epa", "pass_epa", "rush_epa", "all_epa", "neutral_secs", "pass_oe_sum"):
            t[k] = round(t[k], 3)
    for q in Q.values():
        q["epa"] = round(q["epa"], 3)
    return list(T.values()), list(Q.values()), RZ

PCOLS = ["season", "week", "game_id", "season_type", "team", "opp", "player_id", "name", "position", "headshot",
         "attempts", "completions", "passing_yards", "passing_tds", "interceptions", "sacks",
         "carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds",
         "target_share", "air_yards_share", "rz_tgt", "rz_car", "gl_car", "rz_pass_att", "snaps", "snap_pct"]

def players(season, rz):
    """One row per offensive player per game: nflverse weekly stats + red-zone touches + snap counts."""
    st = rows_of(fetch(f"stats_player_week_{season}.csv", f"{REL}/stats_player/stats_player_week_{season}.csv"))
    snaps = {}
    try:
        ros = rows_of(fetch(f"roster_weekly_{season}.csv", f"{REL}/weekly_rosters/roster_weekly_{season}.csv"))
        pfr = {r["pfr_id"]: r["gsis_id"] for r in ros if r.get("pfr_id") and r.get("gsis_id")}
        for r in rows_of(fetch(f"snap_counts_{season}.csv", f"{REL}/snap_counts/snap_counts_{season}.csv")):
            g = pfr.get(r["pfr_player_id"])
            if g:
                snaps[(r["game_id"], g)] = (i(r["offense_snaps"]), f(r["offense_pct"]))
    except Exception as e:
        print(f"snaps {season}: {e}")
    out = []
    for r in st:
        if r["position"] not in ("QB", "RB", "FB", "WR", "TE") or r["season_type"] != "REG":
            continue
        z = rz.get((r["game_id"], r["player_id"]), {})
        sn = snaps.get((r["game_id"], r["player_id"]), (None, None))
        out.append({"season": season, "week": i(r["week"]), "game_id": r["game_id"], "season_type": r["season_type"],
                    "team": r["team"], "opp": r["opponent_team"], "player_id": r["player_id"], "name": r["player_display_name"],
                    "position": "RB" if r["position"] == "FB" else r["position"], "headshot": r["headshot_url"] if season == CUR else "",
                    "attempts": i(r["attempts"]), "completions": i(r["completions"]), "passing_yards": i(r["passing_yards"]),
                    "passing_tds": i(r["passing_tds"]), "interceptions": i(r["passing_interceptions"]), "sacks": i(r["sacks_suffered"]),
                    "carries": i(r["carries"]), "rushing_yards": i(r["rushing_yards"]), "rushing_tds": i(r["rushing_tds"]),
                    "targets": i(r["targets"]), "receptions": i(r["receptions"]), "receiving_yards": i(r["receiving_yards"]),
                    "receiving_tds": i(r["receiving_tds"]), "target_share": round(f(r["target_share"]) or 0, 3),
                    "air_yards_share": round(f(r["air_yards_share"]) or 0, 3),
                    "rz_tgt": z.get("rz_tgt", 0), "rz_car": z.get("rz_car", 0), "gl_car": z.get("gl_car", 0), "rz_pass_att": z.get("rz_pass_att", 0),
                    "snaps": sn[0] if sn[0] is not None else "", "snap_pct": sn[1] if sn[1] is not None else ""})
    return out

def current_roster(season):
    """Latest weekly roster rows for skill players (who is active, depth chart, headshot)."""
    ros = list(rows_of(fetch(f"roster_weekly_{season}.csv", f"{REL}/weekly_rosters/roster_weekly_{season}.csv")))
    # each team's latest listed week (a team on a bye isn't in that week's roster)
    last = {}
    for r in ros:
        last[r["team"]] = max(last.get(r["team"], 0), i(r["week"]))
    return [{"season": season, "week": i(r["week"]), "team": r["team"], "player_id": r["gsis_id"], "name": r["full_name"],
             "position": "RB" if r["position"] == "FB" else r["position"], "depth_chart_position": r["depth_chart_position"],
             "status": r["status"], "headshot": r["headshot_url"]}
            for r in ros if i(r["week"]) == last[r["team"]] and r["position"] in ("QB", "RB", "FB", "WR", "TE") and r.get("gsis_id")]

def write(name, rows, cols=None):
    os.makedirs(OUT, exist_ok=True)
    cols = cols or list(rows[0].keys())
    with open(f"{OUT}/{name}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def main(first, last_season):
    seasons = range(first, last_season + 1)
    status = {"pulled_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "seasons": list(seasons), "errors": []}
    # schedule goes back to 2012 so Elo has a long warm-up
    sched = [g for g in rows_of(fetch("games.csv", f"{REL}/schedules/games.csv")) if 2012 <= i(g["season"]) <= last_season]
    keep = ["game_id", "season", "game_type", "week", "gameday", "weekday", "gametime", "away_team", "away_score", "home_team",
            "home_score", "location", "result", "total", "overtime", "espn", "away_rest", "home_rest", "away_moneyline",
            "home_moneyline", "spread_line", "total_line", "div_game", "roof", "surface", "temp", "wind", "away_qb_id",
            "home_qb_id", "away_qb_name", "home_qb_name", "away_coach", "home_coach", "stadium"]
    write("schedule.csv", sched, keep)
    # Re-summarize only seasons that can still change; keep older summaries already in the repo.
    # Seasons outside the requested range (and finished seasons already summarized) are kept as is.
    old_t, old_q = [], []
    if os.path.exists(f"{OUT}/team_games.csv"):
        old_t = list(csv.DictReader(open(f"{OUT}/team_games.csv")))
        old_q = list(csv.DictReader(open(f"{OUT}/qb_games.csv")))
    have = {i(r["season"]) for r in old_t}
    redo = {s for s in seasons if not (s < CUR and s in have)}
    TG = [r for r in old_t if i(r["season"]) not in redo]
    QG = [r for r in old_q if i(r["season"]) not in redo]
    old_p = list(csv.DictReader(open(f"{OUT}/player_games.csv"))) if os.path.exists(f"{OUT}/player_games.csv") else []
    have_p = {i(r["season"]) for r in old_p}
    redo |= {s for s in seasons if s not in have_p}   # first run for player data re-reads those seasons
    TG = [r for r in old_t if i(r["season"]) not in redo]
    QG = [r for r in old_q if i(r["season"]) not in redo]
    PG = [r for r in old_p if i(r["season"]) not in redo]
    for s in sorted(redo):
        try:
            t, q, rz = summarize(s)
            TG += t; QG += q
            print(f"ok   pbp {s}: {len(t)} team-games")
            try:
                pl = players(s, rz); PG += pl
                print(f"ok   players {s}: {len(pl)} player-games")
            except Exception as e:
                status["errors"].append(f"players {s}: {type(e).__name__}: {e}")
                PG += [r for r in old_p if i(r["season"]) == s]
        except Exception as e:
            status["errors"].append(f"pbp {s}: {type(e).__name__}: {e}")
            TG += [r for r in old_t if i(r["season"]) == s]; QG += [r for r in old_q if i(r["season"]) == s]
            PG += [r for r in old_p if i(r["season"]) == s]
            print(f"FAIL pbp {s}: {e}")
    TG.sort(key=lambda r: (i(r["season"]), i(r["week"]), r["game_id"], r["team"]))
    QG.sort(key=lambda r: (i(r["season"]), i(r["week"]), r["game_id"], r["team"]))
    write("team_games.csv", TG); write("qb_games.csv", QG)
    PG.sort(key=lambda r: (i(r["season"]), i(r["week"]), r["game_id"], r["team"], r["player_id"]))
    if PG:
        write("player_games.csv", PG, PCOLS)
    try:
        rc = current_roster(CUR); write("roster_current.csv", rc)
        print(f"ok   roster: {len(rc)} skill players, {len({r['team'] for r in rc})} teams")
    except Exception as e:
        status["errors"].append(f"roster: {type(e).__name__}: {e}")
    try:
        inj = list(rows_of(fetch(f"injuries_{CUR}.csv", f"{REL}/injuries/injuries_{CUR}.csv")))
        wk = max(i(r["week"]) for r in inj) if inj else 0
        write("injuries.csv", [r for r in inj if i(r["week"]) == wk] or [{"week": wk}],
              ["season", "week", "team", "gsis_id", "position", "full_name", "report_primary_injury", "report_status", "practice_status"])
        print(f"ok   injuries week {wk}")
    except Exception as e:
        status["errors"].append(f"injuries: {type(e).__name__}: {e}")
    status["team_games"] = len(TG)
    with open(f"{OUT}/_status.json", "w") as fh:
        json.dump(status, fh, indent=1)

if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(a[0] if a else CUR - 1, a[1] if len(a) > 1 else CUR)
