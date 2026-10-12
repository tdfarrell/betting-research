#!/usr/bin/env python3
"""Final scores and player box scores from ESPN's public API, for grading bets and the Screener.

Writes data/results/<league>.json for nfl, cfb, nba, cbb, mlb:
  {"league", "updated_utc", "games": [{
      "id", "kickoff_utc", "day_et", "state" (pre/in/post), "final" (bool),
      "home": {"abbr", "name", "short", "loc", "nick", "score"}, "away": {...},
      "players": {"<display name>": {"team": abbr, "s": {stat: number}, "dnp": bool}}   # final games only
  }]}
Stat keys (s):
  football   pass_cmp pass_att pass_yds pass_td int rush_att rush_yds rush_td rec rec_yds rec_td tgt
             ret_td (kick/punt/defensive return TDs)  + derived rr_yds (rush+rec) and td (rush+rec+return)
  basketball min pts reb ast tpm (3s made) stl blk to  + pra (pts+reb+ast)
  baseball   batting: h ab r rbi hr bb so ; pitching: k ip er ha (hits allowed) bb_p
Scoreboards are pulled for the last few days (one request per day per league); a box score is fetched once per
final game and then kept. Games older than 21 days are dropped.
"""
import datetime as dt, json, os, sys, time, urllib.request

OUT = "data/results"
PATHS = {"nfl": "football/nfl", "cfb": "football/college-football", "nba": "basketball/nba",
         "cbb": "basketball/mens-college-basketball", "mlb": "baseball/mlb"}
EXTRA = {"cfb": "&groups=80&limit=300", "cbb": "&groups=50&limit=400"}
ET = dt.timezone(dt.timedelta(hours=-4))
KEEP_DAYS, LOOKBACK = 21, 4
errors = []

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
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
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None

def made(x):  # "3-7" -> 3
    try:
        return float(str(x).split("-")[0])
    except (TypeError, ValueError):
        return None

def team(c):
    t = c.get("team", {})
    return {"abbr": t.get("abbreviation"), "name": t.get("displayName"), "short": t.get("shortDisplayName"),
            "loc": t.get("location"), "nick": t.get("name"), "score": num(c.get("score"))}

FB = {("passing", "C/ATT"): None, ("passing", "YDS"): "pass_yds", ("passing", "TD"): "pass_td", ("passing", "INT"): "int",
      ("rushing", "CAR"): "rush_att", ("rushing", "YDS"): "rush_yds", ("rushing", "TD"): "rush_td",
      ("receiving", "REC"): "rec", ("receiving", "YDS"): "rec_yds", ("receiving", "TD"): "rec_td", ("receiving", "TGTS"): "tgt",
      ("kickReturns", "TD"): "ret_td", ("puntReturns", "TD"): "ret_td", ("interceptions", "TD"): "ret_td", ("defensive", "TD"): "ret_td"}
BB = {"MIN": "min", "PTS": "pts", "REB": "reb", "AST": "ast", "STL": "stl", "BLK": "blk", "TO": "to"}
MLB_BAT = {"AB": "ab", "R": "r", "H": "h", "RBI": "rbi", "HR": "hr", "BB": "bb", "K": "so"}
MLB_PIT = {"IP": "ip", "H": "ha", "ER": "er", "BB": "bb_p", "K": "k"}

def players(box, lg):
    out = {}
    for t in box.get("players", []):
        abbr = t.get("team", {}).get("abbreviation")
        for grp in t.get("statistics", []):
            labels = grp.get("labels") or grp.get("names") or []
            gname, gtype = grp.get("name"), grp.get("type")
            for a in grp.get("athletes", []):
                nm = (a.get("athlete") or {}).get("displayName")
                if not nm:
                    continue
                p = out.setdefault(nm, {"team": abbr, "s": {}, "dnp": False})
                if a.get("didNotPlay"):
                    p["dnp"] = True
                vals = a.get("stats") or []
                for lab, v in zip(labels, vals):
                    if lg in ("nfl", "cfb"):
                        if (gname, lab) == ("passing", "C/ATT"):
                            c, _, at = str(v).partition("/")
                            p["s"]["pass_cmp"], p["s"]["pass_att"] = num(c), num(at)
                            continue
                        k = FB.get((gname, lab))
                        if k and num(v) is not None:
                            p["s"][k] = p["s"].get(k, 0) + num(v) if k == "ret_td" else num(v)
                    elif lg in ("nba", "cbb"):
                        if lab == "3PT":
                            p["s"]["tpm"] = made(v)
                        elif lab in BB and num(v) is not None:
                            p["s"][BB[lab]] = num(v)
                    else:
                        m = MLB_BAT if gtype == "batting" else MLB_PIT if gtype == "pitching" else {}
                        if lab in m and num(v) is not None:
                            p["s"][m[lab]] = num(v)
    for p in out.values():
        s = p["s"]
        if lg in ("nfl", "cfb"):
            s["rr_yds"] = s.get("rush_yds", 0) + s.get("rec_yds", 0)
            s["td"] = s.get("rush_td", 0) + s.get("rec_td", 0) + s.get("ret_td", 0)
        elif lg in ("nba", "cbb") and not p["dnp"]:
            s["pra"] = s.get("pts", 0) + s.get("reb", 0) + s.get("ast", 0)
        p["s"] = {k: (int(v) if float(v).is_integer() else v) for k, v in s.items() if v is not None}
    return out

def pull(lg):
    path = f"{OUT}/{lg}.json"
    old = json.load(open(path)) if os.path.exists(path) else {"games": []}
    games = {g["id"]: g for g in old.get("games", [])}
    today = dt.datetime.now(ET).date()
    for d in range(LOOKBACK, -1, -1):
        day = (today - dt.timedelta(days=d)).strftime("%Y%m%d")
        try:
            sb = get(f"https://site.api.espn.com/apis/site/v2/sports/{PATHS[lg]}/scoreboard?dates={day}{EXTRA.get(lg, '')}")
        except Exception as e:
            errors.append(f"{lg} {day}: {type(e).__name__}: {e}"); continue
        for ev in sb.get("events", []):
            try:
                c = ev["competitions"][0]
                side = {x["homeAway"]: x for x in c["competitors"]}
                st = ev.get("status", {}).get("type", {})
                k = dt.datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
                g = games.get(ev["id"], {})
                g.update({"id": ev["id"], "kickoff_utc": ev["date"], "day_et": k.astimezone(ET).date().isoformat(),
                          "state": st.get("state"), "final": bool(st.get("completed")) and st.get("state") == "post",
                          "detail": st.get("shortDetail"), "home": team(side["home"]), "away": team(side["away"])})
                games[ev["id"]] = g
            except Exception as e:
                errors.append(f"{lg} parse {ev.get('id')}: {type(e).__name__}: {e}")
        time.sleep(0.2)
    fetched = 0
    for g in games.values():
        if g.get("final") and "players" not in g:
            try:
                sm = get(f"https://site.api.espn.com/apis/site/v2/sports/{PATHS[lg]}/summary?event={g['id']}")
                g["players"] = players(sm.get("boxscore", {}), lg)
                fetched += 1
                time.sleep(0.2)
            except Exception as e:
                errors.append(f"{lg} box {g['id']}: {type(e).__name__}: {e}")
    cutoff = (today - dt.timedelta(days=KEEP_DAYS)).isoformat()
    keep = sorted((g for g in games.values() if g["day_et"] >= cutoff), key=lambda g: (g["kickoff_utc"], g["id"]))
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"league": lg, "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "games": keep},
                  f, separators=(",", ":"))
    print(f"ok   {lg}: {len(keep)} games ({sum(1 for g in keep if g.get('final'))} final), {fetched} box scores fetched")

if __name__ == "__main__":
    for lg in (sys.argv[1:] or PATHS):
        pull(lg)
    with open(f"{OUT}/_status.json", "w") as f:
        json.dump({"pulled_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "errors": errors[:50]}, f, indent=1)
