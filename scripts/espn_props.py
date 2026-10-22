#!/usr/bin/env python3
"""DraftKings NFL player prop over/under lines via ESPN's core API (provider 100 = Draft Kings).

ESPN publishes DK's prop lines (current + open), but not the prices, for each upcoming game:
  sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/<id>/competitions/<id>/odds/100/propBets
Runs with every data pull (lines move), separately from the twice-daily prop cards.

Writes data/props/dk_lines_nfl.json:
  {"pulled_utc", "source", "games": {espn_event_id: n_lines},
   "lines": {gsis_id: {market: {"line", "open", "updated", "game_id"}}},
   "unmatched": [espn athlete ids we could not map to a gsis id]}
Markets use the prop cards' keys: pass_yds, pass_td, rush_yds, rec_yds, rec, rr_yds (anytime TD has no line).
Athletes are matched to nflverse ids by roster_current.csv's espn_id, then by name + team from ESPN's team rosters.
"""
import csv, datetime as dt, json, os, re, sys, urllib.request

CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
PROVIDER = "100"   # Draft Kings
OUT = "data/props/dk_lines_nfl.json"
TYPES = {"total passing yards": "pass_yds", "total passing touchdowns": "pass_td", "total rushing yards": "rush_yds",
         "total receiving yards": "rec_yds", "total receptions": "rec", "total rushing plus receiving yards": "rr_yds"}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
errors = []

def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def norm(s):
    s = re.sub(r"[^a-z ]", "", (s or "").lower().replace("-", " "))
    return " ".join(w for w in s.split() if w not in ("jr", "sr", "ii", "iii", "iv", "v"))

def market_of(type_name):
    t = re.sub(r"\s*\(incl\. overtime\)\s*$", "", (type_name or "").strip().lower())
    return TYPES.get(t)

def athlete_id(ref):
    m = re.search(r"/athletes/(\d+)", ref or "")
    return m.group(1) if m else None

def prop_items(ev):
    items, page = [], 1
    while True:
        d = fetch(f"{CORE}/events/{ev}/competitions/{ev}/odds/{PROVIDER}/propBets?lang=en&region=us&limit=1000&page={page}")
        items += d.get("items", [])
        if page >= int(d.get("pageCount", 1) or 1):
            return items
        page += 1

def main():
    board = json.load(open("data/odds/latest_nfl.json"))
    games = [g for g in board["games"] if g.get("status") == "pre"]
    roster = list(csv.DictReader(open("data/nfl/roster_current.csv"))) if os.path.exists("data/nfl/roster_current.csv") else []
    by_espn = {r["espn_id"].split(".")[0]: r["player_id"] for r in roster if r.get("espn_id")}
    by_name = {(norm(r["name"]), r["team"]): r["player_id"] for r in roster}
    espn_names = {}   # espn athlete id -> (name, team abbr), filled lazily from ESPN team rosters
    fetched_teams = set()

    def name_lookup(aid, teams):
        for t in teams:
            if t in fetched_teams:
                continue
            fetched_teams.add(t)
            try:
                d = fetch(f"{SITE}/teams/{t.lower()}/roster")
                for grp in d.get("athletes", []):
                    for a in grp.get("items", []):
                        espn_names[str(a.get("id"))] = (a.get("fullName") or a.get("displayName"), t)
            except Exception as e:
                errors.append(f"roster {t}: {type(e).__name__}: {e}")
            if aid in espn_names:
                break
        nm = espn_names.get(aid)
        if not nm:
            return None
        for t in teams:   # ESPN and nflverse abbreviations mostly agree; try both teams in the game
            pid = by_name.get((norm(nm[0]), t))
            if pid:
                return pid
        return None

    old = json.load(open(OUT)) if os.path.exists(OUT) else {}
    lines, per_game, unmatched = {}, {}, set()
    for g in games:
        ev = str(g["game_id"])
        try:
            items = prop_items(ev)
        except Exception as e:
            errors.append(f"{ev}: {type(e).__name__}: {e}")
            continue
        n = 0
        for it in items:
            mk = market_of((it.get("type") or {}).get("name"))
            aid = athlete_id((it.get("athlete") or {}).get("$ref"))
            cur = ((it.get("current") or {}).get("target") or {}).get("value")
            if not mk or not aid or cur is None:
                continue
            pid = by_espn.get(aid) or name_lookup(aid, [g["home"], g["away"]])
            if not pid:
                unmatched.add(aid); continue
            opn = ((it.get("open") or {}).get("target") or {}).get("value")
            rec = lines.setdefault(pid, {})
            if mk not in rec:
                n += 1
            rec[mk] = {"line": float(cur), "open": float(opn) if opn is not None else None, "updated": it.get("lastUpdated"), "game_id": ev,
                       "kickoff_utc": g.get("kickoff_utc")}
        per_game[ev] = n
        print(f"{g['away']} @ {g['home']} ({ev}): {n} DK player lines")
    # keep lines for games that already kicked off (ESPN drops them), so the page still shows the closing number
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")
    pre_ids = {str(g["game_id"]) for g in games}
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=4)).strftime("%Y-%m-%dT%H:%MZ")
    for pid, mks in (old.get("lines") or {}).items():
        for mk, v in mks.items():
            if v.get("game_id") not in pre_ids and (v.get("kickoff_utc") or "") >= cutoff:
                lines.setdefault(pid, {}).setdefault(mk, {**v, "closed": True})
    doc = {"pulled_utc": now, "source": "ESPN core API propBets, provider 100 (Draft Kings): line only, no prices",
           "games": per_game, "lines": lines, "unmatched": sorted(unmatched), "errors": errors[:20]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"DK prop lines: {sum(len(v) for v in lines.values())} lines for {len(lines)} players; {len(unmatched)} ESPN athletes unmatched; {len(errors)} errors")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:   # never break the data job
        print(f"espn_props failed: {type(e).__name__}: {e}"); sys.exit(0)
