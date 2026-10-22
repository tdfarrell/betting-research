#!/usr/bin/env python3
"""DraftKings NBA player prop lines and prices via ESPN's core API (provider 100 = Draft Kings).

Unlike the NFL feed, ESPN includes DraftKings' prices for NBA props: each line comes as two items, which we read
as over then under (the order ESPN lists them; check against the DK app).

Writes data/props/dk_lines_nba.json:
  {"pulled_utc", "source", "games": {event_id: n_lines},
   "lines": {espn_athlete_id: {market: {"line", "open", "over", "under", "over_open", "under_open", "updated",
                                         "game_id", "kickoff_utc"}}}}
Markets: pts reb ast tpm pra pr pa ra stl blk. Player ids are ESPN athlete ids (the same ids the NBA game logs use).
"""
import datetime as dt, json, os, re, sys, urllib.request

CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
OUT = "data/props/dk_lines_nba.json"
TYPES = {"total points": "pts", "total rebounds": "reb", "total assists": "ast", "total 3-point field goals": "tpm",
         "total points, rebounds, and assists": "pra", "total points and rebounds": "pr", "total points and assists": "pa",
         "total assists and rebounds": "ra", "total steals": "stl", "total blocks": "blk"}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
errors = []

def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def prop_items(ev):
    items, page = [], 1
    while True:
        d = fetch(f"{CORE}/events/{ev}/competitions/{ev}/odds/100/propBets?lang=en&region=us&limit=1000&page={page}")
        items += d.get("items", [])
        if page >= int(d.get("pageCount", 1) or 1):
            return items
        page += 1

def main():
    board = json.load(open("data/odds/latest_nba.json"))
    games = [g for g in board["games"] if g.get("status") == "pre"]
    old = json.load(open(OUT)) if os.path.exists(OUT) else {}
    lines, per_game = {}, {}
    for g in games:
        ev = str(g["game_id"])
        try:
            items = prop_items(ev)
        except Exception as e:
            errors.append(f"{ev}: {type(e).__name__}: {e}"); continue
        seen = {}
        for it in items:
            mk = TYPES.get(((it.get("type") or {}).get("name") or "").strip().lower())
            m = re.search(r"/athletes/(\d+)", (it.get("athlete") or {}).get("$ref", ""))
            cur = ((it.get("current") or {}).get("target") or {}).get("value")
            if not mk or not m or cur is None:
                continue
            aid = m.group(1)
            am = ((it.get("odds") or {}).get("american") or {})
            k = (aid, mk)
            n = seen.get(k, 0); seen[k] = n + 1
            rec = lines.setdefault(aid, {}).setdefault(mk, {"line": float(cur), "game_id": ev, "kickoff_utc": g.get("kickoff_utc"),
                                                          "updated": it.get("lastUpdated")})
            opn = ((it.get("open") or {}).get("target") or {}).get("value")
            if opn is not None:
                rec["open"] = float(opn)
            side = "over" if n == 0 else "under" if n == 1 else None
            if side:
                rec[side] = am.get("value"); rec[side + "_open"] = am.get("open")
        per_game[ev] = sum(1 for (a, mk) in seen)
        print(f"{g['away']} @ {g['home']} ({ev}): {per_game[ev]} DK player lines")
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = (now - dt.timedelta(days=3)).strftime("%Y-%m-%dT%H:%MZ")
    pre_ids = {str(g["game_id"]) for g in games}
    for pid, mks in (old.get("lines") or {}).items():   # keep closing lines of started games for a few days
        for mk, v in mks.items():
            if v.get("game_id") not in pre_ids and (v.get("kickoff_utc") or "") >= cutoff:
                lines.setdefault(pid, {}).setdefault(mk, {**v, "closed": True})
    doc = {"pulled_utc": now.isoformat(timespec="minutes"), "source": "ESPN core API propBets, provider 100 (Draft Kings); prices read as over, under",
           "games": per_game, "lines": lines, "errors": errors[:20]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, "w"), separators=(",", ":"))
    print(f"DK NBA prop lines: {sum(len(v) for v in lines.values())} lines for {len(lines)} players; {len(errors)} errors")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"espn_props_nba failed: {type(e).__name__}: {e}"); sys.exit(0)
