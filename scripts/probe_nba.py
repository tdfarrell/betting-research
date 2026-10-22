#!/usr/bin/env python3
"""One-off probe: which free NBA sources work from GitHub Actions? -> data/probe/nba_probe.json
  A) ESPN NBA scoreboard (DK game lines), next few days + last few days
  B) ESPN core propBets, provider 100 (Draft Kings), for one upcoming game
  C) ESPN summary box score for one finished regular-season game (player minutes, starters)
  D) sportsdataverse / hoopR release assets (season player box scores, like nflverse)
  E) stats.nba.com leaguegamelog (often blocked from cloud IPs)
  F) ESPN athlete game log endpoint
"""
import json, os, urllib.request, urllib.error, datetime as dt

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
      "Accept": "application/json"}
out = {"run_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "tries": []}

def get(url, headers=None, keep=600, timeout=40):
    rec = {"url": url}
    try:
        req = urllib.request.Request(url, headers={**UA, **(headers or {})})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(); rec.update(status=r.status, bytes=len(body))
            try:
                data = json.loads(body)
            except Exception:
                data = None; rec["sample"] = body[:keep].decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        rec.update(status=e.code, error=e.read()[:300].decode("utf-8", "replace")); data = None
    except Exception as e:
        rec.update(status=None, error=f"{type(e).__name__}: {e}"); data = None
    out["tries"].append(rec)
    return data, rec

now = dt.datetime.now(dt.timezone.utc)
SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}"
up, done = [], []
for i in range(-4, 4):
    day = (now + dt.timedelta(days=i)).strftime("%Y%m%d")
    d, rec = get(SB.format(d=day))
    evs = (d or {}).get("events", [])
    rec["n_events"] = len(evs)
    if evs:
        rec["season"] = evs[0].get("season")
        c = evs[0]["competitions"][0]
        rec["odds_sample"] = (c.get("odds") or [None])[0]
    for e in evs:
        st = e["competitions"][0]["status"]["type"]["state"]
        (up if st == "pre" else done if st == "post" else []).append(e)
out["upcoming"] = len(up); out["finished"] = len(done)

if up:
    ev = up[0]["id"]
    base = f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{ev}/competitions/{ev}/odds"
    d, rec = get(base)
    rec["providers"] = [(it.get("provider") or {}).get("name") for it in (d or {}).get("items", [])]
    d, rec = get(f"{base}/100/propBets?lang=en&region=us&limit=1000")
    if d:
        items = d.get("items", [])
        rec["count"] = d.get("count"); rec["pageCount"] = d.get("pageCount")
        types = {}
        for it in items:
            t = (it.get("type") or {}).get("name"); types[t] = types.get(t, 0) + 1
        rec["types"] = types
        rec["sample_items"] = items[:2]
if done:
    reg = [e for e in done if (e.get("season") or {}).get("type") == 2] or done
    ev = reg[-1]["id"]
    d, rec = get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={ev}")
    if d:
        bs = (d.get("boxscore") or {}).get("players", [])
        if bs:
            st = bs[0]["statistics"][0]
            rec["box_keys"] = st.get("keys") or st.get("labels")
            a = st["athletes"][0]
            rec["athlete_sample"] = {"athlete": {k: a["athlete"].get(k) for k in ("id", "displayName", "position")}, "starter": a.get("starter"), "stats": a.get("stats")}
        rec["has_injuries"] = bool(d.get("injuries"))
        rec["season"] = (d.get("header") or {}).get("season")
        aid = bs[0]["statistics"][0]["athletes"][0]["athlete"]["id"] if bs else None
        if aid:
            d2, r2 = get(f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{aid}/gamelog?season=2026")
            if d2:
                r2["keys"] = list(d2.keys()); r2["labels"] = d2.get("labels")

for repo in ("sportsdataverse/hoopR-nba-data", "sportsdataverse/sportsdataverse-data"):
    d, rec = get(f"https://api.github.com/repos/{repo}/releases?per_page=100", headers={"Accept": "application/vnd.github+json"})
    if isinstance(d, list):
        rec["releases"] = [{"tag": r["tag_name"], "assets": [a["name"] for a in r["assets"] if any(y in a["name"] for y in ("2025", "2026", "2027"))][:8],
                            "n": len(r["assets"])} for r in d][:40]

get("https://stats.nba.com/stats/leaguegamelog?Counter=1000&DateFrom=&DateTo=&Direction=DESC&LeagueID=00&PlayerOrTeam=T&Season=2025-26&SeasonType=Regular%20Season&Sorter=DATE",
    headers={"Referer": "https://www.nba.com/", "Origin": "https://www.nba.com", "x-nba-stats-origin": "stats", "x-nba-stats-token": "true"}, timeout=25)
get("https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json")

os.makedirs("data/probe", exist_ok=True)
json.dump(out, open("data/probe/nba_probe.json", "w"), indent=1, default=str)
for t in out["tries"]:
    print(t.get("status"), t.get("bytes"), t["url"][:120], t.get("n_events", ""), t.get("count", ""), (t.get("error") or "")[:100])
