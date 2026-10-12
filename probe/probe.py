import json, urllib.request, os
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
def get(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=60) as r: return json.load(r)
os.makedirs("probe/out", exist_ok=True)
SB = {"nfl": "football/nfl", "cfb": "football/college-football", "nba": "basketball/nba", "mlb": "baseball/mlb", "cbb": "basketball/mens-college-basketball"}
DAYS = {"nfl": "20261011", "cfb": "20261010", "nba": "20261010", "mlb": "20261010", "cbb": "20251210"}
for lg, path in SB.items():
    try:
        sb = get(f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard?dates={DAYS[lg]}" + ("&groups=80&limit=300" if lg == "cfb" else ""))
        evs = sb.get("events", [])
        fin = [e for e in evs if e.get("status", {}).get("type", {}).get("completed")]
        ev = (fin or evs or [None])[0]
        json.dump({"n_events": len(evs), "n_final": len(fin), "event0": ev}, open(f"probe/out/{lg}_scoreboard.json", "w"), indent=1)
        if ev:
            sm = get(f"https://site.api.espn.com/apis/site/v2/sports/{path}/summary?event={ev['id']}")
            bx = sm.get("boxscore", {})
            json.dump({"top_keys": list(sm.keys()), "boxscore": bx}, open(f"probe/out/{lg}_summary_boxscore.json", "w"), indent=1)
        print("ok", lg, len(evs), len(fin))
    except Exception as e:
        print("FAIL", lg, e)
