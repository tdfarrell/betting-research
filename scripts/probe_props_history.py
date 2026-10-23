#!/usr/bin/env python3
"""One-off probe: does ESPN still serve DraftKings prop lines (and prices) for FINISHED games?
Writes data/probe/props_history_probe.json."""
import json, os, urllib.request, urllib.error
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
out = {"tries": []}
def get(url):
    rec = {"url": url}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
            d = json.load(r); rec.update(status=r.status)
    except urllib.error.HTTPError as e:
        rec.update(status=e.code); d = None
    except Exception as e:
        rec.update(status=None, error=str(e)); d = None
    out["tries"].append(rec); return d, rec
CASES = [("nba", "basketball", "401809243", "2025-26 opener"), ("nba", "basketball", "401810330", "2025-26 Jan"),
         ("nba", "basketball", "401704627", "2024-25 opener"), ("nfl", "football", "401772510", "2025 wk1"),
         ("nfl", "football", "401773000", "2025 midseason?")]
for lg, path, ev, lab in CASES:
    base = f"https://sports.core.api.espn.com/v2/sports/{path}/leagues/{lg}/events/{ev}/competitions/{ev}/odds"
    d, rec = get(base); rec["label"] = lab
    rec["providers"] = [((it.get("provider") or {}).get("id"), (it.get("provider") or {}).get("name")) for it in (d or {}).get("items", [])]
    for pid in [p[0] for p in rec["providers"]][:3] or ["100"]:
        d2, r2 = get(f"{base}/{pid}/propBets?lang=en&region=us&limit=1000")
        r2["label"] = lab
        if d2:
            items = d2.get("items", [])
            r2["count"] = d2.get("count"); r2["pageCount"] = d2.get("pageCount")
            types = {}
            for it in items:
                t = (it.get("type") or {}).get("name"); types[t] = types.get(t, 0) + 1
            r2["types"] = dict(list(types.items())[:14])
            r2["sample"] = [{k: it.get(k) for k in ("type", "odds", "current", "open", "lastUpdated")} for it in items[:2]]
os.makedirs("data/probe", exist_ok=True)
json.dump(out, open("data/probe/props_history_probe.json", "w"), indent=1, default=str)
for t in out["tries"]:
    print(t.get("status"), t.get("label"), t["url"][-60:], t.get("count", ""), t.get("providers", ""))
