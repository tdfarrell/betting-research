#!/usr/bin/env python3
"""One-off probe: can we get DraftKings NFL player prop O/U lines for free?

Tries
  A) ESPN core API odds providers + propBets for one upcoming NFL game
  B) DraftKings' public sportsbook JSON (the endpoints its own website calls)
Writes data/probe/dk_props_probe.json (status codes, sizes, a few sample markets) so the
result can be read from the repo. Nothing here feeds the Screener.
"""
import json, os, urllib.request, urllib.error, datetime as dt

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
      "Accept": "application/json"}
out = {"run_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "tries": []}

def get(url, keep=1500):
    rec = {"url": url}
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=40) as r:
            body = r.read()
            rec.update(status=r.status, bytes=len(body))
            try:
                data = json.loads(body)
            except Exception:
                data = None
                rec["sample"] = body[:keep].decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        rec.update(status=e.code, error=e.read()[:400].decode("utf-8", "replace")); data = None
    except Exception as e:
        rec.update(status=None, error=f"{type(e).__name__}: {e}"); data = None
    out["tries"].append(rec)
    return data, rec

board = json.load(open("data/odds/latest_nfl.json"))
pre = [g for g in board["games"] if g.get("status") == "pre"]
ev = pre[0]["game_id"] if pre else None
out["event"] = ev

# ---- A) ESPN core API
if ev:
    base = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{ev}/competitions/{ev}/odds"
    d, rec = get(base)
    provs = []
    for it in (d or {}).get("items", []):
        p = it.get("provider", {})
        provs.append({"id": p.get("id"), "name": p.get("name")})
    rec["providers"] = provs
    for p in provs or [{"id": "40"}, {"id": "58"}, {"id": "100"}]:
        d, rec = get(f"{base}/{p['id']}/propBets?lang=en&region=us&limit=1000")
        if d:
            items = d.get("items", [])
            rec["count"] = d.get("count", len(items))
            rec["sample_items"] = items[:4]
            types = {}
            for it in items:
                t = (it.get("type") or {}).get("name") or (it.get("type") or {}).get("id")
                types[str(t)] = types.get(str(t), 0) + 1
            rec["types"] = types

# ---- B) DraftKings sportsbook JSON
for state in ("dkusnj", "dkusny", "dkuspa"):
    d, rec = get(f"https://sportsbook-nash.draftkings.com/api/sportscontent/{state}/v1/leagues/88808")
    if d:
        cats = d.get("categories") or []
        rec["categories"] = [{"id": c.get("id"), "name": c.get("name")} for c in cats][:60]
        rec["top_keys"] = list(d.keys())
        for c in cats:
            nm = (c.get("name") or "").lower()
            if any(k in nm for k in ("pass", "rush", "rec", "td", "player")):
                d2, r2 = get(f"https://sportsbook-nash.draftkings.com/api/sportscontent/{state}/v1/leagues/88808/categories/{c['id']}")
                if d2:
                    r2["top_keys"] = list(d2.keys())
                    r2["subcategories"] = [{"id": s.get("id"), "name": s.get("name")} for s in d2.get("subcategories", [])][:40]
                    r2["n_markets"] = len(d2.get("markets", []))
                    r2["n_selections"] = len(d2.get("selections", []))
                    r2["sample_markets"] = d2.get("markets", [])[:3]
                    r2["sample_selections"] = d2.get("selections", [])[:6]
        break
get("https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/88808?format=json")

os.makedirs("data/probe", exist_ok=True)
with open("data/probe/dk_props_probe.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
for t in out["tries"]:
    print(t.get("status"), t.get("bytes"), t["url"], t.get("count", ""), t.get("n_selections", ""), (t.get("error") or "")[:120])
