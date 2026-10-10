#!/usr/bin/env python3
"""One-time backfill of past NFL DraftKings lines (open + close) from ESPN, for an honest
early-line / CLV backtest. Run from GitHub Actions (Claude's workspace can't reach ESPN):
  Actions -> "Backfill NFL DraftKings history" -> Run workflow

  python scripts/espn_backfill_nfl.py 2023 2025

Writes:
  data/nfl/dk_history.csv          one row per game: provider, open/close spread + total, moneylines, score
  data/nfl/dk_history_raw.jsonl    the raw odds entry per game (so parsing can be fixed without re-pulling)
ESPN's odds provider changed over time (ESPN BET 2023-25, DraftKings after); the provider is recorded and
only DraftKings rows count as DK history. Non-DK rows are kept as a labeled proxy.
"""
import csv, json, os, sys, time, urllib.request

SB = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={d}"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{e}/competitions/{e}/odds"

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    for k in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if k == 2:
                raise
            time.sleep(2 + 3 * k)

def g(d, *keys):
    for k in keys:
        if isinstance(d, list):
            d = d[k] if isinstance(k, int) and k < len(d) else None
        elif isinstance(d, dict):
            d = d.get(k)
        else:
            return None
    return d

def parse_sb(o):
    """Scoreboard odds entry (same shape the live pipeline parses)."""
    row = {"provider": g(o, "provider", "name")}
    for s in ("home", "away"):
        row[f"{s}_spread_open"] = g(o, "pointSpread", s, "open", "line")
        row[f"{s}_spread"] = g(o, "pointSpread", s, "close", "line")
        row[f"{s}_ml_open"] = g(o, "moneyline", s, "open", "odds")
        row[f"{s}_ml"] = g(o, "moneyline", s, "close", "odds")
    row["total_open"] = g(o, "total", "over", "open", "line")
    row["total"] = g(o, "total", "over", "close", "line") or o.get("overUnder")
    if row["home_spread"] is None and o.get("spread") is not None:
        row["home_spread"] = o.get("spread")
    return row

def parse_core(o):
    """sports.core odds item: homeTeamOdds/awayTeamOdds with open/close blocks."""
    row = {"provider": g(o, "provider", "name")}
    for s, key in (("home", "homeTeamOdds"), ("away", "awayTeamOdds")):
        t = o.get(key) or {}
        row[f"{s}_spread_open"] = g(t, "open", "pointSpread", "alternateDisplayValue") or g(t, "open", "spread", "alternateDisplayValue")
        row[f"{s}_spread"] = g(t, "close", "pointSpread", "alternateDisplayValue") or g(t, "close", "spread", "alternateDisplayValue")
        row[f"{s}_ml_open"] = g(t, "open", "moneyLine", "american")
        row[f"{s}_ml"] = g(t, "close", "moneyLine", "american") or t.get("moneyLine")
    row["total_open"] = g(o, "open", "total", "alternateDisplayValue")
    row["total"] = g(o, "close", "total", "alternateDisplayValue") or o.get("overUnder")
    if row["home_spread"] is None and o.get("spread") is not None:
        row["home_spread"] = o.get("spread")
    return row

def main(first, last):
    sched = [r for r in csv.DictReader(open("data/nfl/schedule.csv")) if first <= int(r["season"]) <= last and r["home_score"]]
    days = sorted({r["gameday"].replace("-", "") for r in sched})
    want = {r["espn"]: r for r in sched}
    out, raw, errors = [], open("data/nfl/dk_history_raw.jsonl", "w"), []
    for d in days:
        try:
            evs = fetch(SB.format(d=d)).get("events", [])
        except Exception as e:
            errors.append(f"{d}: {e}"); continue
        for ev in evs:
            if ev["id"] not in want:
                continue
            comp = ev["competitions"][0]
            teams = {c["homeAway"]: c for c in comp["competitors"]}
            base = {"espn": ev["id"], "game_id": want[ev["id"]]["game_id"], "season": want[ev["id"]]["season"],
                    "week": want[ev["id"]]["week"], "kickoff_utc": ev["date"],
                    "away": g(teams["away"], "team", "abbreviation"), "home": g(teams["home"], "team", "abbreviation"),
                    "away_score": teams["away"].get("score"), "home_score": teams["home"].get("score")}
            odds = comp.get("odds", [])
            dk = next((o for o in odds if "draft" in (g(o, "provider", "name") or "").lower()), None)
            row, src = (parse_sb(dk), "scoreboard") if dk else (None, None)
            if row is None or row.get("home_spread_open") is None:
                try:  # the core odds feed lists every provider, including DraftKings, with open/close
                    items = fetch(CORE.format(e=ev["id"])).get("items", [])
                    cdk = next((o for o in items if "draft" in (g(o, "provider", "name") or "").lower()), None)
                    if cdk:
                        row, src, dk = parse_core(cdk), "core", cdk
                    time.sleep(0.2)
                except Exception as e:
                    errors.append(f"core {ev['id']}: {e}")
            if row is None and odds:
                row, src, dk = parse_sb(odds[0]), "scoreboard-other", odds[0]
            if row is None:
                continue
            out.append({**base, **row, "source": src})
            raw.write(json.dumps({"espn": ev["id"], "source": src, "odds": dk}, separators=(",", ":")) + "\n")
        time.sleep(0.3)
    raw.close()
    cols = ["espn", "game_id", "season", "week", "kickoff_utc", "away", "home", "provider", "source", "home_spread_open", "home_spread",
            "away_spread_open", "away_spread", "total_open", "total", "home_ml_open", "home_ml", "away_ml_open", "away_ml",
            "away_score", "home_score"]
    with open("data/nfl/dk_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(out)
    dk_n = sum(1 for r in out if "draft" in (r.get("provider") or "").lower())
    print(f"{len(out)} games ({dk_n} DraftKings, {sum(1 for r in out if r.get('home_spread_open') not in (None, ''))} with an opening line); {len(errors)} errors")
    for e in errors[:20]:
        print("  ", e)

if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(a[0] if a else 2023, a[1] if len(a) > 1 else 2025)
