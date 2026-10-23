#!/usr/bin/env python3
"""Parse data/lines_hist/<league>_<season>.jsonl.gz (scripts/lines_history_pull.py) into one row per game:
opening and closing home spread, total, and their prices, from DraftKings (100) or ESPN BET (58).

  python scripts/lines_hist.py            -> data/lines_hist/<league>.csv for every league pulled
"""
import csv, glob, gzip, json, os, re

def num(x):
    try:
        return float(str(x).replace("+", "")) if x not in (None, "", "EVEN", "even") else (100.0 if str(x).lower() == "even" else None)
    except (TypeError, ValueError):
        return None

def am(block, key):
    """American price or point from an open/close block entry."""
    v = (block or {}).get(key) or {}
    return num(v.get("american") if v.get("american") not in (None, "") else v.get("alternateDisplayValue"))

def parse(rec):
    h, a = rec.get("homeTeamOdds") or {}, rec.get("awayTeamOdds") or {}
    row = {"game_id": rec["game_id"], "provider": rec.get("provider_id")}
    for when in ("open", "close"):
        hb, ab = h.get(when) or {}, a.get(when) or {}
        hs = am(hb, "pointSpread")
        if hs is None:
            asp = am(ab, "pointSpread")
            hs = -asp if asp is not None else None
        row[f"{when}_home_spread"] = hs
        row[f"{when}_home_price"] = am(hb, "spread")
        row[f"{when}_away_price"] = am(ab, "spread")
        row[f"{when}_home_ml"] = am(hb, "moneyLine")
        row[f"{when}_away_ml"] = am(ab, "moneyLine")
        tb = rec.get(when) or {}
        t = tb.get("total") or {}
        tv = num(t.get("alternateDisplayValue")) if t else None
        # the total's point sits in alternateDisplayValue as "o224.5"/"u224.5" or plain; fall back to overUnder for close
        m = re.search(r"(\d+(\.\d+)?)", str(t.get("alternateDisplayValue") or t.get("american") or "")) if t else None
        row[f"{when}_total"] = float(m.group(1)) if m and float(m.group(1)) > 20 else None
        row[f"{when}_over_price"] = am(tb, "over")
        row[f"{when}_under_price"] = am(tb, "under")
    # no fallback to the item's top-level spread/overUnder: on 2022-23 football items those aren't the game line
    return row

def main():
    for lg in sorted({os.path.basename(p).split("_")[0] for p in glob.glob("data/lines_hist/*.jsonl.gz")}):
        rows = []
        for p in sorted(glob.glob(f"data/lines_hist/{lg}_*.jsonl.gz")):
            season = re.search(r"_(\d{4})\.jsonl", p).group(1)
            for line in gzip.open(p, "rt"):
                r = json.loads(line)
                if r.get("missing"):
                    continue
                if lg in ("nfl", "cfb") and int(season) < 2024:
                    continue   # ESPN's 2022-23 football odds items don't carry usable game lines (checked vs nflverse/CFBD)
                x = parse(r); x["season"] = season
                rows.append(x)
        cols = ["season", "game_id", "provider"] + [f"{w}_{k}" for w in ("open", "close") for k in
                ("home_spread", "home_price", "away_price", "home_ml", "away_ml", "total", "over_price", "under_price")]
        with open(f"data/lines_hist/{lg}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        have = lambda k: sum(1 for r in rows if r.get(k) is not None)
        print(lg, len(rows), "games; open spread", have("open_home_spread"), "open total", have("open_total"), "close spread", have("close_home_spread"))

if __name__ == "__main__":
    main()
