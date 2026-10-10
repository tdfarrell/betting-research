#!/usr/bin/env python3
"""Backtest the NFL screens against closing lines (nflverse consensus close; DraftKings history isn't archived).

  python scripts/nfl_backtest.py 2022 2025   -> data/nfl/backtest_games.csv + data/nfl/backtest_summary.json

Every prediction uses only games played before that week. Calibration to the market also uses only
earlier weeks, exactly as the live screener does.
"""
import csv, datetime as dt, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import nfl_model as M

def rec(edge, outcome):
    m = [(e, o) for e, o in zip(edge, outcome) if e is not None and o != 0]
    w = sum(1 for e, o in m if (e > 0) == (o > 0))
    return {"w": w, "l": len(m) - w, "pct": round(w / len(m), 3) if m else None}

def main(first, last):
    rows, _ = M.predict(first - 1, last)
    M.calibrate(rows)
    rows = [r for r in rows if first <= r["season"] <= last and r["result"] is not None and r["close"] is not None and r["blend"] is not None]
    cols = ["season", "week", "game_id", "home", "away", "close", "result", "close_total", "total", "qb_adj",
            "epa_cal", "sr_cal", "pts_cal", "last_cal", "elo_cal", "mkt_cal", "blend", "side_sd", "tot_sd", "epa_tot_cal", "pts_tot_cal"]
    with open("data/nfl/backtest_games.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    res_vs_close = [r["result"] - r["close"] for r in rows]
    summ = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
            "seasons": [first, last], "games": len(rows), "line": "nflverse consensus closing line",
            "rmse": {"close": round(float(np.sqrt(np.mean([(r["result"] - r["close"]) ** 2 for r in rows]))), 2)},
            "by_season": {}}
    for m in ("epa_cal", "sr_cal", "pts_cal", "blend", "last_cal", "elo_cal", "mkt_cal"):
        summ["rmse"][m] = round(float(np.sqrt(np.mean([(r["result"] - r[m]) ** 2 for r in rows]))), 2)
    # the live tiers (see regrade_nfl.py), applied to closing lines
    sys.path.insert(0, os.path.dirname(__file__))
    from regrade_nfl import grade_side, grade_total
    tiers = {"side": {}, "total": {}}
    fav_n = fav_t = over_n = over_t = 0
    for r in rows:
        e = r["blend"] - r["close"]; es = [r[m] - r["close"] for m in ("epa_cal", "sr_cal", "pts_cal")]
        g = grade_side(e, es, r.get("side_sd"))
        if g != "PASS":
            o = r["result"] - r["close"]
            if o != 0:
                d = tiers["side"].setdefault(g, {"w": 0, "l": 0, "fav": 0, "n": 0})
                d["w" if (e > 0) == (o > 0) else "l"] += 1
            d = tiers["side"].setdefault(g, {"w": 0, "l": 0, "fav": 0, "n": 0}); d["n"] += 1
            fav = (e > 0) == (r["close"] > 0); d["fav"] += fav
        if r["close_total"]:
            t1, t2 = r["epa_tot_cal"] - r["close_total"], r["pts_tot_cal"] - r["close_total"]
            g = grade_total(t1, t2)
            if g != "PASS":
                t = (t1 + t2) / 2; o = r["total"] - r["close_total"]
                d = tiers["total"].setdefault(g, {"w": 0, "l": 0, "over": 0, "n": 0}); d["n"] += 1; d["over"] += t > 0
                if o != 0:
                    d["w" if (t > 0) == (o > 0) else "l"] += 1
    for mk in tiers.values():
        for d in mk.values():
            d["pct"] = round(d["w"] / max(d["w"] + d["l"], 1), 3)
            d["per_week"] = round(d["n"] / (18 * (last - first + 1)), 1)
            if "fav" in d: d["fav_share"] = round(d.pop("fav") / d["n"], 2)
            if "over" in d: d["over_share"] = round(d.pop("over") / d["n"], 2)
    summ["tiers_vs_close"] = tiers
    gaps = {}
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 99)):
        sub = [(r["blend"] - r["close"], r["result"] - r["close"]) for r in rows if lo <= abs(r["blend"] - r["close"]) < hi]
        gaps[f"{lo}-{hi if hi < 99 else '+'}"] = rec([a for a, _ in sub], [b for _, b in sub])
    summ["side_by_gap"] = gaps
    for s in range(first, last + 1):
        sub = [(r["blend"] - r["close"] if 2 <= abs(r["blend"] - r["close"]) < 4 else None, r["result"] - r["close"]) for r in rows if r["season"] == s]
        summ["by_season"][str(s)] = rec([a for a, _ in sub], [b for _, b in sub])
    with open("data/nfl/backtest_summary.json", "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(a[0] if a else 2022, a[1] if len(a) > 1 else 2025)
