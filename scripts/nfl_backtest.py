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
            "epa_cal", "sr_cal", "pts_cal", "last_cal", "elo_cal", "mkt_cal", "blend", "epa_tot_cal", "pts_tot_cal"]
    with open("data/nfl/backtest_games.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    res_vs_close = [r["result"] - r["close"] for r in rows]
    summ = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
            "seasons": [first, last], "games": len(rows), "line": "nflverse consensus closing line",
            "rmse": {"close": round(float(np.sqrt(np.mean([(r["result"] - r["close"]) ** 2 for r in rows]))), 2)},
            "ats_vs_close": {}, "by_season": {}, "totals_vs_close": {}}
    for m in ("epa_cal", "sr_cal", "pts_cal", "blend", "last_cal", "elo_cal", "mkt_cal"):
        summ["rmse"][m] = round(float(np.sqrt(np.mean([(r["result"] - r[m]) ** 2 for r in rows]))), 2)
    for t in (1, 2, 3):
        e = [(r["blend"] - r["close"]) if abs(r["blend"] - r["close"]) >= t else None for r in rows]
        summ["ats_vs_close"][f"blend_edge_{t}+"] = rec(e, res_vs_close)
        e = [(r["blend"] - r["close"]) if abs(r["blend"] - r["close"]) >= t and all(
            (r[m] - r["close"]) * (r["blend"] - r["close"]) > 0 and abs(r[m] - r["close"]) >= 1 for m in ("epa_cal", "sr_cal", "pts_cal")) else None for r in rows]
        summ["ats_vs_close"][f"blend_{t}+_all3_agree"] = rec(e, res_vs_close)
        tr = [r["total"] - r["close_total"] if r["close_total"] else 0 for r in rows]
        e = [(r["epa_tot_cal"] - r["close_total"]) if r["close_total"] and abs(r["epa_tot_cal"] - r["close_total"]) >= t + 1 and
             abs(r["pts_tot_cal"] - r["close_total"]) >= t + 1 and (r["epa_tot_cal"] - r["close_total"]) * (r["pts_tot_cal"] - r["close_total"]) > 0 else None for r in rows]
        summ["totals_vs_close"][f"both_{t + 1}+"] = rec(e, tr)
    for s in range(first, last + 1):
        sub = [(r["blend"] - r["close"] if abs(r["blend"] - r["close"]) >= 2 else None, r["result"] - r["close"]) for r in rows if r["season"] == s]
        summ["by_season"][str(s)] = rec([a for a, _ in sub], [b for _, b in sub])
    with open("data/nfl/backtest_summary.json", "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(a[0] if a else 2022, a[1] if len(a) > 1 else 2025)
