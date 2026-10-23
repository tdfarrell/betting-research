#!/usr/bin/env python3
"""Confidence for game-line picks: how often picks like this one won in the backtest.

For each sport and market, picks are grouped by how far the model sat from the closing line (the same gap
bands the PLAY / LEAN tags use), and each band's win rate against the close is shrunk toward 50% by
SHRINK games, so a small band can't claim a big number. Live picks look up their band
(scripts/picks.py -> conf_for) and carry `conf` (the shrunk win rate) and `conf_n` (games behind it).

  python scripts/confidence.py      -> data/grades/confidence.json

Sources:  NFL  data/nfl/backtest_games.csv (2022-25, nflverse closing lines)
          CFB  scripts/cfb_backtest.py rows (Elo screen, 2022-25; totals not backtested)
          NBA  data/nba/backtest_games.csv (2025-26 + this season, DK/ESPN closing lines)
          CBB  data/cbb/backtest_games.csv (2025-26)
"""
import csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
OUT = "data/grades/confidence.json"
SHRINK = 60.0

# band edges, in units of the sport's usual gap (z) or in points (pts), matching each sport's tags
BANDS = {
    ("nfl", "side"): ("z", [0, 0.3, 0.6, 1.0]),
    ("nfl", "total"): ("pts", [0, 1.5, 2.0, 4.5]),
    ("nba", "side"): ("z", [0, 0.3, 0.6, 1.0]),
    ("nba", "total"): ("z", [0, 0.3, 0.6, 1.0]),
    ("cbb", "side"): ("z", [0, 0.3, 0.6, 1.0]),
    ("cbb", "total"): ("z", [0, 0.3, 0.6, 1.0]),
    ("cfb", "side"): ("z", [0, 0.6, 1.0, 2.0]),
}

def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def table(xs, wins, edges):
    xs, wins = np.asarray(xs, float), np.asarray(wins, float)
    out = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else 1e9
        m = (xs >= lo) & (xs < hi)
        n, w = int(m.sum()), float(wins[m].sum())
        out.append({"lo": lo, "hi": None if hi >= 1e9 else hi, "n": n, "w": int(w), "raw": round(w / n, 3) if n else None,
                    "conf": round((w + 0.5 * SHRINK) / (n + SHRINK), 3)})
    return out

def nfl_rows():
    side, tot = ([], []), ([], [])
    for r in csv.DictReader(open("data/nfl/backtest_games.csv")):
        c, res, b, sd = num(r["close"]), num(r["result"]), num(r["blend"]), num(r["side_sd"])
        if None not in (c, res, b, sd) and sd > 0 and res != c:
            e = b - c
            side[0].append(abs(e) / sd); side[1].append(int((res - c) * e > 0))
        ct, t, et, pt = num(r["close_total"]), num(r["total"]), num(r["epa_tot_cal"]), num(r["pts_tot_cal"])
        if None not in (ct, t, et, pt) and t != ct:
            e = (et + pt) / 2 - ct
            tot[0].append(abs(e)); tot[1].append(int((t - ct) * e > 0))
    return side, tot

def bb_rows(path):
    side, tot = ([], []), ([], [])
    for r in csv.DictReader(open(path)):
        e, sd, res = num(r["side_edge"]), num(r["side_sd"]), r["side_res"]
        if e is not None and sd and res in ("0", "1"):
            side[0].append(abs(e) / sd); side[1].append(int(res))
        e, sd, res = num(r["tot_edge"]), num(r["tot_sd"]), r["tot_res"]
        if e is not None and sd and res in ("0", "1"):
            tot[0].append(abs(e) / sd); tot[1].append(int(res))
    return side, tot

def cfb_rows():
    import cfb_backtest
    _, r = cfb_backtest.run()
    r = r[r.method == "new"]
    return (list(r.z), list(r.win.astype(int))), None

def main():
    out = {"shrink_games": SHRINK, "note": "Win rate against the closing line for picks in the same gap band, shrunk toward 50%.", "tables": {}}
    src = {}
    if os.path.exists("data/nfl/backtest_games.csv"):
        src["nfl"] = (nfl_rows(), "NFL 2022-25 vs nflverse closing lines")
    for lg in ("nba", "cbb"):
        p = f"data/{lg}/backtest_games.csv"
        if os.path.exists(p):
            src[lg] = (bb_rows(p), f"{lg.upper()} backtest vs closing lines")
    if os.path.exists("data/cfbd/history_games.csv"):
        try:
            src["cfb"] = (cfb_rows(), "CFB 2022-25, Elo screen vs closing lines")
        except Exception as e:
            print("cfb backtest rows failed:", e)
    for lg, ((side, tot), label) in src.items():
        for mk, rows in (("side", side), ("total", tot)):
            if not rows or (lg, mk) not in BANDS or not rows[0]:
                continue
            unit, edges = BANDS[(lg, mk)]
            out["tables"][f"{lg}_{mk}"] = {"unit": unit, "source": label, "games": len(rows[0]), "bands": table(rows[0], rows[1], edges)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    for k, t in out["tables"].items():
        print(k, t["games"], [(b["lo"], b["n"], b["raw"], b["conf"]) for b in t["bands"]])

if __name__ == "__main__":
    main()
