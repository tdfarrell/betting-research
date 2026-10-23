#!/usr/bin/env python3
"""College basketball game lines v3: accuracy-first margin and total predictions, walk-forward, then compared with
the opening and closing lines (scripts/games_v3_common.py).

Inputs: the season-long and recent-form opponent-adjusted ratings (scripts/regrade_cbb.py), games played,
neutral site, rest days, conference game, and recent scoring margin / pace.

  python scripts/cbb_games_v3.py      -> data/cbb/v3_preds.csv + report
"""
import os, sys
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import regrade_cbb as C
import games_v3_common as V

def build():
    G = C.load()
    cache = "data/cbb/v3_rows.csv"
    if os.path.exists(cache):
        rows = pd.read_csv(cache, dtype={"id": str}).to_dict("records")
    else:
        rows = C.history_rows(G, [2025, 2026])
        pd.DataFrame(rows).to_csv(cache, index=False)
    meta = {g["id"]: g for g in G}
    last = {}
    hist = defaultdict(list)
    for g in sorted(G, key=lambda g: (g["date"], g["id"])):
        for t, m, tot in ((g["h"], g["hpts"] - g["apts"], g["hpts"] + g["apts"]), (g["a"], g["apts"] - g["hpts"], g["hpts"] + g["apts"])):
            key = (g["season"], t)
            h = hist[key]
            g.setdefault("_f", {})[t] = (float(np.mean(h[-5:])) if h else 0.0,)
            prev = last.get(key)
            g.setdefault("_r", {})[t] = (pd.Timestamp(g["date"]) - pd.Timestamp(prev)).days if prev else 7
            last[key] = g["date"]
            h.append(m)
    for r in rows:
        g = meta[r["id"]]
        r["neutral"] = int(bool(g.get("neutral")))
        r["conf"] = int(bool(g.get("conf")))
        r["h_rest"], r["a_rest"] = min(g["_r"][g["h"]], 10), min(g["_r"][g["a"]], 10)
        r["h_form"], r["a_form"] = g["_f"][g["h"]][0], g["_f"][g["a"]][0]
    df = pd.DataFrame(rows)
    return df

FEATS = ["eff_margin", "pts_margin", "eff_total", "pts_total", "gp", "neutral", "conf", "h_rest", "a_rest", "h_form", "a_form"]

if __name__ == "__main__":
    df = build()
    df.to_csv("data/cbb/v3_games.csv", index=False)
    p = V.walk(df, FEATS, min_train=1500)
    p.to_csv("data/cbb/v3_preds.csv", index=False)
    V.report(p, "CBB")
    V.tiers(p, "cbb")
