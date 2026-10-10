#!/usr/bin/env python3
"""CFB calibration backtest on past seasons (Elo only: past SP+ is only published as end-of-season ratings,
which would leak results). Each week is treated as a slate and fitted to its own lines, as the live grader does.

  python scripts/cfb_backtest.py   (needs data/cfbd/history_games.csv from the "Backfill CFB history" workflow)
-> data/cfbd/backtest_summary.json
"""
import json, sys
import numpy as np
import pandas as pd

HFA, ELO_PER_PT, OPTION = 2.5, 25.0, {"Army", "Navy", "Air Force"}

def fwd(x, y):
    b, a = np.polyfit(x, y, 1); return b, a

def rev(x, y):
    b, a = np.polyfit(y, x, 1); return 1 / b, -a / b

def run(bands=(0.3, 0.6, 1.0)):
    d = pd.read_csv("data/cfbd/history_games.csv")
    d = d[(d.home_div == "fbs") & (d.away_div == "fbs") & d.spread.notna() & d.home_elo.notna() & d.away_elo.notna()]
    d = d[(d.spread.abs() < 25) & ~d.home.isin(OPTION) & ~d.away.isin(OPTION)].copy()
    d["x"] = -((d.home_elo - d.away_elo) / ELO_PER_PT + np.where(d.neutral == 1, 0, HFA))   # model home spread
    d["cover"] = (d.home_points - d.away_points) + d.spread                                 # + = home covered
    out = []
    for (s, w), g in d.groupby(["season", "week"]):
        if len(g) < 15:
            continue
        for name, f in (("old", fwd), ("new", rev)):
            b, a = f(g.x.values, g.spread.values)
            cal = b * g.x + a
            e = g.spread - cal                     # + = home value
            usual = float(np.sqrt(np.mean(e ** 2)))
            for idx, ee in e.items():
                out.append((name, s, w, ee, usual, g.at[idx, "spread"], g.at[idx, "cover"]))
    r = pd.DataFrame(out, columns=["method", "season", "week", "edge", "usual", "spread", "cover"])
    r["fav_pick"] = np.where(r.spread < 0, r.edge < 0, r.edge > 0)   # home fav & edge toward away... see below
    # picking home when edge > 0; home is favorite when spread < 0
    r["fav_pick"] = (r.edge > 0) == (r.spread < 0)
    r["z"] = r.edge.abs() / r.usual
    r["win"] = np.sign(r.edge) == np.sign(r.cover)
    r = r[r.cover != 0]
    summ = {}
    for name, g in r.groupby("method"):
        res = {"corr_edge_vs_line": round(float(np.corrcoef(np.where(g.spread < 0, g.edge, -g.edge), g.spread.abs())[0, 1]), 2)}
        lo_b = [0, bands[0], bands[1], bands[2], 9]
        for lo, hi, lab in zip(lo_b[:-1], lo_b[1:], ("under LEAN", "LEAN", "PLAY", "CHECK")):
            m = g[(g.z >= lo) & (g.z < hi)]
            res[lab] = {"w": int(m.win.sum()), "l": int((~m.win).sum()), "pct": round(float(m.win.mean()), 3),
                        "fav_share": round(float(m.fav_pick.mean()), 2), "per_week": round(len(m) / g.groupby(["season", "week"]).ngroups, 1),
                        "by_season": {int(s): f"{int(x.win.sum())}-{int((~x.win).sum())}" for s, x in m.groupby("season")}}
        bigline = g[(g.z >= bands[1])]
        res["flags_fav_share"] = round(float(g[(g.z >= bands[0]) & (g.z < bands[2])].fav_pick.mean()), 2)
        summ[name] = res
    return summ, r

if __name__ == "__main__":
    summ, r = run()
    with open("data/cfbd/backtest_summary.json", "w") as f:
        json.dump({"note": "Elo-only, each week fitted to its own lines; lines are DraftKings where CFBD has them, else ESPN Bet/consensus",
                   "games": int(len(r) / 2), **summ}, f, indent=1)
    print(json.dumps(summ, indent=1))
