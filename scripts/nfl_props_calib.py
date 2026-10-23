#!/usr/bin/env python3
"""Calibration layer for the NFL prop model (Oct 23 rebuild).

The simulation model in scripts/nfl_props.py was too sure of itself: when it said 70%+ the pick won about 60%.
This fits, per market, logit(P over) = a + b * logit(model P over) on graded backtest picks against real lines
(scripts/props_backtest.py, ESPN BET lines 2024-25, starting QB taken from the previous game so the backtest
doesn't know who actually started).

  python scripts/nfl_props_calib.py      walk-forward report + data/models/nfl_props_calib.json (fit on everything)

Walk-forward (each week calibrated only on earlier weeks), calibrated chance of the model's side:
  60%+ won 62.4% (2024 62.6%, 2025 61.7%), 57-60% won 57.2% (58.6% / 55.2%); about 98% are unders.
The live pick (scripts/picks.py prop_pick, page propPick) uses the calibrated chance; PLAY 60%+, LEAN 57-60%.
DraftKings' NFL prop prices aren't in ESPN's feed, so grading assumes -120 (unders usually cost more than -110).
Passing TDs are never picked: their lines (0.5 / 1.5) are usually priced far from even (-250 and the like), so a
win rate graded at -110 says nothing about profit.
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SRC = "data/props_hist/nfl_backtest_picks.csv.gz"
OUT = "data/models/nfl_props_calib.json"
NO_PICK = ["pass_td"]

def lg(p):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return np.log(p / (1 - p))

def fit(d, markets):
    oh = np.stack([(d.market == m).astype(float) for m in markets], 1)
    X = np.c_[oh, oh * lg(d.p_over)[:, None]]
    lr = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False).fit(X, d.y)
    k = len(markets)
    return {m: {"a": round(float(lr.coef_[0][i]), 4), "b": round(float(lr.coef_[0][k + i]), 4)} for i, m in enumerate(markets)}

def apply(cal, mk, p):
    c = cal.get(mk)
    if not c:
        return None
    z = c["a"] + c["b"] * float(lg(p))
    return 1 / (1 + np.exp(-z))

def main():
    df = pd.read_csv(SRC)
    df["y"] = np.where(df.side == "over", df.win, 1 - df.win)
    df["wk"] = df.season * 100 + df.date
    markets = sorted(df.market.unique())
    rows = []
    for w in sorted(df.wk.unique()):
        tr, te = df[df.wk < w], df[df.wk == w].copy()
        if len(tr) < 1500:
            continue
        cal = fit(tr, markets)
        te["pc"] = [apply(cal, m, p) for m, p in zip(te.market, te.p_over)]
        rows.append(te)
    r = pd.concat(rows)
    r = r[~r.market.isin(NO_PICK)]
    r["chance"] = np.maximum(r.pc, 1 - r.pc)
    r["won"] = np.where(r.pc >= 0.5, r.y, 1 - r.y)
    rep = {}
    for lo, hi, name in ((0.60, 1, "play"), (0.57, 0.60, "lean"), (0.5, 0.57, "untagged")):
        b = r[(r.chance >= lo) & (r.chance < hi)]
        rep[name] = {"n": int(len(b)), "win": round(float(b.won.mean()), 3),
                     "by_season": {int(s): [int(len(x)), round(float(x.won.mean()), 3)] for s, x in b.groupby("season")},
                     "under_share": round(float((b.pc < 0.5).mean()), 3)}
        print(name, rep[name])
    final = fit(df, markets)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"fitted_on": "ESPN BET NFL prop lines 2024-25 (2025 through week 13)", "picks": int(len(df)),
               "form": "P(over) = 1 / (1 + exp(-(a + b * logit(model P over))))", "markets": final,
               "tags": {"play": 0.60, "lean": 0.57}, "no_pick": NO_PICK, "assumed_price": -120, "walk_forward": rep}, open(OUT, "w"), indent=1)
    print("saved", OUT, final)

if __name__ == "__main__":
    main()
