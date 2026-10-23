#!/usr/bin/env python3
"""Walk-forward test of every game-line model against the lines it would bet into (Oct 23 rebuild).

For each sport, the model's own home margin / total for every past game (built only from earlier games) is
compared with the OPENING and the CLOSING line (data/lines_hist/<league>.csv, DraftKings or ESPN BET). A
calibration P(cover) = 1 / (1 + exp(-(a + b * gap))) is fit only on earlier games, then a bet is made when the
expected value at that line's actual price clears a threshold. Gate (Tim's rule): the model's bets as a whole
must be profitable at the real prices, in every season tested, or the model doesn't ship.

  python scripts/game_models_eval.py [nba|cbb|nfl|cfb ...]   -> printed report + data/grades/game_models_eval.json
"""
import json, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def dec(o):
    try:
        x = float(o)
    except (TypeError, ValueError):
        return 1.909
    if not np.isfinite(x) or x == 0:
        return 1.909
    return 1 + (x / 100 if x > 0 else 100 / -x)

def lines(lg):
    L = pd.read_csv(f"data/lines_hist/{lg}.csv", dtype={"game_id": str})
    L = L.drop_duplicates("game_id", keep="last").drop(columns=["season"])
    # longer closing-line history for football at an assumed -110 (nflverse 2022-25; CFBD consensus/DK 2022-25)
    if lg == "nfl":
        s = pd.read_csv("data/nfl/schedule.csv", dtype={"espn": str})
        s = s[s.espn.notna()]
        ref = pd.DataFrame({"game_id": s.espn.astype(str).str.split(".").str[0], "ref_home_spread": -s.spread_line,
                            "ref_total": s.total_line})
        L = ref.merge(L, on="game_id", how="left")
    if lg == "cfb":
        h = pd.read_csv("data/cfbd/history_games.csv", dtype={"id": str})
        ref = pd.DataFrame({"game_id": h.id, "ref_home_spread": h.spread, "ref_total": h.total,
                            "refopen_home_spread": h.spread_open, "refopen_total": h.total_open})
        L = ref.merge(L, on="game_id", how="left")
    for w in ("ref", "refopen"):
        if f"{w}_home_spread" in L:
            for k in ("home_price", "away_price", "over_price", "under_price"):
                L[f"{w}_{k}"] = -110.0
    return L

def model_rows(lg):
    """season, date, game_id, m_margin (model home margin), m_total, margin (actual), total (actual)."""
    if lg in ("nba", "cbb"):
        d = pd.read_csv(f"data/{lg}/backtest_games.csv", dtype={"game_id": str})
        return pd.DataFrame({"season": d.season, "date": d.date, "game_id": d.game_id, "m_margin": d.number,
                             "m_total": d.total_number, "margin": d.margin, "total": d.total})
    if lg == "nfl":
        d = pd.read_csv("data/nfl/backtest_games.csv")
        s = pd.read_csv("data/nfl/schedule.csv", dtype={"espn": str})[["game_id", "espn", "gameday"]]
        d = d.merge(s, on="game_id", how="left")
        d["m_total"] = (d.epa_tot_cal + d.pts_tot_cal) / 2
        return pd.DataFrame({"season": d.season, "date": d.gameday, "game_id": d.espn.astype(str).str.split(".").str[0],
                             "m_margin": d.blend, "m_total": d.m_total, "margin": d.result, "total": d.total})
    if lg == "cfb":
        d = pd.read_csv("data/cfbd/history_games.csv", dtype={"id": str})
        d = d[(d.home_div == "fbs") & (d.away_div == "fbs") & d.home_elo.notna() & d.away_elo.notna()]
        hfa = np.where(d.neutral == 1, 0, 2.5)
        return pd.DataFrame({"season": d.season, "date": d.start.str[:10], "game_id": d.id,
                             "m_margin": (d.home_elo - d.away_elo) / 25.0 + hfa, "m_total": np.nan,
                             "margin": d.home_points - d.away_points, "total": d.home_points + d.away_points})
    raise ValueError(lg)

def walk(df, gap_col, y_col, min_train=300):
    """Expanding-window logistic calibration by date. Returns P(y=1) per row (NaN before enough history)."""
    df = df.sort_values("date").copy()
    df["p"] = np.nan
    dates = sorted(df.date.unique())
    for d in dates:
        tr = df[(df.date < d) & df[y_col].notna()]
        if len(tr) < min_train or tr[y_col].nunique() < 2:
            continue
        lr = LogisticRegression(C=1.0).fit(tr[[gap_col]].values, tr[y_col].values.astype(int))
        m = df.date == d
        df.loc[m, "p"] = lr.predict_proba(df.loc[m, [gap_col]].values)[:, 1]
    return df

def evaluate(lg):
    M, L = model_rows(lg), lines(lg)
    D = M.merge(L, on="game_id", how="inner")
    out = {}
    for when in [w for w in ("open", "close", "ref", "refopen") if f"{w}_home_spread" in D]:
        for mkt in ("side", "total"):
            if mkt == "side":
                d = D.dropna(subset=["m_margin", f"{when}_home_spread", "margin"]).copy()
                hs = d[f"{when}_home_spread"]
                d["gap"] = d.m_margin + hs                          # + = model likes home vs this line
                cov = d.margin + hs
                d = d[cov != 0]; cov = cov[cov != 0]
                d["y"] = (cov > 0).astype(int)                      # home covered
                pa, pb = f"{when}_home_price", f"{when}_away_price"
            else:
                d = D.dropna(subset=["m_total", f"{when}_total", "total"]).copy()
                d["gap"] = d.m_total - d[f"{when}_total"]
                diff = d.total - d[f"{when}_total"]
                d = d[diff != 0]; diff = diff[diff != 0]
                d["y"] = (diff > 0).astype(int)                     # over
                pa, pb = f"{when}_over_price", f"{when}_under_price"
            if len(d) < 400:
                continue
            w = walk(d, "gap", "y").dropna(subset=["p"])
            da, db = w[pa].map(dec), w[pb].map(dec)
            ev_a, ev_b = w.p * da - 1, (1 - w.p) * db - 1
            side_a = ev_a >= ev_b
            w["ev"] = np.where(side_a, ev_a, ev_b)
            w["win"] = np.where(side_a, w.y, 1 - w.y)
            w["profit"] = np.where(w.win == 1, np.where(side_a, da - 1, db - 1), -1.0)
            # how much the model's gap tells you (slope of the walk-forward calibration on the last fit)
            lr = LogisticRegression(C=1.0).fit(d[["gap"]].values, d.y.values)
            res = {"games": int(len(w)), "slope_per_pt": round(float(lr.coef_[0][0]), 4), "thresholds": {}}
            for thr in (0.0, 0.02, 0.04, 0.06):
                b = w[w.ev >= thr]
                if len(b) < 30:
                    continue
                res["thresholds"][thr] = {"bets": int(len(b)), "win": round(float(b.win.mean()), 3), "roi": round(float(b.profit.mean()), 3),
                                          "by_season": {str(s): [int(len(x)), round(float(x.profit.mean()), 3)] for s, x in b.groupby("season")}}
            out[f"{mkt}_{when}"] = res
            print(lg, mkt, when, "games", len(w), "slope/pt", res["slope_per_pt"])
            for t, r in res["thresholds"].items():
                print(f"   EV>={t:.2f}: {r['bets']:5d} bets win {r['win']:.3f} ROI {r['roi']:+.3f}  {r['by_season']}")
    return out

if __name__ == "__main__":
    lgs = sys.argv[1:] or ["nba", "cbb", "nfl", "cfb"]
    allr = {}
    for lg in lgs:
        allr[lg] = evaluate(lg)
    try:
        prev = json.load(open("data/grades/game_models_eval.json"))
    except Exception:
        prev = {}
    prev.update(allr)
    json.dump(prev, open("data/grades/game_models_eval.json", "w"), indent=1)
