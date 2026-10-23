#!/usr/bin/env python3
"""NFL game lines v3: accuracy-first margin and total predictions, walk-forward, then compared with the lines
(scripts/games_v3_common.py).

Inputs per game (data/nfl/backtest_games.csv, built week by week from earlier games only): EPA, success-rate and
points ratings, last season's rating, Elo, the market rating (from earlier weeks' closing lines), the QB
adjustment, EPA and points totals; plus rest days, divisional game, roof, temperature and wind (nflverse schedule).

  python scripts/nfl_games_v3.py      -> data/nfl/v3_preds.csv + report
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import games_v3_common as V

def build():
    b = pd.read_csv("data/nfl/backtest_games.csv")
    s = pd.read_csv("data/nfl/schedule.csv", dtype={"espn": str})
    s = s[["game_id", "gameday", "espn", "away_rest", "home_rest", "div_game", "roof", "temp", "wind"]]
    d = b.merge(s, on="game_id", how="left")
    d["id"] = d.espn.astype(str).str.split(".").str[0]
    d["date"] = d.gameday
    d["margin"] = d.result
    d["dome"] = d.roof.isin(["dome", "closed"]).astype(int)
    d["temp"] = np.where(d.dome == 1, 70, d.temp.fillna(60))
    d["wind"] = np.where(d.dome == 1, 0, d.wind.fillna(5))
    d["rest_diff"] = d.home_rest - d.away_rest
    d["eff_margin"], d["eff_total"] = d.blend, (d.epa_tot_cal + d.pts_tot_cal) / 2
    return d

FEATS = ["epa_cal", "sr_cal", "pts_cal", "last_cal", "elo_cal", "mkt_cal", "qb_adj", "blend", "epa_tot_cal", "pts_tot_cal",
         "home_rest", "away_rest", "rest_diff", "div_game", "dome", "temp", "wind", "week"]

if __name__ == "__main__":
    d = build()
    p = V.walk(d, FEATS, min_train=250)
    p.to_csv("data/nfl/v3_preds.csv", index=False)
    V.report(p, "NFL")
    ref = pd.DataFrame({"game_id": d.id, "ref_home_spread": -d.close, "ref_total": d.close_total})
    V.tiers(p, "nfl", which=("ref", "open", "close"), extra_lines=ref, min_train=200)
