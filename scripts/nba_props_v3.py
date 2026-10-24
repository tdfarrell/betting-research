#!/usr/bin/env python3
"""NBA props v3: predict each player's stat line as accurately as possible (no betting line as an input), then
compare the prediction with the prop line. Walk-forward: retrained each month on earlier games only, so every
prediction is scored on a game the model never saw.

  python scripts/nba_props_v3.py build     -> data/props_hist/nba_v3_table.csv.gz (every player-game + features)
  python scripts/nba_props_v3.py walk      -> data/props_hist/nba_v3_preds.csv.gz + accuracy report
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import nba_props_v2 as V2

H = "data/props_hist"
MKS = V2.MKS
TABLE = f"{H}/nba_v3_table.csv.gz"
PREDS = f"{H}/nba_v3_preds.csv.gz"

def build():
    pg, tg, hist = V2.player_history()
    tc = V2.team_context(pg, tg)
    lc = pd.read_csv("data/nba/lines_close.csv", dtype={"game_id": str})
    df = hist.merge(tc, on=["game_id", "team", "opp"], how="left")
    oppc = tc[["game_id", "team", "pace_e", "out_min", "out_pts"] + [f"allow_{c}" for c in ["pts", "reb", "ast", "tpm", "stl", "blk"]]]
    oppc = oppc.rename(columns={"team": "opp", "pace_e": "opp_pace", "out_min": "opp_out_min", "out_pts": "opp_out_pts",
                                **{f"allow_{c}": f"oallow_{c}" for c in ["pts", "reb", "ast", "tpm", "stl", "blk"]}})
    df = df.merge(oppc, on=["game_id", "opp"], how="left")
    df = df.merge(lc[["game_id", "home_spread", "total"]], on="game_id", how="left")
    df["spread"] = np.where(df.home == 1, df.home_spread, -df.home_spread)
    df["implied_tt"] = df.total / 2 - df.spread / 2
    df["b2b"] = (pd.to_numeric(df.rest, errors="coerce") == 1).astype(int)
    df = projected_starter(df)
    df.drop(columns=["home_spread"]).to_csv(TABLE, index=False, compression="gzip")
    print(len(df), "player-games", df.groupby("season").size().to_dict())

def projected_starter(df):
    """`starter` = the projected starter, known before tip, using the same rule as the live cards
    (scripts/nba_props.py Day.minutes): 3+ games this season -> share of the last 5 games started; otherwise this
    season's starts blended with last season's last-20 start share. Training on whether he ACTUALLY started (kept as
    `started`) taught the model that starter=0 means a bench game for sure, which the live card can't know; projected
    bench players then sat well under DK's line (Oct 23 research: claude/nba-props-under-lean.md)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df.date)
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    df["started"] = df.starter
    gs = df.groupby(["player_id", "season"], sort=False)
    n = gs.cumcount()
    starts = gs.starter.transform(lambda x: x.shift(1).fillna(0).cumsum())
    l5 = gs.starter.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    prev = df.groupby(["player_id", "season"]).starter.apply(lambda x: x.tail(20).mean()).to_dict()
    pr = pd.Series([prev.get((p, s - 1), np.nan) for p, s in zip(df.player_id, df.season)], index=df.index)
    has = pr.notna().astype(float)
    rest = (5 - n).clip(lower=0) * has
    blend = (starts + pr.fillna(0) * rest) / np.maximum(1, n + rest)
    df["start_rate"] = np.where(n >= 3, l5, blend)
    df["starter"] = (df.start_rate >= 0.5).astype(int)
    return df

BASE = ["starter", "min_e3", "min_e10", "min_last", "start_e5", "n_prev", "n_prev_season", "home", "rest", "b2b",
        "out_min", "out_pts", "out_n", "opp_out_min", "opp_out_pts", "pace_e", "opp_pace", "spread", "total", "implied_tt"]
COMBO = {"pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"], "pa": ["pts", "ast"], "ra": ["reb", "ast"]}

def feats_for(df, mk):
    parts = COMBO.get(mk, [mk])
    X = df[BASE].copy()
    for k in ("pm_e5", "pm_e20", "e5", "e15", "sd"):
        X[f"s_{k}"] = df[f"{mk}_{k}"]
    X["s_allow"] = sum(df[f"oallow_{p}"] for p in parts)
    X["proj_simple"] = (0.5 * df[f"{mk}_pm_e5"] + 0.5 * df[f"{mk}_pm_e20"]) * (0.6 * df.min_e3 + 0.4 * df.min_e10)
    # the single stats that make up a combo, so the model can weigh them
    for p in parts if mk in COMBO else []:
        X[f"{p}_pm_e20"] = df[f"{p}_pm_e20"]
    return X

QS = (0.1, 0.25, 0.5, 0.75, 0.9)

def walk(params=None, rounds=150):
    """Quantile models (10/25/50/75/90th percentile of the stat) per market, retrained monthly on earlier games.
    The over/under is decided by the middle of the distribution, not the average, so the chance of going over a
    line comes from these quantiles; it's then calibrated walk-forward."""
    import lightgbm as lgb
    df = pd.read_csv(TABLE, dtype={"game_id": str, "player_id": str})
    df["date"] = pd.to_datetime(df.date)
    df["month"] = df.date.dt.to_period("M")
    df = df[df.n_prev >= 3]
    if os.path.exists(f"{H}/nba_features.csv.gz"):
        lines = pd.read_csv(f"{H}/nba_features.csv.gz", dtype={"game_id": str, "athlete_id": str},
                            usecols=["game_id", "athlete_id", "market", "line", "over", "under", "p_mkt", "sim_p"])
    else:   # the features file isn't committed; the last walk-forward output carries the same lines
        lines = pd.read_csv(f"{H}/nba_v3_lines.csv.gz", dtype={"game_id": str, "athlete_id": str},
                            usecols=["game_id", "athlete_id", "market", "line", "over", "under", "p_mkt", "sim_p"]).drop_duplicates(["game_id", "athlete_id", "market"])
    P = dict(learning_rate=0.08, num_leaves=31, min_data_in_leaf=100, max_bin=63, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=5)
    if params:
        P.update(params)
    months = sorted(df.month.unique())
    out = []
    for mk in MKS:
        X = feats_for(df, mk); y = df[mk].astype(float)
        for m in months:
            tr, te = (df.month < m).values, (df.month == m).values
            if tr.sum() < 8000 or te.sum() == 0:
                continue
            o = df.loc[te, ["season", "date", "game_id", "player_id", "min"]].copy()
            o["market"], o["actual"], o["base_e15"], o["base_simple"] = mk, y[te].values, X.s_e15[te].values, X.proj_simple[te].values
            b = lgb.train({**P, "objective": "regression"}, lgb.Dataset(X[tr], y[tr]), rounds)
            o["pred"] = b.predict(X[te])
            for q in QS:
                bq = lgb.train({**P, "objective": "quantile", "alpha": q}, lgb.Dataset(X[tr], y[tr]), rounds)
                o[f"q{int(q*100)}"] = bq.predict(X[te])
            out.append(o)
        print(" ", mk, "done", flush=True)
    r = pd.concat(out)
    r = r.merge(lines, left_on=["game_id", "player_id", "market"], right_on=["game_id", "athlete_id", "market"], how="left")
    r.to_csv(PREDS, index=False, compression="gzip")
    report(r)

def p_over_from_q(r, line):
    """Chance the stat beats `line`, read off the predicted quantiles (linear between them, tails stretched)."""
    qs = np.array(QS)
    vals = np.sort(np.c_[[r[f"q{int(q*100)}"] for q in QS]].T, axis=1)
    out = np.empty(len(line))
    for i, (v, L) in enumerate(zip(vals, line)):
        cdf = np.interp(L, v, qs, left=None, right=None)
        if L < v[0]:
            cdf = max(0.01, 0.1 * (L - (v[0] - 2 * (v[1] - v[0]) - 1e-6)) / (2 * (v[1] - v[0]) + 1e-6))
        elif L > v[-1]:
            cdf = min(0.99, 0.9 + 0.1 * (L - v[-1]) / (2 * (v[-1] - v[-2]) + 1e-6))
        out[i] = 1 - min(max(cdf, 0.01), 0.99)
    return out

def report(r):
    print("\nAccuracy on every player-game (RMSE / MAE; lower is better):")
    for mk, g in r.groupby("market"):
        e = lambda c: (np.sqrt(((g[c] - g.actual) ** 2).mean()), (g[c] - g.actual).abs().mean())
        print(f"  {mk:4s} n={len(g):6d}  model {e('pred')[0]:.2f}/{e('pred')[1]:.2f}   recent avg {e('base_e15')[0]:.2f}/{e('base_e15')[1]:.2f}   rate x minutes {e('base_simple')[0]:.2f}/{e('base_simple')[1]:.2f}")
    L = r[r.line.notna()]
    print("\nOn games with a sportsbook prop line (the line as a prediction is the benchmark):")
    for mk, g in L.groupby("market"):
        rm = lambda c: np.sqrt(((g[c] - g.actual) ** 2).mean())
        print(f"  {mk:4s} n={len(g):6d}  model {rm('pred'):.2f}   line {rm('line'):.2f}   recent avg {rm('base_e15'):.2f}")
    L = L.copy()
    L["p_raw"] = p_over_from_q(L, L.line.values)
    L["side"] = np.where(L.p_raw >= .5, "over", "under")
    L["win"] = np.where(L.side == "over", L.actual > L.line, L.actual < L.line).astype(int)
    L["chance_raw"] = np.maximum(L.p_raw, 1 - L.p_raw)
    print("\nModel's side vs the line, by the model's chance (raw, before calibration):")
    print(L.groupby(["side", pd.cut(L.chance_raw, [.5, .55, .6, .65, .7, 1])]).win.agg(["size", "mean"]).round(3).to_string())
    L.to_csv(f"{H}/nba_v3_lines.csv.gz", index=False, compression="gzip")

if __name__ == "__main__":
    {"build": build, "walk": walk}[sys.argv[1]]()
