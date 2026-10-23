"""Shared walk-forward machinery for the v3 game-line models (accuracy first, then compare with the line).

walk(df, feats)            monthly retrain on earlier games: ridge + small boosted trees for margin and total,
                           averaged ("blend"); returns predictions for every game after the warm-up
report(p, label)           RMSE of each prediction vs the closing line's RMSE (the benchmark)
tiers(p, league)           P(home covers) / P(over) from a logistic fit of results on (prediction - line),
                           fit only on earlier months, vs DraftKings/ESPN BET opening and closing lines
                           (data/lines_hist/<league>.csv); prints how often the model's side won by confidence
"""
import numpy as np
import pandas as pd

def walk(df, feats, min_train=500, period="M"):
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    df = df.copy()
    df["date"] = pd.to_datetime(df.date)
    df["per"] = df.date.dt.to_period(period)
    X = df[feats].astype(float).fillna(0)
    preds = []
    for m in sorted(df.per.unique()):
        tr, te = (df.per < m).values, (df.per == m).values
        if tr.sum() < min_train or not te.any():
            continue
        o = df.loc[te].copy()
        for tgt, col in (("margin", "m"), ("total", "t")):
            ok = tr & df[tgt].notna().values
            y = df[tgt].astype(float)
            rg = Ridge(alpha=10.0).fit(X[ok], y[ok])
            o[f"{col}_ridge"] = rg.predict(X[te])
            b = lgb.train(dict(objective="regression", learning_rate=0.03, num_leaves=7, min_data_in_leaf=40, lambda_l2=10,
                               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=3),
                          lgb.Dataset(X[ok], y[ok]), 250)
            o[f"{col}_gbm"] = b.predict(X[te])
            o[f"{col}_blend"] = (o[f"{col}_ridge"] + o[f"{col}_gbm"]) / 2
        preds.append(o)
    return pd.concat(preds)

def report(p, label, old=("eff_margin", "eff_total")):
    rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    q = p[p.close.notna()]
    s = f"{label} margin RMSE ({len(q)} games): ridge {rm(q.m_ridge, q.margin):.2f} gbm {rm(q.m_gbm, q.margin):.2f} blend {rm(q.m_blend, q.margin):.2f}"
    if old[0] in q:
        s += f" | old {rm(q[old[0]], q.margin):.2f}"
    s += f" | close {rm(q.close, q.margin):.2f}"
    print(s)
    q = p[p.close_total.notna() & p.t_blend.notna()]
    if len(q):
        s = f"{label} total RMSE: ridge {rm(q.t_ridge, q.total):.2f} gbm {rm(q.t_gbm, q.total):.2f} blend {rm(q.t_blend, q.total):.2f}"
        if old[1] in q and q[old[1]].notna().all():
            s += f" | old {rm(q[old[1]], q.total):.2f}"
        s += f" | close {rm(q.close_total, q.total):.2f}"
        print(s)

def tiers(p, league, bands=(.5, .52, .54, .56, .58, 1), min_train=300, which=("open", "close"), extra_lines=None):
    from sklearn.linear_model import LogisticRegression
    L = pd.read_csv(f"data/lines_hist/{league}.csv", dtype={"game_id": str}).drop(columns=["season"])
    if extra_lines is not None:
        L = extra_lines.merge(L, on="game_id", how="outer")
    p = p.copy()
    p["game_id"] = p["id"].astype(str)
    p = p.merge(L, on="game_id", how="left", suffixes=("_m", ""))
    p["month"] = pd.to_datetime(p.date).dt.to_period("M")
    res = {}
    for when in which:
        for mk in ("side", "total"):
            if f"{when}_home_spread" not in p:
                continue
            if mk == "side":
                line = p[f"{when}_home_spread"]; gap = p.m_blend + line; out = p.margin + line
            else:
                if f"{when}_total" not in p or p.t_blend.isna().all():
                    continue
                line = p[f"{when}_total"]; gap = p.t_blend - line; out = p.total - line
            d = pd.DataFrame({"season": p.season, "month": p.month, "gap": gap, "out": out}).dropna()
            d = d[d.out != 0]
            d["y"] = (d.out > 0).astype(int)
            d["prob"] = np.nan
            for m in sorted(d.month.unique()):
                tr, te = d.month < m, d.month == m
                if tr.sum() < min_train:
                    continue
                lr = LogisticRegression().fit(d.loc[tr, ["gap"]], d.loc[tr, "y"])
                d.loc[te, "prob"] = lr.predict_proba(d.loc[te, ["gap"]])[:, 1]
            d = d.dropna(subset=["prob"])
            if d.empty:
                continue
            d["chance"] = np.maximum(d.prob, 1 - d.prob)
            d["won"] = np.where(d.prob >= .5, d.y, 1 - d.y)
            tab = d.groupby(pd.cut(d.chance, list(bands))).agg(n=("won", "size"), won=("won", "mean"), said=("chance", "mean"))
            res[f"{mk}_{when}"] = d
            print(f"\n{league} {mk} vs {when} ({len(d)} games): every pick won {d.won.mean():.3f}  by season {d.groupby('season').won.mean().round(3).to_dict()}")
            print(tab.round(3).to_string())
    return res
