#!/usr/bin/env python3
"""Game-line models v3, live: train the final margin / total models per sport and turn a game's prediction into
the model's chance on each side of DraftKings' line.

Tested walk-forward in scripts/{nba,cbb,nfl,cfb}_games_v3.py: more accurate than the old models in every sport,
still behind the closing line, and the model's side wins about 50-53% of the time. Game lines get Picks only
(no Lean tag); the percentage is shown for context.

  python scripts/games_live.py train [nba cbb nfl cfb]   -> data/models/<lg>_games_v3.json (+ boosted trees .txt.gz)
  attach(lg, game, feats) is called by the graders for every board game: game["v3"] = model margin / total /
  chance of each side at DK's line.
"""
import gzip, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
MD = "data/models"

FEATS = {
    "nba": ["eff_margin", "pts_margin", "eff_total", "pts_total", "pace", "b2b", "gp", "hrest", "arest",
            "h_out_min", "a_out_min", "h_out_pts", "a_out_pts", "h_form", "a_form", "out_diff_pts", "form_diff",
            "h_lineup_pm", "a_lineup_pm", "lineup_diff", "h_avail_pts", "a_avail_pts", "avail_sum"],
    "cbb": ["eff_margin", "pts_margin", "eff_total", "pts_total", "gp", "neutral", "conf", "h_rest", "a_rest", "h_form", "a_form"],
    "nfl": ["epa_cal", "sr_cal", "pts_cal", "last_cal", "elo_cal", "mkt_cal", "qb_adj", "blend", "epa_tot_cal", "pts_tot_cal",
            "home_rest", "away_rest", "rest_diff", "div_game", "dome", "temp", "wind", "week"],
    "cfb": ["pts_margin", "pts_total", "elo_diff", "neutral", "conf_game", "gp", "week"],
}

def derive(lg, d):
    d = d.copy()
    if lg == "nba":
        d["out_diff_pts"] = d.a_out_pts - d.h_out_pts
        d["form_diff"] = d.h_form - d.a_form
        d["lineup_diff"] = d.h_lineup_pm - d.a_lineup_pm
        d["avail_sum"] = d.h_avail_pts + d.a_avail_pts
    return d

def table(lg):
    if lg == "nba":
        return pd.read_csv("data/nba/v3_games.csv", dtype={"id": str})
    if lg == "cbb":
        return pd.read_csv("data/cbb/v3_games.csv", dtype={"id": str})
    if lg == "nfl":
        import nfl_games_v3
        return nfl_games_v3.build()
    return pd.read_csv("data/cfbd/v3_games.csv", dtype={"id": str})

def cal_rows(lg):
    """Walk-forward predictions (games the model never saw) joined to opening and closing lines."""
    p = pd.read_csv(f"data/{ {'nba': 'nba', 'cbb': 'cbb', 'nfl': 'nfl', 'cfb': 'cfbd'}[lg] }/v3_preds.csv", dtype={"id": str})
    L = pd.read_csv(f"data/lines_hist/{lg}.csv", dtype={"game_id": str}).drop(columns=["season"])
    p = p.merge(L, left_on="id", right_on="game_id", how="left", suffixes=("_m", ""))
    out = {"side": [], "total": []}
    for when in ("open", "close"):
        hs, tl = p[f"{when}_home_spread"], p[f"{when}_total"]
        s = pd.DataFrame({"gap": p.m_blend + hs, "out": p.margin + hs}).dropna()
        out["side"].append(s[s.out != 0])
        if "t_blend" in p and p.t_blend.notna().any():
            t = pd.DataFrame({"gap": p.t_blend - tl, "out": p.total - tl}).dropna()
            out["total"].append(t[t.out != 0])
    return {k: pd.concat(v) if v else pd.DataFrame() for k, v in out.items()}

def train(lg):
    import lightgbm as lgb
    from sklearn.linear_model import Ridge, LogisticRegression
    d = derive(lg, table(lg))
    F = FEATS[lg]
    X = d[F].astype(float).fillna(0)
    cfg = {"features": F, "ridge": {}, "cal": {}, "walk_forward": {}}
    for tgt, col in (("margin", "m"), ("total", "t")):
        ok = d[tgt].notna().values
        rg = Ridge(alpha=10.0).fit(X[ok], d.loc[ok, tgt])
        cfg["ridge"][col] = {"coef": [float(c) for c in rg.coef_], "intercept": float(rg.intercept_)}
        b = lgb.train(dict(objective="regression", learning_rate=0.03, num_leaves=7, min_data_in_leaf=40, lambda_l2=10,
                           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=3), lgb.Dataset(X[ok], d.loc[ok, tgt]), 250)
        with gzip.open(f"{MD}/{lg}_games_{col}.txt.gz", "wt") as f:
            f.write(b.model_to_string())
    for mk, r in cal_rows(lg).items():
        if len(r) < 200:
            continue
        lr = LogisticRegression().fit(r[["gap"]], (r.out > 0).astype(int))
        a_, b_ = float(lr.intercept_[0]), float(lr.coef_[0][0])
        if b_ <= 0:
            # the model's disagreements with the line haven't pointed the right way: no confidence either side
            a_, b_ = 0.0, 0.0
        cfg["cal"][mk] = {"a": a_, "b": b_, "n": int(len(r))}
        won = np.where(r.gap >= 0, r.out > 0, r.out < 0)   # the model's side
        cfg["walk_forward"][mk] = {"n": int(len(r)), "won": round(float(won.mean()), 3)}
    json.dump(cfg, open(f"{MD}/{lg}_games_v3.json", "w"), indent=1)
    print(lg, "saved", {k: v for k, v in cfg["cal"].items()}, cfg["walk_forward"])

_M = {}
def model(lg):
    import lightgbm as lgb
    if lg not in _M:
        p = f"{MD}/{lg}_games_v3.json"
        if not os.path.exists(p):
            _M[lg] = None
        else:
            cfg = json.load(open(p))
            cfg["gbm"] = {c: lgb.Booster(model_str=gzip.open(f"{MD}/{lg}_games_{c}.txt.gz", "rt").read()) for c in ("m", "t")}
            _M[lg] = cfg
    return _M[lg]

def predict(lg, feats):
    """feats: dict of feature values for one game -> (margin, total) or None."""
    cfg = model(lg)
    if not cfg:
        return None
    d = derive(lg, pd.DataFrame([feats]))
    x = d[cfg["features"]].astype(float).fillna(0).values
    out = []
    for c in ("m", "t"):
        r = cfg["ridge"][c]
        ridge = float(x[0] @ np.array(r["coef"]) + r["intercept"])
        gbm = float(cfg["gbm"][c].predict(x)[0])
        out.append((ridge + gbm) / 2)
    return out

def chance(lg, mk, gap):
    c = (model(lg) or {}).get("cal", {}).get(mk)
    if not c or gap is None:
        return None
    # slope only: the pick is always the side the model's own number favors, and the chance grows with the gap.
    # (The fitted intercept is a small home / over lean that would sometimes flip a pick against the model's number.)
    return float(1 / (1 + np.exp(-(c["b"] * gap))))

def attach(lg, g, feats):
    """g: board game with g['dk']. Sets g['v3'] with the model's margin/total, its spread/total on DK's scale,
    and the chance of each side at DK's current line."""
    pr = predict(lg, feats)
    if pr is None:
        return
    m, t = pr
    d = g.get("dk") or {}
    hs, tl = d.get("home_spread"), d.get("total")
    v = {"margin": round(m, 1), "home_spread": round(-m, 1), "total": round(t, 1) if lg != "cfb" or t else None}
    if hs is not None:
        p = chance(lg, "side", m + hs)
        if p is not None:
            v["p_home"] = round(p, 3)
    if tl is not None and t is not None:
        p = chance(lg, "total", t - tl)
        if p is not None:
            v["p_over"] = round(p, 3)
    g["v3"] = v

if __name__ == "__main__":
    if sys.argv[1] == "train":
        for lg in sys.argv[2:] or ["nba", "cbb", "nfl", "cfb"]:
            train(lg)
