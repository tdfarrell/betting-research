#!/usr/bin/env python3
"""NBA game lines v3: predict each game's margin and total as accurately as possible (no betting line as an input),
then compare with DraftKings' line. Walk-forward: retrained every month on earlier games only.

Inputs per game (all known before tip): the existing opponent-adjusted efficiency and points ratings
(scripts/regrade_nba.py Ratings), pace, rest / back-to-backs, recent form (last ~5 games), and who's out:
the usual minutes and points of rotation players not playing for each team (the injury report, before tip).

  python scripts/nba_games_v3.py      -> data/nba/v3_games.csv + accuracy report
"""
import os, sys
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import regrade_nba as R

OUT = "data/nba/v3_games.csv"

def availability():
    """Per (game, team): usual minutes / points of rotation players who didn't play."""
    pg = pd.read_csv("data/nba/player_games.csv", dtype={"game_id": str, "player_id": str})
    for c in ("min", "pts", "dnp"):
        pg[c] = pd.to_numeric(pg[c], errors="coerce").fillna(0)
    pg["date"] = pd.to_datetime(pg.date)
    pg = pg.sort_values(["player_id", "date"])
    played = (pg.dnp == 0) & (pg["min"] > 0)
    pg["mp"] = pg["min"].where(played); pg["pp"] = pg["pts"].where(played)
    g = pg.groupby("player_id")
    pg["usual_min"] = g.mp.transform(lambda x: x.shift(1).ewm(halflife=8, min_periods=1).mean())
    pg["usual_pts"] = g.pp.transform(lambda x: x.shift(1).ewm(halflife=8, min_periods=1).mean())
    out = pg[~played & (pg.usual_min >= 12)].groupby(["game_id", "team"]).agg(out_min=("usual_min", "sum"), out_pts=("usual_pts", "sum")).reset_index()
    av = {(r.game_id, r.team): (r.out_min, r.out_pts) for r in out.itertuples()}
    # lineup strength of who's actually available: each player's plus-minus per minute (shrunk toward 0, slow average)
    # times his usual minutes, summed over players who play; and the usual points of those players
    pg["pmm"] = (pd.to_numeric(pg.pm, errors="coerce") / pg["min"].clip(lower=1)).where(played)
    pg["pmm_n"] = g.pmm.transform(lambda x: x.shift(1).notna().cumsum())
    pg["pmm_e"] = g.pmm.transform(lambda x: x.shift(1).ewm(halflife=25, min_periods=1).mean()).fillna(0) * (pg.pmm_n / (pg.pmm_n + 15))
    on = pg[played]
    st = (on.assign(v=on.pmm_e * on.usual_min.fillna(0), sp=on.usual_pts.fillna(0))
          .groupby(["game_id", "team"]).agg(lineup_pm=("v", "sum"), avail_pts=("sp", "sum")).reset_index())
    ls = {(r.game_id, r.team): (r.lineup_pm, r.avail_pts) for r in st.itertuples()}
    return av, ls

def form(G):
    """Exponentially weighted recent margin (per team, halflife 4 games) entering each game."""
    hist = defaultdict(list)
    out = {}
    for g in G:
        for t, o, m in ((g["home"], g["away"], g["hpts"] - g["apts"]), (g["away"], g["home"], g["apts"] - g["hpts"])):
            h = hist[(g["season"], t)]
            if h:
                w = 0.5 ** (np.arange(len(h))[::-1] / 4)
                out[(g["id"], t)] = float(np.dot(w, h) / w.sum())
            h.append(m)
    return out

def build():
    T, L = R.load()
    G = R.games_of(T)
    rows = R.history_rows(G, L, [2025, 2026, 2027])
    (av, ls), fm = availability(), form(G)
    for r in rows:
        for side, t in (("h", r["home"]), ("a", r["away"])):
            om, op = av.get((r["id"], t), (0.0, 0.0))
            r[f"{side}_out_min"], r[f"{side}_out_pts"] = om, op
            r[f"{side}_form"] = fm.get((r["id"], t), 0.0)
            r[f"{side}_lineup_pm"], r[f"{side}_avail_pts"] = ls.get((r["id"], t), (0.0, 0.0))
    g = {x["id"]: x for x in G}
    for r in rows:
        r["hrest"], r["arest"] = g[r["id"]]["hrest"], g[r["id"]]["arest"]
    df = pd.DataFrame(rows)
    import nba_player_rating as PR
    df["rapm_margin"] = PR.rapm(df.rename(columns={"id": "id"}), alpha=10.0)
    df.to_csv(OUT, index=False)
    return df

FEATS = ["eff_margin", "pts_margin", "eff_total", "pts_total", "pace", "b2b", "gp", "hrest", "arest",
         "h_out_min", "a_out_min", "h_out_pts", "a_out_pts", "h_form", "a_form"]

def walk(df):
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    df = df.copy()
    df["date"] = pd.to_datetime(df.date); df["month"] = df.date.dt.to_period("M")
    df["out_diff_pts"] = df.a_out_pts - df.h_out_pts
    df["form_diff"] = df.h_form - df.a_form
    df["lineup_diff"] = df.h_lineup_pm - df.a_lineup_pm
    df["avail_sum"] = df.h_avail_pts + df.a_avail_pts
    F = FEATS + ["rapm_margin", "out_diff_pts", "form_diff", "h_lineup_pm", "a_lineup_pm", "lineup_diff", "h_avail_pts", "a_avail_pts", "avail_sum"]
    X = df[F].fillna(0).astype(float)
    preds = []
    for m in sorted(df.month.unique()):
        tr, te = (df.month < m).values, (df.month == m).values
        if tr.sum() < 500 or not te.any():
            continue
        o = df.loc[te, ["id", "season", "date", "home", "away", "margin", "total", "close", "close_total", "eff_margin", "pts_margin", "eff_total", "pts_total"]].copy()
        for tgt, col in (("margin", "m"), ("total", "t")):
            y = df[tgt].astype(float)
            rg = Ridge(alpha=10.0).fit(X[tr], y[tr])
            o[f"{col}_ridge"] = rg.predict(X[te])
            b = lgb.train(dict(objective="regression", learning_rate=0.03, num_leaves=7, min_data_in_leaf=40, lambda_l2=10,
                               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=3), lgb.Dataset(X[tr], y[tr]), 250)
            o[f"{col}_gbm"] = b.predict(X[te])
            o[f"{col}_blend"] = (o[f"{col}_ridge"] + o[f"{col}_gbm"]) / 2
        preds.append(o)
    return pd.concat(preds)

def report(p):
    rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    ok = p.close.notna()
    q = p[ok]
    print(f"Margin RMSE on {len(q)} games (lower is better): ridge {rm(q.m_ridge, q.margin):.2f}  gbm {rm(q.m_gbm, q.margin):.2f}  blend {rm(q.m_blend, q.margin):.2f}"
          f"  | old model: efficiency {rm(q.eff_margin, q.margin):.2f}, points {rm(q.pts_margin, q.margin):.2f}  | DK close {rm(q.close, q.margin):.2f}")
    q = p[p.close_total.notna()]
    print(f"Total RMSE: ridge {rm(q.t_ridge, q.total):.2f}  gbm {rm(q.t_gbm, q.total):.2f}  blend {rm(q.t_blend, q.total):.2f}"
          f"  | old: efficiency {rm(q.eff_total, q.total):.2f}, points {rm(q.pts_total, q.total):.2f}  | DK close {rm(q.close_total, q.total):.2f}")

def calibrate_and_tier(p):
    """Compare predictions with DraftKings' opening and closing lines. P(home covers) / P(over) comes from a
    logistic fit of results on the gap (prediction vs line), using only earlier months. Reports how often the
    model's side won at each confidence level."""
    from sklearn.linear_model import LogisticRegression
    L = pd.read_csv("data/lines_hist/nba.csv", dtype={"game_id": str}).drop(columns=["season"])
    p = p.merge(L, left_on=p["id"].astype(str), right_on="game_id", how="left", suffixes=("_m", ""))
    p["month"] = pd.to_datetime(p.date).dt.to_period("M")
    res = {}
    for when in ("open", "close"):
        for mk in ("side", "total"):
            if mk == "side":
                line = p[f"{when}_home_spread"]; gap = p.m_blend + line; out = p.margin + line
            else:
                line = p[f"{when}_total"]; gap = p.t_blend - line; out = p.total - line
            d = pd.DataFrame({"season": p.season, "month": p.month, "gap": gap, "y": (out > 0).astype(int), "push": out == 0}).dropna()
            d = d[~d.push]
            d["prob"] = np.nan
            for m in sorted(d.month.unique()):
                tr, te = d.month < m, d.month == m
                if tr.sum() < 300:
                    continue
                lr = LogisticRegression().fit(d.loc[tr, ["gap"]], d.loc[tr, "y"])
                d.loc[te, "prob"] = lr.predict_proba(d.loc[te, ["gap"]])[:, 1]
            d = d.dropna(subset=["prob"])
            d["chance"] = np.maximum(d.prob, 1 - d.prob)
            d["won"] = np.where(d.prob >= .5, d.y, 1 - d.y)
            tab = d.groupby(pd.cut(d.chance, [.5, .52, .54, .56, .58, 1])).agg(n=("won", "size"), won=("won", "mean"), said=("chance", "mean"))
            res[f"{mk}_{when}"] = d
            print(f"\n{mk} vs {when} ({len(d)} games): every pick won {d.won.mean():.3f}")
            print(tab.round(3).to_string())
    return res

if __name__ == "__main__":
    df = build()
    p = walk(df)
    p.to_csv("data/nba/v3_preds.csv", index=False)
    report(p)
    calibrate_and_tier(p)

# ------------------------------------------------------------------ live (scripts/regrade_nba.py -> games_live.attach)
def live_team_feats(day, season, INJ):
    """Per team for a slate day, from games before it + the injury report (Out = not playing):
    out_min / out_pts (usual minutes and points of rotation players listed Out), lineup_pm and avail_pts
    (players who played for the team in its last 10 games and aren't Out), and recent form."""
    pg = pd.read_csv("data/nba/player_games.csv", dtype={"game_id": str, "player_id": str})
    for c in ("min", "pts", "dnp", "pm"):
        pg[c] = pd.to_numeric(pg[c], errors="coerce").fillna(0)
    pg = pg[pg.date < day].copy()
    pg["date"] = pd.to_datetime(pg.date)
    pg = pg.sort_values(["player_id", "date"])
    played = (pg.dnp == 0) & (pg["min"] > 0)
    P = pg[played].copy()
    g = P.groupby("player_id")
    usual_min = g["min"].apply(lambda x: x.ewm(halflife=8).mean().iloc[-1])
    usual_pts = g["pts"].apply(lambda x: x.ewm(halflife=8).mean().iloc[-1])
    pmm = (P.pm / P["min"].clip(lower=1))
    P["pmm"] = pmm
    n = g.size()
    pmm_e = P.groupby("player_id").pmm.apply(lambda x: x.ewm(halflife=25).mean().iloc[-1]) * (n / (n + 15))
    cur = P[P.season == season]
    out = {}
    tg = pd.read_csv("data/nba/team_games.csv")
    tg = tg[(tg.season == season) & (tg.date < day)].sort_values("date")
    for team, tgrp in cur.groupby("team"):
        dates = sorted(tgrp.date.unique())[-10:]
        recent = set(tgrp[tgrp.date.isin(dates)].player_id)
        is_out = lambda pid: ((INJ.get(pid) or {}).get("status") or "").lower() == "out"
        avail = [p for p in recent if not is_out(p)]
        outs = [p for p in recent if is_out(p) and usual_min.get(p, 0) >= 12]
        m = tg[tg.team == team]
        diffs = (m.pts - m.opp_pts).tolist()
        w = 0.5 ** (np.arange(len(diffs))[::-1] / 4) if diffs else []
        out[team] = {"out_min": float(sum(usual_min.get(p, 0) for p in outs)), "out_pts": float(sum(usual_pts.get(p, 0) for p in outs)),
                     "lineup_pm": float(sum(pmm_e.get(p, 0) * usual_min.get(p, 0) for p in avail)),
                     "avail_pts": float(sum(usual_pts.get(p, 0) for p in avail)),
                     "form": float(np.dot(w, diffs) / w.sum()) if len(diffs) else 0.0}
    return out

def live_feats(r, h, a, gp, T):
    th, ta = T.get(h, {}), T.get(a, {})
    return {"eff_margin": r["eff_margin"], "pts_margin": r["pts_margin"], "eff_total": r["eff_total"], "pts_total": r["pts_total"],
            "pace": r["pace"], "b2b": r["b2b"], "gp": gp, "hrest": r.get("hrest"), "arest": r.get("arest"),
            "h_out_min": th.get("out_min", 0), "a_out_min": ta.get("out_min", 0), "h_out_pts": th.get("out_pts", 0), "a_out_pts": ta.get("out_pts", 0),
            "h_form": th.get("form", 0), "a_form": ta.get("form", 0), "h_lineup_pm": th.get("lineup_pm", 0), "a_lineup_pm": ta.get("lineup_pm", 0),
            "h_avail_pts": th.get("avail_pts", 0), "a_avail_pts": ta.get("avail_pts", 0)}
