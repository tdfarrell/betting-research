#!/usr/bin/env python3
"""NBA props v3, live: train the final models and attach predictions to the daily prop cards.

The model (tested walk-forward in scripts/nba_props_v3.py) predicts the 10th / 25th / 50th / 75th / 90th
percentile of each stat for a player's next game from what's known before tip: minutes and per-minute rates over
several windows, projected starter, rest, teammates out (injury report), opponent allowances and pace, and the
game's spread / total. The chance of beating any line comes from those percentiles (p_over below), then a
calibration fit on the walk-forward results makes the percentage honest (in testing, 57% picks won 57%).

  python scripts/nba_props_live.py train    -> data/models/nba_props/*.txt.gz + data/models/nba_props_v3.json
  (scripts/nba_props.py calls enrich() for every card it builds)

Pick = the side the calibrated chance favors at DraftKings' line. LEAN = 56%+ (walk-forward: 57.4%).
The page (screener/index.html v3Over) repeats p_over + calibration so any line on the trends slider works.
"""
import gzip, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
MD = "data/models/nba_props"
CFG = "data/models/nba_props_v3.json"
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
LEAN = 0.56

def knots(q):
    """CDF knots from the five predicted percentiles, with stretched tails; monotone."""
    v = np.maximum.accumulate(np.asarray(q, float))
    lo = v[0] - 1.6 * max(v[1] - v[0], 0.5)
    hi = v[4] + 1.6 * max(v[4] - v[3], 0.5)
    return np.r_[lo, v, hi], np.array([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])

def p_over(q, line):
    x, c = knots(q)
    # np.interp needs increasing x; ties are fine (it takes the right-most)
    return float(1 - np.interp(line, x, c))

def logit(p):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return np.log(p / (1 - p))

def calibrated(cfg, mk, p_raw):
    c = cfg["cal"]
    z = c["const"] + c["offset"].get(mk, 0.0) + c["slope"] * float(logit(p_raw))
    return float(1 / (1 + np.exp(-z)))

# ------------------------------------------------------------------ training
def train(rounds=150):
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    import nba_props_v3 as V3
    V3.build()
    df = pd.read_csv(V3.TABLE, dtype={"game_id": str, "player_id": str})
    df = df[df.n_prev >= 3]
    P = dict(learning_rate=0.08, num_leaves=31, min_data_in_leaf=100, max_bin=63, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=5)
    os.makedirs(MD, exist_ok=True)
    feats = {}
    for mk in V3.MKS:
        X = V3.feats_for(df, mk); y = df[mk].astype(float)
        feats[mk] = list(X.columns)
        for q in QS:
            b = lgb.train({**P, "objective": "quantile", "alpha": q}, lgb.Dataset(X, y), rounds)
            with gzip.open(f"{MD}/{mk}_q{int(q * 100)}.txt.gz", "wt") as f:
                f.write(b.model_to_string())
        print("trained", mk, flush=True)
    # calibration: on the walk-forward predictions (games each model never saw), same p_over as live
    L = pd.read_csv(f"{V3.H}/nba_v3_lines.csv.gz")
    L = L[L.actual != L.line].copy()
    L["praw"] = [p_over([r.q10, r.q25, r.q50, r.q75, r.q90], r.line) for r in L.itertuples()]
    # Slope only (Oct 23): calibration sets how far to trust the model's chance, never the side. With an intercept
    # and per-market offsets it learned the league-wide under habit at prop lines and overrode the model on close
    # calls (73% unders on Oct 23 while the model's own median favored the under on 57%).
    # Research: claude/nba-props-under-lean.md
    lr = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False).fit(logit(L.praw).reshape(-1, 1), (L.actual > L.line).astype(int))
    cfg = {"quantiles": QS, "lean": LEAN, "features": feats, "trained_rows": int(len(df)), "starter": "projected",
           "cal": {"const": 0.0, "slope": float(lr.coef_[0][0]), "offset": {}},
           "walk_forward": walk_forward_tiers(L)}
    json.dump(cfg, open(CFG, "w"), indent=1)
    print("saved", CFG, cfg["cal"], cfg["walk_forward"])

def walk_forward_tiers(L):
    """Hit rate by calibrated chance, refit month by month on earlier months only (the honest version)."""
    from sklearn.linear_model import LogisticRegression
    L = L.copy(); L["month"] = pd.to_datetime(L.date).dt.to_period("M")
    L["pc"] = np.nan
    X = lambda d: logit(d.praw).reshape(-1, 1)
    for m in sorted(L.month.unique()):
        tr, te = L.month < m, L.month == m
        if tr.sum() < 3000:
            continue
        f = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False).fit(X(L[tr]), (L[tr].actual > L[tr].line).astype(int))
        L.loc[te, "pc"] = f.predict_proba(X(L[te]))[:, 1]
    d = L.dropna(subset=["pc"])
    ch = np.maximum(d.pc, 1 - d.pc); won = np.where(d.pc >= .5, d.actual > d.line, d.actual < d.line)
    lean = ch >= LEAN
    return {"lean": {"n": int(lean.sum()), "won": round(float(won[lean].mean()), 3)},
            "pick": {"n": int((~lean).sum()), "won": round(float(won[~lean].mean()), 3)},
            "all": {"n": int(len(d)), "won": round(float(won.mean()), 3)}}

# ------------------------------------------------------------------ live
_B = {}
def boosters():
    import lightgbm as lgb
    if not _B:
        for f in os.listdir(MD):
            if f.endswith(".txt.gz"):
                _B[f[:-7]] = lgb.Booster(model_str=gzip.open(f"{MD}/{f}", "rt").read())
    return _B

def enrich(cards, day, S, games, inj):
    """Attach v3 percentiles (m['v3q']) to every card's markets. games: DK board rows for the day."""
    if not cards or not os.path.exists(CFG):
        return
    import nba_props as NP
    import nba_props_v2 as V2
    import nba_props_v3 as V3
    cfg = json.load(open(CFG))
    pg, tg = NP.load()
    gid = {}
    syn_p, syn_t = [], []
    for b in games:
        for team, opp, home in ((b["home"], b["away"], 1), (b["away"], b["home"], 0)):
            gid[(team, str(b["game_id"]))] = (opp, home, b)
            syn_t.append({"season": S, "date": day, "game_id": str(b["game_id"]), "team": team, "opp": opp, "home": home})
    for c in cards:
        if c.get("started"):
            continue
        syn_p.append({"season": S, "date": day, "game_id": c["game_id"], "team": c["team"], "opp": c["opp"], "home": int(c["home"]),
                      "player_id": str(c["id"]), "name": c["name"], "pos": c["pos"], "starter": int(bool(c.get("starter"))),
                      "dnp": 0, "min": 1.0, **{k: 0.0 for k in ["pts", "reb", "ast", "tpm", "stl", "blk", "tov"]}})
    # injured (Out) players on these teams -> did-not-play rows, so "teammates out" works like in training
    teams = {t for t, _ in gid}
    for pid, r in inj.items():
        if (r.get("status") or "").lower() == "out" and r.get("team") in teams:
            g = next(k for k in gid if k[0] == r["team"])
            opp, home, _ = gid[g]
            syn_p.append({"season": S, "date": day, "game_id": g[1], "team": r["team"], "opp": opp, "home": home, "player_id": str(pid),
                          "name": r.get("name", ""), "pos": "F", "starter": 0, "dnp": 1, "min": 0.0,
                          **{k: 0.0 for k in ["pts", "reb", "ast", "tpm", "stl", "blk", "tov"]}})
    if not syn_p:
        return
    sp = pd.DataFrame(syn_p)
    for k, parts in V3.COMBO.items():
        sp[k] = 0.0
    sp["pos3"] = sp.pos.map(NP.pos3)
    st = pd.DataFrame(syn_t)
    pg2 = pd.concat([pg, sp], ignore_index=True)
    tg2 = pd.concat([tg, st], ignore_index=True)
    # same feature code as training, on history + today's rows
    import types
    old_load = NP.load
    NP.load = lambda: (pg2.copy(), tg2.copy())
    try:
        pgx, tgx, hist = V2.player_history()
        tc = V2.team_context(pgx, tgx)
    finally:
        NP.load = old_load
    df = hist[hist.date == pd.Timestamp(day)].merge(tc, on=["game_id", "team", "opp"], how="left")
    df = df[df.n_prev >= 3]   # same as training: rookies / players with under 3 NBA games get no model number
    oppc = tc[["game_id", "team", "pace_e", "out_min", "out_pts"] + [f"allow_{c}" for c in ["pts", "reb", "ast", "tpm", "stl", "blk"]]]
    oppc = oppc.rename(columns={"team": "opp", "pace_e": "opp_pace", "out_min": "opp_out_min", "out_pts": "opp_out_pts",
                                **{f"allow_{c}": f"oallow_{c}" for c in ["pts", "reb", "ast", "tpm", "stl", "blk"]}})
    df = df.merge(oppc, on=["game_id", "opp"], how="left")
    sp_tot = {str(b["game_id"]): (b.get("home_spread"), b.get("total")) for b in games}
    hs = df.game_id.map(lambda g: sp_tot.get(g, (None, None))[0]).astype(float)
    tot = df.game_id.map(lambda g: sp_tot.get(g, (None, None))[1]).astype(float)
    df["spread"] = np.where(df.home == 1, hs, -hs)
    df["total"] = tot
    df["implied_tt"] = df.total / 2 - df.spread / 2
    df["b2b"] = (pd.to_numeric(df.rest, errors="coerce") == 1).astype(int)
    df["starter"] = df.player_id.map({str(c["id"]): int(bool(c.get("starter"))) for c in cards}).fillna(0)
    B = boosters()
    pred = {}
    for mk in V3.MKS:
        X = V3.feats_for(df, mk)[cfg["features"][mk]]
        qs = np.c_[[B[f"{mk}_q{int(q * 100)}"].predict(X) for q in QS]].T
        for pid, q in zip(df.player_id, qs):
            pred[(str(pid), mk)] = [round(float(v), 2) for v in np.maximum.accumulate(q)]
    for c in cards:
        for mk, m in c["markets"].items():
            q = pred.get((str(c["id"]), mk))
            if q:
                m["v3q"] = q
                m["proj"] = round(q[2], 1)      # the model's median
    return cfg

def pick(cfg, mk, q, line):
    """Calibrated chance and side at a line. Returns dict or None."""
    if q is None or line is None:
        return None
    p = calibrated(cfg, mk, p_over(q, float(line)))
    over = p >= 0.5
    ch = p if over else 1 - p
    return {"side": "over" if over else "under", "p_over": round(p, 3), "chance": round(ch, 3),
            "tag": "LEAN" if ch >= cfg.get("lean", LEAN) else "PICK"}

if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
