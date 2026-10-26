#!/usr/bin/env python3
"""NBA props v2: market-anchored, calibrated, judged walk-forward at the book's real prices.

Built from the Oct 23 research (claude/model-research-and-redesign.md):
  * start from the book's own probability (power-method devig of the over/under prices)
  * the model only moves that number with pre-game information: minutes and per-minute rates over several windows,
    usage when teammates are out, rest, home/away, blowout risk (spread), game total, opponent and pace, and the old
    simulation model's chance
  * fit by log loss (calibration), retrained every month on earlier games only (walk-forward), never graded on a
    game it learned from
  * bet only positive-EV sides at the actual price

  python scripts/nba_props_v2.py features       -> data/props_hist/nba_features.csv.gz
  python scripts/nba_props_v2.py walk            -> data/props_hist/nba_v2_walk.csv.gz + printed report
"""
import csv, gzip, json, math, os, sys
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
H = "data/props_hist"
STATS = ["pts", "reb", "ast", "tpm", "stl", "blk"]
COMBO = {"pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"], "pa": ["pts", "ast"], "ra": ["reb", "ast"]}
MKS = STATS + list(COMBO)

def implied(o):
    try:
        x = float(str(o).replace("+", ""))
    except (TypeError, ValueError):
        return None
    if not x:
        return None
    return -x / (-x + 100) if x < 0 else 100 / (x + 100)

def power_devig(a, b):
    """Power method: find k with a^k + b^k = 1 (lowest log loss in the literature). Returns fair P(first)."""
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        s = a ** k + b ** k
        if s > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    return a ** k / (a ** k + b ** k)

def decimal(o):
    x = float(str(o).replace("+", ""))
    return 1 + (x / 100 if x > 0 else 100 / -x)

# ------------------------------------------------------------------ features
# How much a previous season's game counts next to a game from the current season, in the rolling windows below
# (Oct 26: was 1.0, i.e. last season's games counted fully, so 3 games into a new season the 20-game windows were
# still ~85% last season and new roles barely moved the model). Each season further back multiplies again.
SEASON_CARRY = float(os.environ.get("NBA_SEASON_CARRY", "0.25"))

def ewm_season(vals, pid, season, hl, carry=None, min_periods=1):
    """EWM over each player's (or team's) PREVIOUS rows, like x.shift(1).ewm(halflife=hl).mean(), except rows from
    an earlier season are down-weighted by `carry` per season of distance from the row being predicted.
    carry=1 reproduces pandas exactly. vals: (n, k) array, rows sorted by pid then date; NaN = skip."""
    carry = SEASON_CARRY if carry is None else carry
    v = np.asarray(vals, float); v = v.reshape(len(v), -1)
    a = 0.5 ** (1.0 / hl)
    out = np.full(v.shape, np.nan)
    num = np.zeros(v.shape[1]); den = np.zeros(v.shape[1]); cnt = np.zeros(v.shape[1])
    last_p, last_s = None, None
    for i in range(len(v)):
        p, s_ = pid[i], season[i]
        if p != last_p:
            num[:] = 0; den[:] = 0; cnt[:] = 0
        elif s_ != last_s:
            f = carry ** (s_ - last_s)
            num *= f; den *= f
        ok = (cnt >= min_periods) & (den > 0)
        out[i, ok] = num[ok] / den[ok]
        x = v[i]; m = ~np.isnan(x)
        num *= a; den *= a
        num[m] += x[m]; den[m] += 1; cnt[m] += 1
        last_p, last_s = p, s_
    return out

def player_history():
    import nba_props as NP
    pg, tg = NP.load()
    pg = pg.copy()
    pg["date"] = pd.to_datetime(pg["date"])
    pg = pg.sort_values(["player_id", "date"], kind="stable")
    played = (pg.dnp == 0) & (pg["min"] > 0)
    pg["played"] = played.astype(int)
    g = pg.groupby("player_id", sort=False)
    # minutes windows (only games played count toward rates; DNPs count toward availability)
    pgp = pg[played].copy()
    gp = pgp.groupby("player_id", sort=False)
    pid, sea = pgp.player_id.values, pgp.season.astype(int).values
    def ew(cols, hl, mp=1):
        return ewm_season(pgp[cols].values if isinstance(cols, list) else cols, pid, sea, hl, min_periods=mp)
    out = pd.DataFrame(index=pgp.index)
    m = ew(["min", "starter"], 3); out["min_e3"] = m[:, 0]
    out["min_e10"] = ew(["min"], 10)[:, 0]
    out["min_last"] = gp["min"].shift(1)
    out["start_e5"] = ew(["starter"], 5)[:, 0]
    out["n_prev"] = gp.cumcount()
    gs = pgp.groupby(["player_id", "season"], sort=False)
    out["n_prev_season"] = gs.cumcount()
    # this season so far (NaN before his first game of the season) and last season, as plain averages, so the
    # model can see a role change directly instead of only through blended windows
    out["min_cs"] = gs["min"].transform(lambda x: x.shift(1).expanding().mean())
    ly = pgp.groupby(["player_id", "season"])["min"].mean()
    out["min_ly"] = [ly.get((p, s - 1), np.nan) for p, s in zip(pid, sea)]
    pm = pd.DataFrame({st: pgp[st] / pgp["min"].clip(lower=1) for st in MKS}, index=pgp.index)
    e5, e20 = ew(pm[MKS].values, 5, 1), ew(pm[MKS].values, 20, 1)
    r5, r15 = ew(MKS, 5), ew(MKS, 15)
    tot_cs = pgp[MKS + ["min"]].groupby([pgp.player_id, pgp.season]).transform(lambda x: x.shift(1).cumsum())
    tot_ly = pgp.groupby(["player_id", "season"])[MKS + ["min"]].sum()
    for j, st in enumerate(MKS):
        out[f"{st}_pm_e5"], out[f"{st}_pm_e20"] = e5[:, j], e20[:, j]
        out[f"{st}_e5"], out[f"{st}_e15"] = r5[:, j], r15[:, j]
        out[f"{st}_pm_cs"] = tot_cs[st] / tot_cs["min"].clip(lower=1)
        d = {(p, s): v for (p, s), v in (tot_ly[st] / tot_ly["min"].clip(lower=1)).items()}
        out[f"{st}_pm_ly"] = [d.get((p, s - 1), np.nan) for p, s in zip(pid, sea)]
    # spread of the stat around its recent mean (per-player overdispersion)
    dev = (pgp[MKS].values - r15) ** 2
    sd = np.sqrt(ewm_season(dev, pid, sea, 20, min_periods=3))
    for j, st in enumerate(MKS):
        out[f"{st}_sd"] = sd[:, j]
    keep = pgp[["player_id", "game_id", "date", "season", "team", "opp", "home", "starter", "min"] + MKS]
    hist = pd.concat([keep, out], axis=1)
    return pg, tg, hist

def team_context(pg, tg):
    """Per team-game: minutes and usage missing (rotation players who didn't play), rest, pace and opponent allowances."""
    pg = pg.copy()
    pg = pg.sort_values("date")
    # a player's usual minutes entering the game (ewm of minutes over games he played, carried through DNPs)
    pg["min_play"] = np.where(pg.played == 1, pg["min"], np.nan)
    pg["usual_min"] = pg.groupby("player_id")["min_play"].transform(lambda x: x.shift(1).ewm(halflife=8, min_periods=1).mean())
    pg["usual_pts"] = pg.groupby("player_id")["pts"].transform(lambda x: x.where(pg.loc[x.index, "played"] == 1).shift(1).ewm(halflife=8, min_periods=1).mean())
    miss = pg[(pg.played == 0) & (pg.usual_min >= 15)]
    m = miss.groupby(["game_id", "team"]).agg(out_min=("usual_min", "sum"), out_pts=("usual_pts", "sum"), out_n=("player_id", "size")).reset_index()
    t = tg.copy()
    t["date"] = pd.to_datetime(t["date"])
    t = t.sort_values("date")
    t = t.merge(m, on=["game_id", "team"], how="left").fillna({"out_min": 0, "out_pts": 0, "out_n": 0})
    for c in ["poss", "pts", "opp_pts", "reb", "ast", "tpm", "stl", "blk"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.sort_values(["team", "date"], kind="stable")
    tsea = pd.to_numeric(t.season, errors="coerce").fillna(0).astype(int).values
    t["pace_e"] = ewm_season(t[["poss"]].values, t.team.values, tsea, 15)[:, 0]
    # what this team allows: opponent's stats in its games
    opp = t[["game_id", "team", "pts", "reb", "ast", "tpm", "stl", "blk"]].rename(columns={"team": "opp"})
    opp.columns = ["game_id", "opp"] + [f"a_{c}" for c in ["pts", "reb", "ast", "tpm", "stl", "blk"]]
    t = t.merge(opp, on=["game_id", "opp"], how="left")
    t = t.sort_values(["team", "date"], kind="stable")
    tsea = pd.to_numeric(t.season, errors="coerce").fillna(0).astype(int).values
    ac = [f"a_{c}" for c in ["pts", "reb", "ast", "tpm", "stl", "blk"]]
    al = ewm_season(t[ac].apply(pd.to_numeric, errors="coerce").values, t.team.values, tsea, 15)
    for j, c in enumerate(["pts", "reb", "ast", "tpm", "stl", "blk"]):
        t[f"allow_{c}"] = al[:, j]
    return t[["game_id", "team", "opp", "rest", "out_min", "out_pts", "out_n", "pace_e"] + [f"allow_{c}" for c in ["pts", "reb", "ast", "tpm", "stl", "blk"]]]

def build_features():
    from props_backtest import load_lines
    lines = pd.DataFrame(load_lines("nba", [2025, 2026]))
    pg, tg, hist = player_history()
    tc = team_context(pg, tg)
    lc = pd.read_csv("data/nba/lines_close.csv", dtype={"game_id": str})
    old = pd.read_csv(f"{H}/nba_backtest_picks.csv.gz", dtype={"game_id": str, "player": str})
    old = old[["game_id", "player", "market", "line", "p_over"]].rename(columns={"player": "athlete_id", "p_over": "sim_p"})
    lines["game_id"] = lines.game_id.astype(str); lines["athlete_id"] = lines.athlete_id.astype(str)
    df = lines.merge(hist, left_on=["game_id", "athlete_id"], right_on=["game_id", "player_id"], how="inner")
    df = df.merge(tc, on=["game_id", "team", "opp"], how="left")
    # teammates out for the player's team, opponent's pace and allowances
    oppc = tc.rename(columns={"team": "opp2", "opp": "team2"})[["game_id", "opp2", "pace_e", "out_min"]].rename(columns={"opp2": "opp", "pace_e": "opp_pace", "out_min": "opp_out_min"})
    df = df.merge(oppc, on=["game_id", "opp"], how="left")
    df = df.merge(lc[["game_id", "home_spread", "total"]], on="game_id", how="left")
    df["spread"] = np.where(df.home == 1, df.home_spread, -df.home_spread)
    df["implied_tt"] = df.total / 2 - df.spread / 2
    df = df.merge(old, on=["game_id", "athlete_id", "market", "line"], how="left")
    # market probability (power devig) and the stat-specific columns
    po, pu = df.over.map(implied), df.under.map(implied)
    df["p_mkt"] = [power_devig(a, b) for a, b in zip(po, pu)]
    df["y"] = [r[m] > r["line"] for r, m in zip(df[MKS + ["line"]].to_dict("records"), df.market)]
    df["y"] = df["y"].astype(int)
    rows = []
    for mk in MKS:
        sub = df[df.market == mk].copy()
        base = mk if mk in STATS else None
        sub["s_e5"], sub["s_e15"], sub["s_sd"] = sub[f"{mk}_e5"], sub[f"{mk}_e15"], sub[f"{mk}_sd"]
        sub["s_pm_e5"], sub["s_pm_e20"] = sub[f"{mk}_pm_e5"], sub[f"{mk}_pm_e20"]
        parts = COMBO.get(mk, [mk])
        sub["s_allow"] = sum(sub[f"allow_{p}"] for p in parts)
        rows.append(sub)
    df = pd.concat(rows)
    df["proj"] = df.s_pm_e5 * 0.5 + df.s_pm_e20 * 0.5
    df["proj"] = df["proj"] * (0.6 * df.min_e3 + 0.4 * df.min_e10)
    df["z_proj"] = (df.proj - df.line) / df.s_sd.clip(lower=0.5)
    df["z_e5"] = (df.s_e5 - df.line) / df.s_sd.clip(lower=0.5)
    df["z_e15"] = (df.s_e15 - df.line) / df.s_sd.clip(lower=0.5)
    df["line_rel"] = df.line / df.s_e15.clip(lower=0.3)
    df["min_trend"] = df.min_e3 - df.min_e10
    df["min_vs_last"] = df.min_last - df.min_e10
    df["b2b"] = (pd.to_numeric(df.rest, errors="coerce") == 0).astype(int)
    df["abs_spread"] = df.spread.abs()
    keep = ["season_x", "date_x", "game_id", "athlete_id", "market", "line", "over", "under", "p_mkt", "sim_p", "y",
            "proj", "z_proj", "z_e5", "z_e15", "line_rel", "s_sd", "s_e5", "s_e15", "s_pm_e5", "s_pm_e20", "s_allow",
            "min_e3", "min_e10", "min_last", "min_trend", "min_vs_last", "start_e5", "n_prev", "n_prev_season",
            "home", "rest", "b2b", "out_min", "out_pts", "out_n", "opp_out_min", "pace_e", "opp_pace",
            "spread", "abs_spread", "total", "implied_tt", "n_lines"]
    df = df[keep].rename(columns={"date_x": "date", "season_x": "season"})
    df = df[df.p_mkt.notna()]
    df.to_csv(f"{H}/nba_features.csv.gz", index=False, compression="gzip")
    print(len(df), "rows", df.groupby("season").size().to_dict())

# ------------------------------------------------------------------ walk-forward
FEATS = ["lp_mkt", "lsim", "z_proj", "z_e5", "z_e15", "line_rel", "s_sd", "min_e3", "min_e10", "min_trend", "min_vs_last",
         "start_e5", "n_prev", "n_prev_season", "home", "rest", "b2b", "out_min", "out_pts", "out_n", "opp_out_min",
         "pace_e", "opp_pace", "spread", "abs_spread", "total", "implied_tt", "s_allow", "mk_code", "line", "n_lines"]

def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))

def prep(df):
    df = df.copy()
    df["lp_mkt"] = logit(df.p_mkt)
    df["lsim"] = logit(df.sim_p.fillna(df.p_mkt))
    df["mk_code"] = df.market.map({m: i for i, m in enumerate(MKS)})
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df.date.dt.to_period("M")
    return df

def walk(params=None, min_train=15000, label="v2"):
    import lightgbm as lgb
    df = prep(pd.read_csv(f"{H}/nba_features.csv.gz", dtype={"game_id": str, "athlete_id": str}))
    months = sorted(df.month.unique())
    P = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=400, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7)
    if params:
        P.update(params)
    preds = []
    for m in months:
        tr = df[df.month < m]
        te = df[df.month == m]
        if len(tr) < min_train or te.empty:
            continue
        # init_score = market logit, so trees only learn corrections to the market
        dtr = lgb.Dataset(tr[FEATS], tr.y, init_score=tr.lp_mkt)
        bst = lgb.train(P, dtr, num_boost_round=300)
        raw = bst.predict(te[FEATS], raw_score=True) + te.lp_mkt.values
        out = te[["season", "date", "game_id", "athlete_id", "market", "line", "over", "under", "p_mkt", "y"]].copy()
        out["p"] = 1 / (1 + np.exp(-raw))
        preds.append(out)
        print(f"  {m}: trained on {len(tr)}, predicted {len(te)}")
    res = pd.concat(preds)
    res.to_csv(f"{H}/nba_{label}_walk.csv.gz", index=False, compression="gzip")
    report(res)
    return res

def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

def bets(res, ev_min=0.0):
    r = res.copy()
    dov, dun = r.over.map(decimal), r.under.map(decimal)
    r["ev_over"] = r.p * dov - 1
    r["ev_under"] = (1 - r.p) * dun - 1
    r["side"] = np.where(r.ev_over >= r.ev_under, "over", "under")
    r["ev"] = r[["ev_over", "ev_under"]].max(axis=1)
    r["win"] = np.where(r.side == "over", r.y, 1 - r.y)
    r["profit"] = np.where(r.win == 1, np.where(r.side == "over", dov - 1, dun - 1), -1.0)
    return r[r.ev >= ev_min]

def report(res):
    print(f"log loss  market {ll(res.p_mkt, res.y):.5f}   model {ll(res.p, res.y):.5f}   (lower is better; n={len(res)})")
    for thr in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
        b = bets(res, thr)
        if len(b) < 50:
            continue
        by = {int(s): (len(x), round(x.profit.mean(), 3)) for s, x in b.groupby("season")}
        print(f"  EV>={thr:.2f}: {len(b):6d} bets  win {b.win.mean():.3f}  ROI {b.profit.mean():+.3f}  per day {len(b)/res.date.nunique():.1f}  by season {by}")

if __name__ == "__main__":
    if sys.argv[1] == "features":
        build_features()
    elif sys.argv[1] == "walk":
        walk()
