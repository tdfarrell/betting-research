#!/usr/bin/env python3
"""NFL player prop cards: hit rates first, model probability alongside. No prices.

  python scripts/nfl_props.py                 cards for every week on the DK board -> data/props/
  python scripts/nfl_props.py backtest 2023 2025   calibration test -> data/props/backtest_summary.json

Hit rates (lead the card), per DraftKings-style tier:
  L3      last 3 games this season         Season   this season so far
  Def     the opponent's current-season rate allowed to the same role
Roles are set ENTERING each game (season-to-date per-game volume, last season in Week 1):
  WR1-3 / TE1 by targets per game, RB1-2 by carries+targets per game (season total / (games + 1), so one
  big game can't outrank a steady starter), QB1 = projected starter.
Model (shown alongside, never replacing hit rates):
  volume   player's share of team targets / carries x projected team pass attempts / rushes;
           team volume from its own and its opponent's averages, the DK spread and the team's
           implied total (from DK spread/total)
  efficiency  catch rate, yards per catch / carry / completion, shrunk toward position averages
           and scaled by what this defense allows to the position
  TDs      team TDs from the implied total x player's share of red-zone / goal-line opportunity
  Prior season: weight fades from Week 1 to zero by Week 6, and is dropped when the player changed
  teams, the team's QB or head coach changed, or his target share moved 5+ points.
  Probabilities come from simulating each player's game 4,000 times.
"""
import csv, datetime as dt, json, math, os, sys
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from nfl_model import prior_weight

OUT = "data/props"
TIERS = {
    "rec_yds": [25, 40, 50, 60, 70, 80, 90, 100],
    "rush_yds": [25, 40, 50, 60, 70, 80, 90, 100],
    "rr_yds": [40, 50, 60, 70, 80, 90, 100, 110, 125],
    "pass_yds": [180, 200, 220, 240, 260, 280, 300],
    "rec": [2, 3, 4, 5, 6, 7, 8],
    "pass_td": [1, 2, 3],
    "atd": [1],
}
LABEL = {"rec_yds": "Receiving yards", "rush_yds": "Rushing yards", "rr_yds": "Rush + rec yards", "pass_yds": "Passing yards",
         "rec": "Receptions", "pass_td": "Passing TDs", "atd": "Anytime TD"}
STAT = {"rec_yds": "receiving_yards", "rush_yds": "rushing_yards", "rr_yds": "rr_yds", "pass_yds": "passing_yards",
        "rec": "receptions", "pass_td": "passing_tds", "atd": "tds"}
MARKETS = {"QB": ["pass_yds", "pass_td", "rush_yds", "atd"], "RB": ["rush_yds", "rr_yds", "rec_yds", "rec", "atd"],
           "WR": ["rec_yds", "rec", "rr_yds", "atd"], "TE": ["rec_yds", "rec", "rr_yds", "atd"]}
ROLE_SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
SUMS = ["targets", "receptions", "receiving_yards", "carries", "rushing_yards", "attempts", "completions", "passing_yards",
        "passing_tds", "rz_tgt", "rz_car", "gl_car", "rushing_tds", "receiving_tds", "tds", "eq",
        "tm_targets", "tm_carries", "tm_eq", "tm_attempts"]

# single-play spread, measured from 2024-25 play-by-play
CATCH_CV = {"WR": 0.85, "TE": 0.78, "RB": 1.06, "QB": 1.0}
CARRY_SD = {"RB": 6.25, "QB": 5.3, "WR": 6.8, "TE": 7.6}
COMP_CV = 0.89
# game-to-game volume spread (negative binomial size; larger = steadier). Tuned in the backtest.
DISP = {"targets": 10.0, "carries": 5.0, "attempts": 25.0}
PASS_YDS_SCALE = 1.0  # tuned in the backtest
N_SIM = 4000
SHARE_K, PRIOR_GAMES = 2.0, 4.0

def num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

# ------------------------------------------------------------------ data
def load():
    pg = pd.read_csv("data/nfl/player_games.csv", low_memory=False)
    pg = num(pg, ["attempts", "completions", "passing_yards", "passing_tds", "interceptions", "sacks", "carries", "rushing_yards",
                  "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds", "rz_tgt", "rz_car", "gl_car", "rz_pass_att"])
    pg["rr_yds"] = pg.receiving_yards + pg.rushing_yards
    pg["tds"] = pg.rushing_tds + pg.receiving_tds
    sched = pd.read_csv("data/nfl/schedule.csv", low_memory=False)
    sched = sched[sched.game_type == "REG"]
    return pg, sched

def fit_eq(pg):
    """League TD rate per opportunity type (goal-line carry, other red-zone carry, other carry, RZ target, other target)."""
    X = np.c_[pg.gl_car, pg.rz_car - pg.gl_car, pg.carries - pg.rz_car, pg.rz_tgt, pg.targets - pg.rz_tgt]
    b = np.linalg.lstsq(X, pg.tds.values, rcond=None)[0]
    return np.clip(b, 0.002, None)

def add_team_cols(pg, eq):
    pg["eq"] = eq[0] * pg.gl_car + eq[1] * (pg.rz_car - pg.gl_car) + eq[2] * (pg.carries - pg.rz_car) + eq[3] * pg.rz_tgt + eq[4] * (pg.targets - pg.rz_tgt)
    tm = pg.groupby(["game_id", "team"]).agg(tm_targets=("targets", "sum"), tm_carries=("carries", "sum"), tm_eq=("eq", "sum"),
                                              tm_attempts=("attempts", "sum"), tm_pass_td=("passing_tds", "sum"), tm_td=("tds", "sum"),
                                              tm_sacks=("sacks", "sum")).reset_index()
    return pg.merge(tm, on=["game_id", "team"], how="left"), tm

def team_games(pg, tm, sched):
    """Team-game rows with implied totals from the closing line (history) for the volume regressions."""
    g = tm.merge(pg[["game_id", "team", "opp", "season", "week"]].drop_duplicates(["game_id", "team"]), on=["game_id", "team"])
    s = sched.set_index("game_id")
    def implied(r):
        if r.game_id not in s.index:
            return pd.Series([np.nan, np.nan])
        x = s.loc[r.game_id]; sp, tot = x.spread_line, x.total_line
        if pd.isna(sp) or pd.isna(tot):
            return pd.Series([np.nan, np.nan])
        home = r.team == x.home_team
        mine = tot / 2 + (sp / 2 if home else -sp / 2)
        return pd.Series([mine, -sp if home else sp])  # implied points, team spread (- = favored)
    g[["implied", "spread"]] = g.apply(implied, axis=1)
    return g

# ------------------------------------------------------------------ entering-game state
class Week:
    """Everything known before week W of season S."""
    def __init__(self, D, S, W):
        pg, sched = D["pg"], D["sched"]
        self.S, self.W = S, W
        self.cur = pg[(pg.season == S) & (pg.week < W)]
        self.pri = pg[pg.season == S - 1]
        self.pw = prior_weight(W) * PRIOR_GAMES
        hist = pg[(pg.season < S) | ((pg.season == S) & (pg.week < W))]
        self.lg = league(hist[hist.season >= S - 3])
        self.C = agg(self.cur); self.P = agg(self.pri)
        # coach / QB continuity per team (for dropping last season)
        sc = sched[sched.season.isin([S - 1, S])]
        self.coach = {}
        for _, r in sc.iterrows():
            for side in ("home", "away"):
                self.coach[(r.season, r[f"{side}_team"])] = r[f"{side}_coach"]
        qb = pg[(pg.season.isin([S - 1, S])) & (pg.position == "QB")]
        qcur = self.cur[self.cur.position == "QB"].groupby(["team", "player_id"]).attempts.sum().reset_index()
        qpri = self.pri[self.pri.position == "QB"].groupby(["team", "player_id"]).attempts.sum().reset_index()
        self.qb_pri = qpri.sort_values("attempts").groupby("team").player_id.last().to_dict()
        self.qb_cur = qcur.sort_values("attempts").groupby("team").player_id.last().to_dict()
        self.team = team_avgs(D, S, W)
        self.defense = def_factors(self, D)

def agg(df):
    if df.empty:
        return pd.DataFrame(columns=SUMS + ["g", "team", "position"]).set_index(pd.Index([], name="player_id"))
    a = df.groupby("player_id")[SUMS].sum()
    a["g"] = df.groupby("player_id").size()
    last = df.sort_values(["season", "week"]).groupby("player_id").last()
    a["team"] = last.team; a["position"] = last.position; a["name"] = last.name
    return a

def league(h):
    out = {}
    for pos in ("WR", "TE", "RB", "QB"):
        x = h[h.position == pos]
        out[pos] = {"cr": x.receptions.sum() / max(x.targets.sum(), 1), "ypr": x.receiving_yards.sum() / max(x.receptions.sum(), 1),
                    "ypc": x.rushing_yards.sum() / max(x.carries.sum(), 1)}
    q = h[h.position == "QB"]
    out["pass"] = {"cmp": q.completions.sum() / max(q.attempts.sum(), 1), "ypcmp": q.passing_yards.sum() / max(q.completions.sum(), 1)}
    return out

def team_avgs(D, S, W):
    """Per-game team pass attempts / rushes / TDs for, and allowed, entering week W (last season fades)."""
    tg = D["tg"]
    pw = prior_weight(W) * PRIOR_GAMES
    cur = tg[(tg.season == S) & (tg.week < W)]; pri = tg[tg.season == S - 1]
    out = {}
    for side, key in (("for", "team"), ("allowed", "opp")):
        c = cur.groupby(key).agg(g=("game_id", "size"), att=("tm_attempts", "sum"), car=("tm_carries", "sum"), td=("tm_td", "sum"), ptd=("tm_pass_td", "sum"))
        p = pri.groupby(key).agg(g=("game_id", "size"), att=("tm_attempts", "sum"), car=("tm_carries", "sum"), td=("tm_td", "sum"), ptd=("tm_pass_td", "sum"))
        teams = set(c.index) | set(p.index)
        for t in teams:
            gc = c.g.get(t, 0); gp = p.g.get(t, 0)
            w = pw if gp else 0
            res = {}
            for k in ("att", "car", "td", "ptd"):
                cv = c[k].get(t, 0) / gc if gc else 0; pv = p[k].get(t, 0) / gp if gp else 0
                res[k] = (gc * cv + w * pv) / (gc + w) if gc + w else np.nan
            out[(side, t)] = res
    return out

def def_factors(wk, D):
    """How a defense does vs each position group, as a multiplier on league average (shrunk)."""
    pg = D["pg"]
    cur = pg[(pg.season == wk.S) & (pg.week < wk.W)]; pri = pg[pg.season == wk.S - 1]
    out = {}
    for pos in ("WR", "TE", "RB"):
        lg = wk.lg[pos]
        c = cur[cur.position == pos].groupby("opp")[["targets", "receptions", "receiving_yards", "carries", "rushing_yards"]].sum()
        p = pri[pri.position == pos].groupby("opp")[["targets", "receptions", "receiving_yards", "carries", "rushing_yards"]].sum()
        gcur = cur.groupby("opp").game_id.nunique(); gpri = pri.groupby("opp").game_id.nunique()
        for t in set(c.index) | set(p.index):
            sc = prior_weight(wk.W) * PRIOR_GAMES / max(gpri.get(t, 1), 1) if t in p.index else 0
            v = {k: (c[k].get(t, 0) if t in c.index else 0) + sc * (p[k].get(t, 0) if t in p.index else 0) for k in c.columns}
            out[(t, pos)] = {
                "cr": ((v["receptions"] + 60 * lg["cr"]) / (v["targets"] + 60)) / lg["cr"],
                "ypr": ((v["receiving_yards"] + 40 * lg["ypr"]) / (v["receptions"] + 40)) / lg["ypr"],
                "ypc": ((v["rushing_yards"] + 80 * lg["ypc"]) / (v["carries"] + 80)) / lg["ypc"] if pos == "RB" else 1.0}
    q = pg[pg.position == "QB"]
    c = cur[cur.position == "QB"].groupby("opp")[["attempts", "completions", "passing_yards"]].sum()
    p = pri[pri.position == "QB"].groupby("opp")[["attempts", "completions", "passing_yards"]].sum()
    gpri = pri.groupby("opp").game_id.nunique()
    L = wk.lg["pass"]
    for t in set(c.index) | set(p.index):
        sc = prior_weight(wk.W) * PRIOR_GAMES / max(gpri.get(t, 1), 1) if t in p.index else 0
        v = {k: (c[k].get(t, 0) if t in c.index else 0) + sc * (p[k].get(t, 0) if t in p.index else 0) for k in c.columns}
        out[(t, "QB")] = {"cmp": ((v["completions"] + 150 * L["cmp"]) / (v["attempts"] + 150)) / L["cmp"],
                          "ypcmp": ((v["passing_yards"] + 100 * L["ypcmp"]) / (v["completions"] + 100)) / L["ypcmp"]}
    return out

# ------------------------------------------------------------------ roles
def volume_score(wk, pid, pos):
    for A in (wk.C, wk.P):
        if pid in A.index and A.at[pid, "g"] > 0:
            r = A.loc[pid]
            v = r.attempts if pos == "QB" else (r.carries + r.targets if pos == "RB" else r.targets)
            return v / (r.g + 1)  # per game, with one empty game added so a single big game can't outrank a steady starter
    return 0.0

def assign_roles(wk, cands, qb1=None):
    """cands: DataFrame(team, player_id, position). Returns {player_id: role}."""
    roles = {}
    for (team, pos), grp in cands.groupby(["team", "position"]):
        if pos not in ROLE_SLOTS:
            continue
        sc = sorted(((volume_score(wk, p, pos), p) for p in grp.player_id), reverse=True)
        if pos == "QB" and qb1 and qb1.get(team) in set(grp.player_id):
            sc = [(9e9, qb1[team])] + [x for x in sc if x[1] != qb1[team]]
        k = 0
        for s, p in sc:
            if s <= 0 and pos != "QB":
                continue
            k += 1
            if k > ROLE_SLOTS[pos]:
                break
            roles[p] = f"{pos}{k}"
    return roles

def season_roles(D, S, upto, prior_only=False):
    """Entering-game roles for every game of season S before week `upto` (for hit-rate splits).
    prior_only (backtests): the starting QB is last game's QB1, not who threw the most passes in this game,
    so the backtest doesn't know about in-game QB changes or surprise starts."""
    pg = D["pg"]; out = {}
    last_qb1 = {}
    for w in range(1, upto):
        games = pg[(pg.season == S) & (pg.week == w)]
        if games.empty:
            continue
        wk = Week(D, S, w) if (S, w) not in D["wk_cache"] else D["wk_cache"][(S, w)]
        D["wk_cache"][(S, w)] = wk
        this_qb1 = games[games.position == "QB"].sort_values("attempts").groupby("team").player_id.last().to_dict()
        qb1 = {t: last_qb1.get(t) for t in this_qb1} if prior_only else this_qb1
        last_qb1.update(this_qb1)
        for pid, role in assign_roles(wk, games[["team", "player_id", "position"]], qb1).items():
            out[(S, w, pid)] = role
    return out

# ------------------------------------------------------------------ projection
def blend_share(wk, pid, team, num_col, den_col, role_mean, drop):
    gc = wk.C.at[pid, "g"] if pid in wk.C.index else 0
    sc = (wk.C.at[pid, num_col] / wk.C.at[pid, den_col]) if gc and wk.C.at[pid, den_col] > 0 else 0.0
    w, sp = 0.0, 0.0
    if not drop and pid in wk.P.index and wk.P.at[pid, "g"] > 0 and wk.P.at[pid, den_col] > 0:
        w, sp = wk.pw, wk.P.at[pid, num_col] / wk.P.at[pid, den_col]
    return (gc * sc + w * sp + SHARE_K * role_mean) / (gc + w + SHARE_K)

def blend_rate(wk, pid, num_col, den_col, prior_mean, k, drop):
    n = wk.C.at[pid, num_col] if pid in wk.C.index else 0.0
    d = wk.C.at[pid, den_col] if pid in wk.C.index else 0.0
    if not drop and pid in wk.P.index and wk.P.at[pid, "g"] > 0:
        f = wk.pw / wk.P.at[pid, "g"]
        n += f * wk.P.at[pid, num_col]; d += f * wk.P.at[pid, den_col]
    return (n + k * prior_mean) / (d + k)

def prior_dropped(wk, pid, team, pos):
    """Reasons last season is ignored for this player (empty list = last season is used, fading)."""
    why = []
    if pid not in wk.P.index or wk.P.at[pid, "g"] == 0:
        return ["no prior season"]
    if wk.P.at[pid, "team"] != team:
        why.append("changed teams")
    if wk.coach.get((wk.S, team)) and wk.coach.get((wk.S - 1, team)) and wk.coach[(wk.S, team)] != wk.coach[(wk.S - 1, team)]:
        why.append("new head coach")
    if pos != "QB" and wk.qb_pri.get(team) and wk.qb_cur.get(team) and wk.qb_pri[team] != wk.qb_cur[team]:
        why.append("new QB")
    if pid in wk.C.index and wk.C.at[pid, "g"] >= 2 and wk.C.at[pid, "tm_targets"] > 0 and wk.P.at[pid, "tm_targets"] > 0:
        d = wk.C.at[pid, "targets"] / wk.C.at[pid, "tm_targets"] - wk.P.at[pid, "targets"] / wk.P.at[pid, "tm_targets"]
        if abs(d) >= 0.05:
            why.append(f"target share {'up' if d > 0 else 'down'} {abs(d) * 100:.0f} pts")
    return why

def project(wk, D, pid, pos, role, team, opp, implied, spread, coef):
    """Inputs for the simulation."""
    drop_why = prior_dropped(wk, pid, team, pos)
    drop = bool(drop_why)
    RM = D["role_means"].get(role, D["role_means"].get(pos + "1"))
    T, O = wk.team.get(("for", team), {}), wk.team.get(("allowed", opp), {})
    lg_att, lg_car, lg_td = D["lg_team"]
    x = lambda d, k, dflt: d.get(k) if d.get(k) is not None and not np.isnan(d.get(k)) else dflt
    imp = implied if implied is not None and not np.isnan(implied) else lg_td * 7 * 0.5 + 17
    spr = spread if spread is not None and not np.isnan(spread) else 0.0
    f_att = np.array([1, x(T, "att", lg_att), x(O, "att", lg_att), spr, imp]) @ coef["att"]
    f_car = np.array([1, x(T, "car", lg_car), x(O, "car", lg_car), spr, imp]) @ coef["car"]
    f_td = np.array([1, imp]) @ coef["td"]
    tshare_pass = (x(T, "ptd", 0) * 6 + 0.6 * 6) / (x(T, "td", 0) * 6 + 6) if x(T, "td", 0) else 0.6
    lg = wk.lg[pos]
    dfac = wk.defense.get((opp, pos if pos != "QB" else "RB"), {"cr": 1, "ypr": 1, "ypc": 1})
    p = {"pos": pos, "drop": drop_why, "team_att": f_att, "team_car": f_car, "team_td": f_td}
    tgt_per_att = 0.93
    p["mu_t"] = blend_share(wk, pid, team, "targets", "tm_targets", RM["tshare"], drop) * f_att * tgt_per_att if pos != "QB" else 0.0
    p["mu_c"] = blend_share(wk, pid, team, "carries", "tm_carries", RM["cshare"], drop) * f_car
    p["cr"] = min(0.95, blend_rate(wk, pid, "receptions", "targets", lg["cr"], 30, drop) * (dfac["cr"] if pos != "QB" else 1))
    p["ypr"] = blend_rate(wk, pid, "receiving_yards", "receptions", lg["ypr"], 20, drop) * (dfac["ypr"] if pos != "QB" else 1)
    p["ypc"] = blend_rate(wk, pid, "rushing_yards", "carries", lg["ypc"], 40, drop) * (dfac["ypc"] if pos == "RB" else 1)
    eq_share = blend_share(wk, pid, team, "eq", "tm_eq", RM["eqshare"], drop)
    p["lam_td"] = f_td * eq_share
    if pos == "QB":
        L, dq = wk.lg["pass"], wk.defense.get((opp, "QB"), {"cmp": 1, "ypcmp": 1})
        p["mu_a"] = f_att
        p["cmp"] = min(0.85, blend_rate(wk, pid, "completions", "attempts", L["cmp"], 100, drop) * dq["cmp"])
        p["ypcmp"] = blend_rate(wk, pid, "passing_yards", "completions", L["ypcmp"], 60, drop) * dq["ypcmp"]
        p["lam_ptd"] = f_td * tshare_pass
    return p

def negbin(rng, mu, r, n):
    mu = max(mu, 1e-6)
    return rng.poisson(rng.gamma(r, mu / r, n))

def gamma_sum(rng, count, mean, cv):
    """Sum of `count` plays each with mean `mean` and coefficient of variation `cv`."""
    k = 1 / cv ** 2
    out = np.zeros(len(count))
    m = count > 0
    out[m] = rng.gamma(count[m] * k, max(mean, 0.1) / k)
    return out

def simulate(p, rng, n=N_SIM, disp=None):
    d = disp or DISP
    pos = p["pos"]; res = {}
    tg = negbin(rng, p["mu_t"], d["targets"], n) if p["mu_t"] > 0 else np.zeros(n, int)
    rec = rng.binomial(tg, p["cr"])
    recy = gamma_sum(rng, rec, p["ypr"], CATCH_CV.get(pos, 0.9))
    car = negbin(rng, p["mu_c"], d["carries"], n) if p["mu_c"] > 0 else np.zeros(n, int)
    sd = CARRY_SD.get(pos, 6.25); sh = 3.0
    rushy = gamma_sum(rng, car, p["ypc"] + sh, sd / (p["ypc"] + sh)) - sh * car
    res.update(rec=rec, rec_yds=recy, rush_yds=rushy, rr_yds=recy + rushy)
    if pos == "QB":
        att = negbin(rng, p["mu_a"], d["attempts"], n)
        cmp = rng.binomial(att, p["cmp"])
        res["pass_yds"] = gamma_sum(rng, cmp, p["ypcmp"] * PASS_YDS_SCALE, COMP_CV)
    return res

def tier_probs(p, sims, mk):
    t = TIERS[mk]
    if mk == "atd":
        return [1 - math.exp(-p["lam_td"])]
    if mk == "pass_td":
        lam = p["lam_ptd"]
        return [1 - sum(math.exp(-lam) * lam ** j / math.factorial(j) for j in range(k)) for k in t]
    v = sims[mk]
    return [float((v >= k).mean()) for k in t]

def fair(p):
    if p is None or p <= 0.005 or p >= 0.995:
        return None
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))

# ------------------------------------------------------------------ fitting (team volume, role means)
def fit_coefs(D, upto_season):
    rows = D["team_feat"]
    rows = rows[(rows.season < upto_season) & rows.implied.notna() & rows.t_att.notna() & rows.o_att.notna()]
    out = {}
    for k, y, a, b in (("att", "tm_attempts", "t_att", "o_att"), ("car", "tm_carries", "t_car", "o_car")):
        X = np.c_[np.ones(len(rows)), rows[a], rows[b], rows.spread, rows.implied]
        out[k] = np.linalg.lstsq(X, rows[y].values, rcond=None)[0]
    X = np.c_[np.ones(len(rows)), rows.implied]
    out["td"] = np.linalg.lstsq(X, rows.tm_td.values, rcond=None)[0]
    return out

def build_team_features(D, seasons):
    """For every team-game: entering averages (for/allowed) + implied total + spread + actual volume."""
    feats = []
    tg = D["tg"]
    for S in seasons:
        for W in sorted(tg[tg.season == S].week.unique()):
            ta = team_avgs(D, S, W)
            for _, r in tg[(tg.season == S) & (tg.week == W)].iterrows():
                T, O = ta.get(("for", r.team), {}), ta.get(("allowed", r.opp), {})
                feats.append({"season": S, "week": W, "team": r.team, "implied": r.implied, "spread": r.spread,
                              "tm_attempts": r.tm_attempts, "tm_carries": r.tm_carries, "tm_td": r.tm_td,
                              "t_att": T.get("att"), "o_att": O.get("att"), "t_car": T.get("car"), "o_car": O.get("car")})
    return pd.DataFrame(feats)

def role_means(D, upto_season, roles):
    """Average share by role (entering-game roles), from seasons before `upto_season`."""
    pg = D["pg"]
    rr = pd.Series({k: v for k, v in roles.items() if k[0] < upto_season})
    if rr.empty:
        return {}
    idx = pd.MultiIndex.from_tuples(rr.index, names=["season", "week", "player_id"])
    r = pd.DataFrame({"role": rr.values}, index=idx).reset_index()
    x = pg.merge(r, on=["season", "week", "player_id"])
    out = {}
    for role, g in x.groupby("role"):
        out[role] = {"tshare": g.targets.sum() / max(g.tm_targets.sum(), 1), "cshare": g.carries.sum() / max(g.tm_carries.sum(), 1),
                     "eqshare": g["eq"].sum() / max(g["tm_eq"].sum(), 1e-9)}
    return out

def setup(seasons_needed):
    pg, sched = load()
    eq = fit_eq(pg[pg.season < pg.season.max()])
    pg, tm = add_team_cols(pg, eq)
    D = {"pg": pg, "sched": sched, "eq": eq, "wk_cache": {}}
    D["tg"] = team_games(pg, tm, sched)
    lt = D["tg"][D["tg"].season >= min(seasons_needed) - 2]
    D["lg_team"] = (lt.tm_attempts.mean(), lt.tm_carries.mean(), lt.tm_td.mean())
    D["team_feat"] = build_team_features(D, sorted(set(D["tg"].season.unique()) & set(range(min(seasons_needed) - 3, max(seasons_needed) + 1))))
    return D

# ------------------------------------------------------------------ hit rates
def hits(vals, tiers):
    return [[int(sum(1 for v in vals if v >= t)), len(vals)] for t in tiers]

def disagreement(mk, tiers, probs, season, l3):
    """Flag tiers where hit rates and the model clearly disagree (needs a real sample)."""
    out = []
    min_n = 5 if mk in ("atd", "pass_td") else 4
    for t, p, (h, n), (h3, n3) in zip(tiers, probs, season, l3):
        lab = f" {t}+" if mk != "atd" else ""
        # only when they point opposite ways (one says likely, the other unlikely) by a wide margin
        if n >= min_n and abs(h / n - p) >= 0.30 and (h / n - 0.5) * (p - 0.5) < 0:
            out.append(f"{lab}: hit {h}/{n} this season, model {p:.0%}")
        elif n3 == 3 and ((h3 == 3 and p < 0.35) or (h3 == 0 and p > 0.65)):
            out.append(f"{lab}: {h3}/3 last 3, model {p:.0%}")
    return out

# ------------------------------------------------------------------ live cards
def current_week_cards(D):
    sched, pg = D["sched"], D["pg"]
    board = json.load(open("data/odds/latest_nfl.json"))
    roster = pd.read_csv("data/nfl/roster_current.csv")
    inj = pd.read_csv("data/nfl/injuries.csv") if os.path.exists("data/nfl/injuries.csv") else pd.DataFrame(columns=["gsis_id", "report_status"])
    inj_status = {r.gsis_id: r.report_status for r in inj.itertuples() if isinstance(getattr(r, "report_status", None), str)}
    s_by_espn = {str(r.espn): r for r in sched.itertuples() if not pd.isna(r.espn)}
    S = int(sched[sched.home_score.notna()].season.max())
    by_week = defaultdict(list)
    for g in board["games"]:
        r = s_by_espn.get(str(g["game_id"]))
        if r is not None and g.get("status") == "pre":
            by_week[(int(r.season), int(r.week))].append((g, r))
    coef = fit_coefs(D, S + 1)
    os.makedirs(OUT, exist_ok=True)
    idx = {"generated_utc": now(), "weeks": []}
    for (S_, W), games in sorted(by_week.items()):
        roles_hist = season_roles(D, S_, W)
        if "roles_prev" not in D:
            D["roles_prev"] = {**season_roles(D, S_ - 2, 19), **season_roles(D, S_ - 1, 19)}
        D["role_means"] = role_means(D, S_, D["roles_prev"])
        wk = D["wk_cache"].get((S_, W)) or Week(D, S_, W)
        rng = np.random.default_rng(S_ * 100 + W)
        D["si"] = D.get("si") or sched_info(sched)
        mates = team_mates(pg, S_, W, inj_status, roster)
        played = played_index(pg, [S_ - 1, S_])
        dvp = dvp_ranks(wk, roles_hist)
        cards = []
        for g, r in games:
            for team, opp, home in ((r.home_team, r.away_team, True), (r.away_team, r.home_team, False)):
                dk_hs, dk_tot = num1(g.get("home_spread")), num1(g.get("total"))
                if dk_hs is not None and dk_tot is not None:
                    spr = dk_hs if home else -dk_hs
                    implied = dk_tot / 2 - spr / 2
                else:
                    sp, tot = r.spread_line, r.total_line
                    implied = tot / 2 + (sp / 2 if home else -sp / 2) if not pd.isna(tot) else None
                    spr = (-sp if home else sp) if not pd.isna(sp) else None
                cand = roster[(roster.team == team) & (roster.status == "ACT") & roster.position.isin(list(ROLE_SLOTS))]
                if cand.empty:
                    print(f"  no roster rows for {team}; skipped"); continue
                cand = cand[~cand.player_id.map(lambda p: inj_status.get(p) == "Out").astype(bool)]
                qb_id = r.home_qb_id if home else r.away_qb_id
                roles = assign_roles(wk, cand[["team", "player_id", "position"]], {team: qb_id} if isinstance(qb_id, str) else None)
                for pid, role in roles.items():
                    pr = cand[cand.player_id == pid].iloc[0]
                    c = make_card(D, wk, roles_hist, pid, pr["name"], pr.position, role, team, opp, home, implied, spr, coef, rng,
                                  g, r, headshot=pr.headshot if isinstance(pr.headshot, str) else "", status=inj_status.get(pid))
                    c["glog"] = game_log(pg, pid, S_, W, D["si"], mates.get(team, []), played)
                    for mk, m in c["markets"].items():
                        m["dvp"] = dvp.get((opp, role, mk))
                    cards.append(c)
        key = f"{S_}_wk{W:02d}"
        doc = {"key": key, "season": S_, "week": W, "generated_utc": now(), "odds_pulled_utc": board["pulled_utc"],
               "prior_weight": prior_weight(W), "tiers": TIERS, "labels": LABEL,
               "glog_cols": GLOG_COLS, "mates": mates, "cards": cards}
        path = f"{OUT}/nfl_props_{key}.json"
        if os.path.exists(path):   # keep players whose game already kicked off (frozen as they were)
            try:
                have = {(c["id"], c["game_id"]) for c in cards}
                nowz = now()
                for c in json.load(open(path)).get("cards", []):
                    if (c["id"], c["game_id"]) not in have and c.get("kickoff_utc", "9999") <= nowz:
                        c["started"] = True; cards.append(c)
                doc["cards"] = cards
            except Exception:
                pass
        with open(path, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
        idx["weeks"].append({"key": key, "season": S_, "week": W, "players": len(doc["cards"]), "flags": sum(1 for c in cards if c["flags"])})
        print(f"props {key}: {len(cards)} players, {idx['weeks'][-1]['flags']} with hit-rate/model disagreements")
    if os.path.exists("data/props/backtest_summary.json"):
        idx["backtest"] = json.load(open("data/props/backtest_summary.json")).get("headline")
    old = json.load(open(f"{OUT}/index.json")) if os.path.exists(f"{OUT}/index.json") else {"weeks": []}
    have = {w["key"] for w in idx["weeks"]}
    idx["weeks"] += [w for w in old.get("weeks", []) if w["key"] not in have and os.path.exists(f"{OUT}/nfl_props_{w['key']}.json")]
    idx["weeks"].sort(key=lambda w: w["key"])
    with open(f"{OUT}/index.json", "w") as f:
        json.dump(idx, f, indent=1)

def num1(x):
    try:
        return float(str(x).lstrip("ou"))
    except (TypeError, ValueError):
        return None

# ------------------------------------------------------------------ game logs, splits, defense vs position
GLOG_COLS = ["season", "wk", "team", "opp", "home", "rest", "div", "tgt", "rec", "rec_yds", "car", "rush_yds", "rr_yds",
             "att", "cmp", "pass_yds", "pass_td", "td", "snap", "out"]
DVP_STAT = {mk: STAT[mk] for mk in TIERS}

def sched_info(sched):
    """game_id -> (home team, home rest, away rest, divisional)."""
    return {r.game_id: (r.home_team, r.home_rest, r.away_rest, int(r.div_game) if not pd.isna(r.div_game) else 0)
            for r in sched.itertuples()}

def team_mates(pg, S, W, inj_status, roster, n=7):
    """Per team: the players a 'without X' filter is useful for. Top touches/game this season (or last season early on)
    plus the QB with the most attempts, including players who are Out this week."""
    cur = pg[(pg.season == S) & (pg.week < W)]
    if cur.empty:
        cur = pg[pg.season == S - 1]
    name = dict(zip(roster.player_id, roster["name"]))
    out = {}
    for team, grp in cur.groupby("team"):
        a = grp.groupby("player_id").agg(g=("game_id", "nunique"), tch=("targets", "sum"), car=("carries", "sum"),
                                        att=("attempts", "sum"), pos=("position", "last"), nm=("name", "last"))
        a["tpg"] = (a.tch + a.car) / a.g
        qb = a[a.pos == "QB"].sort_values("att").tail(1)
        sk = a[(a.pos != "QB") & (a.tpg >= 2.5)].sort_values("tpg", ascending=False).head(n - len(qb))
        rows = []
        for pid, r in pd.concat([qb, sk]).iterrows():
            rows.append({"id": pid, "name": name.get(pid, r.nm), "pos": r.pos, "tpg": round(float(r.tpg), 1),
                         "status": inj_status.get(pid)})
        out[team] = rows
    return out

def played_index(pg, seasons):
    """(game_id, team) -> set of player_ids who played; (season, team) -> set of players who played for that team."""
    g = pg[pg.season.isin(seasons)]
    snaps = pd.to_numeric(g.snaps, errors="coerce").fillna(0)
    touched = (g.targets + g.carries + g.attempts) > 0
    g = g[(snaps > 0) | touched]
    by_game = g.groupby(["game_id", "team"]).player_id.agg(set).to_dict()
    by_season = g.groupby(["season", "team"]).player_id.agg(set).to_dict()
    return by_game, by_season

def game_log(pg, pid, S, W, si, mates, played):
    """Every game this season (before W) and last season, compact rows in GLOG_COLS order."""
    by_game, by_season = played
    rows = pg[(pg.player_id == pid) & ((pg.season == S - 1) | ((pg.season == S) & (pg.week < W)))].sort_values(["season", "week"])
    out = []
    for x in rows.itertuples():
        h = si.get(x.game_id)
        home = int(h[0] == x.team) if h else None
        rest = None
        if h:
            rv = h[1] if home else h[2]
            rest = int(rv) if not pd.isna(rv) else None
        mask = 0
        on_field = by_game.get((x.game_id, x.team), set())
        for i, m in enumerate(mates):   # mates of the player's team this week
            if m["id"] == pid:
                continue
            # out = on this team that season but did not play this game
            if m["id"] in by_season.get((int(x.season), x.team), set()) and m["id"] not in on_field:
                mask |= 1 << i
        snap = pd.to_numeric(x.snap_pct, errors="coerce")
        out.append([int(x.season), int(x.week), x.team, x.opp, home, rest, h[3] if h else None,
                    int(x.targets), int(x.receptions), int(x.receiving_yards), int(x.carries), int(x.rushing_yards), int(x.rr_yds),
                    int(x.attempts), int(x.completions), int(x.passing_yards), int(x.passing_tds), int(x.tds),
                    None if pd.isna(snap) else round(float(snap), 2), mask])
    return out

def dvp_ranks(wk, roles_hist):
    """(opp, role, market) -> {rank 1-32 (32 = allows the most), avg allowed per game, games}. Current season only."""
    cur = wk.cur.copy()
    if cur.empty:
        return {}
    cur["role"] = [roles_hist.get((wk.S, int(w), q)) for w, q in zip(cur.week, cur.player_id)]
    cur = cur[cur.role.notna()]
    out = {}
    for mk, col in DVP_STAT.items():
        a = cur.groupby(["role", "opp"])[col].agg(["mean", "count"]).reset_index()
        for role, grp in a.groupby("role"):
            grp = grp.sort_values("mean")
            n = len(grp)
            # average rank for ties, 1 = fewest allowed
            rk = grp["mean"].rank(method="average")
            for (_, r), k in zip(grp.iterrows(), rk):
                out[(r.opp, role, mk)] = {"rank": round(float(k)), "of": n, "avg": round(float(r["mean"]), 1), "n": int(r["count"])}
    return out

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")

def make_card(D, wk, roles_hist, pid, name, pos, role, team, opp, home, implied, spread, coef, rng, g, r, headshot="", status=None):
    pg = D["pg"]
    p = project(wk, D, pid, pos, role, team, opp, implied, spread, coef)
    sims = simulate(p, rng)
    games = wk.cur[wk.cur.player_id == pid].sort_values("week")
    l3 = games.tail(3)
    # opponent's current-season results vs this role
    dv = wk.cur[(wk.cur.opp == opp)]
    dv = dv[[roles_hist.get((wk.S, int(w), q)) == role for w, q in zip(dv.week, dv.player_id)]].sort_values("week")
    mkts, flags = {}, []
    for mk in MARKETS[pos]:
        col, tiers = STAT[mk], TIERS[mk]
        probs = tier_probs(p, sims, mk)
        sv, l3v, dvv = games[col].tolist(), l3[col].tolist(), dv[col].tolist()
        hs, h3, hd = hits(sv, tiers), hits(l3v, tiers), hits(dvv, tiers)
        dis = disagreement(mk, tiers, probs, hs, h3)
        flags += [f"{LABEL[mk]}{d}" for d in dis]
        mkts[mk] = {"tiers": [{"t": t, "l3": a, "season": b, "def": c, "p": round(pr, 3), "fair": fair(pr)}
                              for t, a, b, c, pr in zip(tiers, h3, hs, hd, probs)],
                    "avg": round(float(np.mean(sv)), 1) if sv else None, "l3_vals": [int(v) for v in l3v],
                    "l3_opps": l3.opp.tolist(), "season_vals": [int(v) for v in sv], "def_avg": round(float(np.mean(dvv)), 1) if dvv else None,
                    "def_n": len(dvv), "proj": round(float(np.median(sims[mk])), 1) if mk in sims else
                    round(p["lam_td"] if mk == "atd" else p["lam_ptd"], 2), "disagree": dis}
        # full model distribution, so the page can price any DraftKings line (not just the tiers)
        if mk in ("rec_yds", "rush_yds", "rr_yds", "pass_yds") and mk in sims:
            mkts[mk]["q"] = [int(round(x)) for x in np.quantile(sims[mk], np.linspace(0, 1, 41))]
        elif mk == "rec" and mk in sims:
            mkts[mk]["ge"] = [round(float((sims[mk] >= k).mean()), 3) for k in range(16)]
    tgt_share = (games.targets.sum() / games.tm_targets.sum()) if len(games) and games.tm_targets.sum() else None
    car_share = (games.carries.sum() / games.tm_carries.sum()) if len(games) and games.tm_carries.sum() else None
    snap = pd.to_numeric(games.snap_pct, errors="coerce").dropna()
    k = dt.datetime.fromisoformat(g["kickoff_utc"].replace("Z", "+00:00")).astimezone(dt.timezone(dt.timedelta(hours=-4)))
    return {"id": pid, "name": name, "pos": pos, "role": role, "team": team, "opp": opp, "home": home, "game_id": g["game_id"],
            "kick_et": k.strftime("%a %-I:%M %p"), "kickoff_utc": g["kickoff_utc"], "headshot": headshot, "status": status,
            "games": int(len(games)), "implied": round(implied, 1) if implied is not None else None,
            "spread": spread, "usage": {"tgt_share": rnd(tgt_share, 3), "car_share": rnd(car_share, 3), "snap_pct": rnd(snap.mean(), 3) if len(snap) else None,
                                         "tgt_pg": rnd(games.targets.mean(), 1) if len(games) else None, "car_pg": rnd(games.carries.mean(), 1) if len(games) else None,
                                         "rz_opps": int(games.rz_tgt.sum() + games.rz_car.sum()) if len(games) else 0},
            "model": {"targets": rnd(p["mu_t"], 1), "carries": rnd(p["mu_c"], 1), "attempts": rnd(p.get("mu_a"), 1), "td_rate": rnd(p["lam_td"], 2),
                      "team_att": rnd(p["team_att"], 1), "team_car": rnd(p["team_car"], 1), "team_td": rnd(p["team_td"], 2)},
            "prior": {"weight": prior_weight(wk.W), "dropped": p["drop"]},
            "markets": mkts, "flags": flags,
            # deep dive: every game this season, and this week's defense vs the same role, game by game
            "log": [{"wk": int(x.week), "opp": x.opp, "tgt": int(x.targets), "rec": int(x.receptions), "rec_yds": int(x.receiving_yards),
                     "car": int(x.carries), "rush_yds": int(x.rushing_yards), "att": int(x.attempts), "cmp": int(x.completions),
                     "pass_yds": int(x.passing_yards), "pass_td": int(x.passing_tds), "int": int(x.interceptions), "td": int(x.tds),
                     "rz": int(x.rz_tgt + x.rz_car), "snap": rnd(pd.to_numeric(x.snap_pct, errors="coerce"), 2),
                     "tgt_share": rnd(x.targets / x.tm_targets, 3) if x.tm_targets else None} for x in games.itertuples()],
            "def_log": [{"wk": int(x.week), "vs": x.team, "name": x.name, "tgt": int(x.targets), "rec": int(x.receptions),
                         "rec_yds": int(x.receiving_yards), "car": int(x.carries), "rush_yds": int(x.rushing_yards),
                         "pass_yds": int(x.passing_yards), "pass_td": int(x.passing_tds), "td": int(x.tds)} for x in dv.itertuples()]}

def rnd(x, n=1):
    try:
        return None if x is None or np.isnan(x) else round(float(x), n)
    except TypeError:
        return None

# ------------------------------------------------------------------ backtest
def backtest(first, last, disp=None, verbose=True):
    D = setup(range(first, last + 1))
    rows = []
    rng = np.random.default_rng(7)
    all_roles = {}
    for S in range(first - 1, last + 1):
        all_roles.update(season_roles(D, S, 19))
    sched = D["sched"].set_index("game_id")
    for S in range(first, last + 1):
        coef = fit_coefs(D, S)
        D["role_means"] = role_means(D, S, all_roles)
        for W in range(1, 19):
            wk = D["wk_cache"].get((S, W)) or Week(D, S, W)
            games = D["pg"][(D["pg"].season == S) & (D["pg"].week == W)]
            for _, x in games.iterrows():
                role = all_roles.get((S, W, x.player_id))
                if not role or x.game_id not in sched.index:
                    continue
                gr = sched.loc[x.game_id]
                home = x.team == gr.home_team
                sp, tot = gr.spread_line, gr.total_line
                if pd.isna(sp) or pd.isna(tot):
                    continue
                implied = tot / 2 + (sp / 2 if home else -sp / 2); spr = -sp if home else sp
                p = project(wk, D, x.player_id, x.position, role, x.team, x.opp, implied, spr, coef)
                sims = simulate(p, rng, 1500, disp)
                cg = wk.cur[wk.cur.player_id == x.player_id]
                for mk in MARKETS[x.position]:
                    probs = tier_probs(p, sims, mk)
                    actual = x[STAT[mk]]
                    sv = cg[STAT[mk]].tolist(); l3 = sv[-3:]
                    for t, pr in zip(TIERS[mk], probs):
                        rows.append((S, W, role, mk, t, pr, int(actual >= t),
                                     (sum(v >= t for v in sv) / len(sv)) if len(sv) >= 3 else np.nan,
                                     (sum(v >= t for v in l3) / 3) if len(l3) == 3 else np.nan))
        if verbose:
            print(f"season {S} done ({len(rows)} tier predictions)")
    return pd.DataFrame(rows, columns=["season", "week", "role", "market", "tier", "p", "hit", "hr_season", "hr_l3"])

def summarize_backtest(df):
    def brier(p, y):
        return float(np.mean((p - y) ** 2))
    out = {"generated_utc": now(), "seasons": [int(df.season.min()), int(df.season.max())], "predictions": int(len(df)), "markets": {}}
    for mk, g in df.groupby("market"):
        m = g.dropna(subset=["hr_season"])
        # hit-rate baselines are shrunk a little so 0/3 and 3/3 aren't treated as certainties
        hr_s = (m.hr_season * 0.85 + 0.075); hr_l3 = m.dropna(subset=["hr_l3"])
        bins = pd.cut(g.p, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
        cal = g.groupby(bins, observed=True).agg(n=("hit", "size"), model=("p", "mean"), actual=("hit", "mean"))
        out["markets"][mk] = {
            "n": int(len(g)), "brier_model": round(brier(g.p, g.hit), 4), "base_rate": round(float(g.hit.mean()), 3),
            "brier_model_where_hr": round(brier(m.p, m.hit), 4), "brier_season_hr": round(brier(hr_s, m.hit), 4),
            "brier_l3_hr": round(brier(hr_l3.hr_l3 * 0.85 + 0.075, hr_l3.hit), 4), "brier_blend": round(brier((m.p + hr_s) / 2, m.hit), 4),
            "calibration": [{"bin": str(b), "n": int(r.n), "model": round(float(r.model), 3), "actual": round(float(r.actual), 3)} for b, r in cal.iterrows()]}
    allm = df.dropna(subset=["hr_season"])
    out["headline"] = {"seasons": out["seasons"], "predictions": out["predictions"],
                       "brier_model": round(float(np.mean((allm.p - allm.hit) ** 2)), 4),
                       "brier_season_hr": round(float(np.mean((allm.hr_season * 0.85 + 0.075 - allm.hit) ** 2)), 4),
                       "calibration": out["markets"]}
    out["headline"].pop("calibration")
    return out

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        a = [int(x) for x in sys.argv[2:]] or [2023, 2025]
        df = backtest(a[0], a[1])
        os.makedirs(OUT, exist_ok=True)
        df.to_csv(f"{OUT}/backtest_tiers.csv.gz", index=False, compression="gzip")
        summ = summarize_backtest(df)
        with open(f"{OUT}/backtest_summary.json", "w") as f:
            json.dump(summ, f, indent=1)
        print(json.dumps(summ["headline"], indent=1))
        for mk, v in summ["markets"].items():
            print(mk, {k: v[k] for k in v if k != "calibration"})
    else:
        D = setup([int(pd.read_csv("data/nfl/schedule.csv", usecols=["season"]).season.max())])
        current_week_cards(D)
