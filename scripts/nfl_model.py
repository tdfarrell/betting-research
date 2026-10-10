#!/usr/bin/env python3
"""NFL ratings used by the screener and the backtest. Reads the summaries in data/nfl/.

CURRENT SEASON FIRST. The three screens that grade games use this season's games only:
  epa     Opponent-adjusted offensive / defensive EPA per play (garbage time removed).
  sr      Opponent-adjusted success rate (steadier than EPA, less swayed by a few big plays).
  pts     Opponent-adjusted points scored / allowed.
Last season enters only as a starting point that fades out: full weight in Week 1, gone by Week 6
(rosters and coaching staffs change too much year to year to lean on it).

Angles (shown for context, never used to grade):
  last    Last season's final EPA rating.
  elo     Multi-season Elo (carries over from year to year by design).
  mkt     Market rating backed out of recent closing spreads.

All screens and angles get the same QB adjustment when the projected starter differs from the QBs
who played for that team this season.
"""
import csv, math
from collections import defaultdict
import numpy as np

DATA = "data/nfl"

def fnum(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d

def load(path=DATA):
    sched = list(csv.DictReader(open(f"{path}/schedule.csv")))
    tg = list(csv.DictReader(open(f"{path}/team_games.csv")))
    qg = list(csv.DictReader(open(f"{path}/qb_games.csv")))
    for r in sched:
        r["season"], r["week"] = int(r["season"]), int(r["week"])
        r["neutral"] = r["location"] == "Neutral"
    for r in tg:
        r["season"], r["week"] = int(r["season"]), int(r["week"])
        for k in r:
            if k not in ("game_id", "season_type", "team", "opp", "season", "week"):
                r[k] = fnum(r[k], 0.0)
    for r in qg:
        r["season"], r["week"] = int(r["season"]), int(r["week"])
        r["dropbacks"], r["epa"] = int(float(r["dropbacks"])), float(r["epa"])
    return sched, tg, qg

def seq(season, week):
    """A running week counter across seasons (offseason counted as a GAP-week break)."""
    return season * 100 + week

class Params:
    half_life = 8.0       # weeks; recency weight halves every N weeks of football
    gap = 6.0             # offseason counts as this many extra weeks of staleness
    ridge = 250.0         # plays of shrinkage toward average for each team's EPA rating
    plays = 62.0          # offensive snaps per team per game (EPA/play -> points)
    hfa = 1.6             # home field, points
    elo_k = 20.0
    elo_hfa = 40.0
    elo_per_pt = 25.0
    elo_regress = 1 / 3
    qb_k = 250.0          # dropbacks of shrinkage for QB value
    qb_prior = -0.06      # EPA/dropback assumed for a QB with no history (replacement level)
    qb_db = 34.0          # dropbacks per game (QB EPA/db -> points)
    mkt_half_life = 4.0

P = Params()

def weeks_ago(s, w, cur_s, cur_w, gap):
    if s == cur_s:
        return cur_w - w
    # weeks left in that season + weeks into this one + offseason gap (+ 18-week seasons in between)
    return (18 - w) + (cur_w - 1) + gap + 18 * (cur_s - s - 1) + 1

def recency(s, w, cur_s, cur_w, hl=None, gap=None):
    a = weeks_ago(s, w, cur_s, cur_w, P.gap if gap is None else gap)
    return 0.5 ** (a / (P.half_life if hl is None else hl))

def epa_ratings(tg, sched, cur_s, cur_w, hl=None, gap=None, ridge=None, field="epa", nfield="plays"):
    """Weighted ridge: y(off-team game EPA/play) = mu + home + off[team] + def[opp]."""
    loc = {r["game_id"]: r for r in sched}
    rows = [r for r in tg if seq(r["season"], r["week"]) < seq(cur_s, cur_w) and r[nfield] > 0
            and (cur_s - r["season"]) <= 2]
    teams = sorted({r["team"] for r in rows} | {r["opp"] for r in rows})
    ix = {t: k for k, t in enumerate(teams)}
    n = len(teams)
    X = np.zeros((len(rows), 2 + 2 * n)); y = np.zeros(len(rows)); w = np.zeros(len(rows))
    for k, r in enumerate(rows):
        g = loc.get(r["game_id"])
        h = 0 if (g and g["neutral"]) else (1 if r["home"] else -1)
        X[k, 0] = 1; X[k, 1] = h; X[k, 2 + ix[r["team"]]] = 1; X[k, 2 + n + ix[r["opp"]]] = 1
        y[k] = r[field] / r[nfield]
        w[k] = r[nfield] * recency(r["season"], r["week"], cur_s, cur_w, hl, gap)
    lam = np.full(2 + 2 * n, P.ridge if ridge is None else ridge); lam[:2] = 1e-6
    W = w[:, None]
    beta = np.linalg.solve(X.T @ (X * W) + np.diag(lam), X.T @ (w * y))
    off = {t: beta[2 + ix[t]] for t in teams}
    de = {t: beta[2 + n + ix[t]] for t in teams}
    return {"mu": beta[0], "home": beta[1], "off": off, "def": de}

def points_ratings(sched, cur_s, cur_w, hl=None, gap=None, ridge=6.0):
    """Opponent-adjusted points scored/allowed per game (ridge, games of shrinkage)."""
    rows = []
    for g in sched:
        if g["game_type"] == "REG" or g["game_type"] in ("WC", "DIV", "CON", "SB"):
            if seq(g["season"], g["week"]) < seq(cur_s, cur_w) and g["home_score"] not in ("", None) and (cur_s - g["season"]) <= 2:
                hs, as_ = float(g["home_score"]), float(g["away_score"])
                h = 0 if g["neutral"] else 1
                wt = recency(g["season"], min(g["week"], 18), cur_s, cur_w, hl, gap)
                rows += [(g["home_team"], g["away_team"], h, hs, wt), (g["away_team"], g["home_team"], -h, as_, wt)]
    teams = sorted({r[0] for r in rows})
    ix = {t: k for k, t in enumerate(teams)}; n = len(teams)
    X = np.zeros((len(rows), 2 + 2 * n)); y = np.zeros(len(rows)); w = np.zeros(len(rows))
    for k, (o, d, h, pts, wt) in enumerate(rows):
        X[k, 0] = 1; X[k, 1] = h; X[k, 2 + ix[o]] = 1; X[k, 2 + n + ix[d]] = 1; y[k] = pts; w[k] = wt
    lam = np.full(2 + 2 * n, ridge); lam[:2] = 1e-6
    beta = np.linalg.solve(X.T @ (X * w[:, None]) + np.diag(lam), X.T @ (w * y))
    return {"mu": beta[0], "home": beta[1], "off": {t: beta[2 + ix[t]] for t in teams}, "def": {t: beta[2 + n + ix[t]] for t in teams}}


PRIOR_FADE_WEEKS = 5      # last season's weight: 1.0 in Week 1, 0.8 Week 2 ... 0 from Week 6
SEASON_WEEKS = 18

def prior_weight(week):
    return max(0.0, 1 - (week - 1) / PRIOR_FADE_WEEKS)

def _solve(items, teams, lam0, lam_p=0.0, prior=None):
    """Ridge for y = mu + home*h + off[o] + def[d], with team terms shrunk toward a prior.
    items: (off_team, def_team, h, y, weight)."""
    ix = {t: k for k, t in enumerate(teams)}; n = len(teams)
    X = np.zeros((len(items), 2 + 2 * n)); y = np.zeros(len(items)); w = np.zeros(len(items))
    for k, (o, d, h, yy, wt) in enumerate(items):
        X[k, 0] = 1; X[k, 1] = h; X[k, 2 + ix[o]] = 1; X[k, 2 + n + ix[d]] = 1; y[k] = yy; w[k] = wt
    lam = np.full(2 + 2 * n, lam0 + lam_p); target = np.zeros(2 + 2 * n)
    lam[:2] = 1.0  # mu and home are learned from data (tiny pull toward the prior when there is none)
    if prior:
        target[0], target[1] = prior["mu"], prior["home"]
        if lam_p > 0:
            for t in teams:
                target[2 + ix[t]] = prior["off"].get(t, 0.0) * lam_p / (lam0 + lam_p)
                target[2 + n + ix[t]] = prior["def"].get(t, 0.0) * lam_p / (lam0 + lam_p)
    A = X.T @ (X * w[:, None]) + np.diag(lam)
    beta = np.linalg.solve(A, X.T @ (w * y) + lam * target)
    return {"mu": beta[0], "home": beta[1], "off": {t: beta[2 + ix[t]] for t in teams},
            "def": {t: beta[2 + n + ix[t]] for t in teams}}

def _play_items(tg, sched, season, before_week, field, nfield):
    loc = {r["game_id"]: r for r in sched}
    out = []
    for r in tg:
        if r["season"] == season and r["week"] < before_week and r["season_type"] == "REG" and r[nfield] > 0:
            g = loc.get(r["game_id"])
            h = 0 if (g and g["neutral"]) else (1 if r["home"] else -1)
            out.append((r["team"], r["opp"], h, r[field] / r[nfield], r[nfield]))
    return out

def _points_items(sched, season, before_week):
    out = []
    for g in sched:
        if g["season"] == season and g["game_type"] == "REG" and g["week"] < before_week and g["home_score"] not in ("", None):
            h = 0 if g["neutral"] else 1
            out += [(g["home_team"], g["away_team"], h, float(g["home_score"]), 1.0),
                    (g["away_team"], g["home_team"], -h, float(g["away_score"]), 1.0)]
    return out

SCREENS = {  # field, denominator, base shrinkage, prior strength (in plays or games)
    "epa": ("epa", "plays", 150.0, 400.0),
    "sr": ("success", "plays", 150.0, 400.0),
    "pts": (None, None, 2.0, 5.0),
}
_END = {}

def season_end(tg, sched, season, screen):
    """A team's full-season rating (no prior). Used as next season's fading starting point."""
    key = (season, screen)
    if key not in _END:
        field, den, lam0, _ = SCREENS[screen]
        items = _points_items(sched, season, 99) if screen == "pts" else _play_items(tg, sched, season, 99, field, den)
        teams = sorted({i[0] for i in items} | {i[1] for i in items})
        _END[key] = _solve(items, teams, lam0) if items else None
    return _END[key]

def current_ratings(tg, sched, season, week, screen):
    """This season's games before `week`, shrunk toward last season's rating with a fading weight."""
    field, den, lam0, lam_p = SCREENS[screen]
    items = _points_items(sched, season, week) if screen == "pts" else _play_items(tg, sched, season, week, field, den)
    prior = season_end(tg, sched, season - 1, screen)
    teams = sorted({i[0] for i in items} | {i[1] for i in items} | set(prior["off"] if prior else []))
    if not teams:
        return {"mu": 0.0, "home": 0.0, "off": {}, "def": {}}
    return _solve(items, teams, lam0, lam_p * prior_weight(week) if prior else 0.0, prior)

def market_ratings(sched, cur_s, cur_w, hl=None):
    """Least squares on past closing spreads: spread_line = hfa*home + R_home - R_away."""
    rows = [g for g in sched if seq(g["season"], g["week"]) < seq(cur_s, cur_w) and fnum(g["spread_line"]) is not None
            and (cur_s - g["season"]) <= 1]
    teams = sorted({g["home_team"] for g in rows} | {g["away_team"] for g in rows})
    ix = {t: k for k, t in enumerate(teams)}; n = len(teams)
    X = np.zeros((len(rows), 1 + n)); y = np.zeros(len(rows)); w = np.zeros(len(rows))
    for k, g in enumerate(rows):
        X[k, 0] = 0 if g["neutral"] else 1; X[k, 1 + ix[g["home_team"]]] = 1; X[k, 1 + ix[g["away_team"]]] = -1
        y[k] = float(g["spread_line"])
        w[k] = recency(g["season"], min(g["week"], 18), cur_s, cur_w, hl or P.mkt_half_life, 4.0)
    lam = np.full(1 + n, 0.5); lam[0] = 1e-6
    beta = np.linalg.solve(X.T @ (X * w[:, None]) + np.diag(lam), X.T @ (w * y))
    r = {t: beta[1 + ix[t]] for t in teams}
    m = sum(r.values()) / len(r)
    return {"hfa": beta[0], "r": {t: v - m for t, v in r.items()}}

class Elo:
    def __init__(self):
        self.r = defaultdict(lambda: 1505.0)
        self.season = None

    def pre(self, g):
        return self.r[g["home_team"]] - self.r[g["away_team"]] + (0 if g["neutral"] else P.elo_hfa)

    def update(self, g):
        if g["season"] != self.season:
            if self.season is not None:
                for t in self.r:
                    self.r[t] = self.r[t] * (1 - P.elo_regress) + 1505 * P.elo_regress
            self.season = g["season"]
        hs, as_ = fnum(g["home_score"]), fnum(g["away_score"])
        if hs is None:
            return
        d = self.pre(g)
        exp = 1 / (1 + 10 ** (-d / 400))
        res = 1.0 if hs > as_ else 0.0 if hs < as_ else 0.5
        mov = math.log(abs(hs - as_) + 1) * 2.2 / ((d if res == 1 else -d) * 0.001 + 2.2)
        delta = P.elo_k * mov * (res - exp)
        self.r[g["home_team"]] += delta; self.r[g["away_team"]] -= delta

class QB:
    """Shrunk career-to-date EPA per dropback for every QB, and each team's recent QB mix."""
    def __init__(self, qg):
        self.qg = sorted(qg, key=lambda r: (r["season"], r["week"]))

    def values(self, cur_s, cur_w):
        tot = defaultdict(lambda: [0.0, 0])
        names = {}
        team_mix = defaultdict(lambda: defaultdict(float))
        prev_mix = defaultdict(lambda: defaultdict(float))
        for r in self.qg:
            if seq(r["season"], r["week"]) >= seq(cur_s, cur_w):
                break
            t = tot[r["qb_id"]]
            # older seasons count less
            age_w = 0.75 ** max(0, cur_s - r["season"])
            t[0] += r["epa"] * age_w; t[1] += r["dropbacks"] * age_w
            names[r["qb_id"]] = r["qb"]
            if r["season"] == cur_s:
                team_mix[r["team"]][r["qb_id"]] += r["dropbacks"]
            elif r["season"] == cur_s - 1:
                prev_mix[r["team"]][r["qb_id"]] += r["dropbacks"]
        for t, m in prev_mix.items():  # Week 1: the ratings start from last season, so compare to last season's QBs
            if t not in team_mix:
                team_mix[t] = m
        val = {q: (s + P.qb_prior * P.qb_k) / (n + P.qb_k) for q, (s, n) in tot.items()}
        return val, team_mix, names

    @staticmethod
    def adjust(team, starter_id, val, team_mix):
        """Points to add to a team's rating when its projected starter differs from its recent QB mix."""
        mix = team_mix.get(team)
        if not mix or not starter_id:
            return 0.0, None
        tot = sum(mix.values())
        base = sum(val.get(q, P.qb_prior) * w for q, w in mix.items()) / tot
        share = mix.get(starter_id, 0) / tot
        if share > 0.80:
            return 0.0, base
        return (val.get(starter_id, P.qb_prior) - base) * P.qb_db, base

def team_pace(tg, cur_s, cur_w, team):
    """Neutral seconds per snap this season, with last season fading out the same way (lower = faster)."""
    secs = plays = 0.0
    for r in tg:
        if r["team"] == team and seq(r["season"], r["week"]) < seq(cur_s, cur_w) and cur_s - r["season"] <= 1:
            wt = 1.0 if r["season"] == cur_s else 0.25 * prior_weight(cur_w)
            secs += r["neutral_secs"] * wt; plays += r["neutral_plays"] * wt
    return secs / plays if plays else None


def matchup_ratings(tg, sched, cur_s, cur_w):
    """Pass/rush splits and sack rate, opponent adjusted (for matchup notes)."""
    SCREENS.setdefault("pass", ("pass_epa", "pass_plays", 120.0, 300.0))
    SCREENS.setdefault("rush", ("rush_epa", "rush_plays", 100.0, 250.0))
    return {"pass": current_ratings(tg, sched, cur_s, cur_w, "pass"), "rush": current_ratings(tg, sched, cur_s, cur_w, "rush")}

def predict(first, last, data=None, with_matchups=False):
    """Raw screen outputs for every regular-season game in seasons first..last (played or not).
    Ratings for week W use only games before week W. Returns (rows, context-by-week)."""
    sched, tg, qg = data or load()
    games = sorted([g for g in sched if g["game_type"] == "REG"], key=lambda g: (g["season"], g["week"], g["gameday"]))
    elo, qb = Elo(), QB(qg)
    out, ctx, pace_cache = [], {}, {}
    byweek = defaultdict(list)
    for g in games:
        byweek[(g["season"], g["week"])].append(g)
    for (s, w) in sorted(byweek):
        wk = byweek[(s, w)]
        if first <= s <= last:
            er = current_ratings(tg, sched, s, w, "epa")
            sr = current_ratings(tg, sched, s, w, "sr")
            pr = current_ratings(tg, sched, s, w, "pts")
            lastr = season_end(tg, sched, s - 1, "epa") or {"mu": 0.0, "home": 0.0, "off": {}, "def": {}}
            mr = market_ratings(sched, s, w)
            val, mix, names = qb.values(s, w)
            c = ctx[(s, w)] = {"epa": er, "sr": sr, "pts": pr, "last": lastr, "mkt": mr, "qb_val": val, "qb_mix": mix,
                               "qb_names": names, "elo": dict(elo.r), "prior_weight": prior_weight(w)}
            if with_matchups:
                c["match"] = matchup_ratings(tg, sched, s, w)
            for g in wk:
                h, a, neu = g["home_team"], g["away_team"], g["neutral"]
                if h not in er["off"] or a not in er["off"]:
                    continue
                hh = 0 if neu else er["home"]
                eh = er["mu"] + er["off"][h] + er["def"][a] + hh
                ea = er["mu"] + er["off"][a] + er["def"][h] - hh
                qh, bh = QB.adjust(h, g["home_qb_id"], val, mix)
                qa, ba = QB.adjust(a, g["away_qb_id"], val, mix)
                ph = 0 if neu else pr["home"]
                sh = 0 if neu else sr["home"]
                lh = 0 if neu else lastr["home"]
                sr_m = ((sr["off"][h] + sr["def"][a] + sh) - (sr["off"][a] + sr["def"][h] - sh))
                last_m = ((lastr["off"].get(h, 0) + lastr["def"].get(a, 0) + lh) - (lastr["off"].get(a, 0) + lastr["def"].get(h, 0) - lh)) * P.plays
                for t in (h, a):
                    if (s, w, t) not in pace_cache:
                        pace_cache[(s, w, t)] = team_pace(tg, s, w, t)
                played = g["home_score"] not in ("", None)
                out.append({
                    "season": s, "week": w, "game_id": g["game_id"], "espn": g["espn"], "home": h, "away": a, "neutral": int(neu),
                    "close": fnum(g["spread_line"]), "close_total": fnum(g["total_line"]),
                    "result": float(g["home_score"]) - float(g["away_score"]) if played else None,
                    "total": float(g["home_score"]) + float(g["away_score"]) if played else None,
                    "epa_raw": (eh - ea) * P.plays, "epa_sum": eh + ea, "sr_raw": sr_m * 100, "last_raw": last_m,
                    "elo_raw": elo.pre(g) / P.elo_per_pt,
                    "mkt_raw": (0 if neu else mr["hfa"]) + mr["r"].get(h, 0) - mr["r"].get(a, 0),
                    "pts_raw": (pr["off"][h] + pr["def"][a] + ph) - (pr["off"][a] + pr["def"][h] - ph),
                    "pts_total": 2 * pr["mu"] + pr["off"][h] + pr["def"][a] + pr["off"][a] + pr["def"][h],
                    "qb_adj": qh - qa, "qb_h": qh, "qb_a": qa,
                    "pace": (pace_cache[(s, w, h)] or 28) + (pace_cache[(s, w, a)] or 28),
                    "dome": int(g["roof"] in ("dome", "closed")), "wind": fnum(g["wind"], 0),
                    "div": int(g["div_game"] == "1"), "rest_diff": fnum(g["home_rest"], 7) - fnum(g["away_rest"], 7),
                })
        for g in wk:
            elo.update(g)
    return out, ctx

SIDE_MODELS = ("epa", "sr", "pts")     # current season: these grade
ANGLES = ("last", "elo", "mkt")         # context only

def match_fit(x, y):
    """Map a screen onto the market's scale so that its edges are UNRELATED to the size of the line.
    We fit what the screen typically says for a given market number (screen = a + b * market) and invert
    it. A plain regression (market on screen) pulls predictions toward the middle, so big favorites and
    high totals always looked too big and the picks were nearly all underdogs and unders.
    Returns (slope, intercept) to apply to the screen."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if y.std() < 1e-9 or x.std() < 1e-9:
        return 0.0, float(y.mean())
    b, a = np.polyfit(y, x, 1)
    if abs(b) < 0.15:  # screen barely tracks the market: fall back to matching spreads
        b = np.sign(np.corrcoef(x, y)[0, 1] or 1) * x.std() / y.std()
        a = x.mean() - b * y.mean()
    return float(1 / b), float(-a / b)

def calibrate(rows, n=64, min_n=40):
    """Put each screen on the market's scale using recent closing lines (rolling, earlier weeks only).
    Adds <m>_cal (home margin, + = home favored), blend (the BW number), epa_tot_cal, pts_tot_cal, tot_blend."""
    done = []
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[seq(r["season"], r["week"])].append(r)
    for sq in sorted(by_seq):
        # recent weeks + THIS week's own lines (like the CFB fit-to-the-slate step): ratings change shape as
        # last season fades out, so this week's numbers must be scaled against this week's board too
        cur = [r for r in by_seq[sq] if r["close"] is not None]
        past = [r for r in done if r["close"] is not None][-max(n - len(cur), 16):] + cur
        fits = {}
        if len(past) >= min_n:
            y = np.array([r["close"] for r in past])
            for m in SIDE_MODELS + ANGLES:
                fits[m] = match_fit([r[m + "_raw"] + r["qb_adj"] for r in past], y)
            # the average of three screens is narrower than any one of them: rescale it to the market too
            bl = np.array([np.mean([fits[m][0] * (r[m + "_raw"] + r["qb_adj"]) + fits[m][1] for m in SIDE_MODELS]) for r in past])
            fits["blend"] = match_fit(bl, y)
            tp = [r for r in past if r["close_total"] is not None]
            yt = np.array([r["close_total"] for r in tp])
            res = [r for r in done if r["total"] is not None][-512:]
            X = np.array([[1, r["epa_sum"], r["pace"], r["dome"]] for r in res]); yy = np.array([r["total"] for r in res])
            coef = np.linalg.lstsq(X, yy, rcond=None)[0]
            et = lambda r, c=coef: float(c @ np.array([1, r["epa_sum"], r["pace"], r["dome"]]))
            fits["epa_tot"] = match_fit([et(r) for r in tp], yt)
            fits["pts_tot"] = match_fit([r["pts_total"] for r in tp], yt)
            tb = np.array([(fits["epa_tot"][0] * et(r) + fits["epa_tot"][1] + fits["pts_tot"][0] * r["pts_total"] + fits["pts_tot"][1]) / 2 for r in tp])
            fits["tot_blend"] = match_fit(tb, yt)
            # how far the BW numbers have typically strayed from the close lately (for gap sizes that mean
            # the same thing in Week 4 as in Week 14)
            bwp = fits["blend"][0] * bl + fits["blend"][1]
            side_sd = float(np.sqrt(np.mean((bwp - y) ** 2)))
            tot_sd = float(np.sqrt(np.mean((tb - yt) ** 2)))
        for r in by_seq[sq]:
            if fits:
                for m in SIDE_MODELS + ANGLES:
                    r[m + "_cal"] = fits[m][0] * (r[m + "_raw"] + r["qb_adj"]) + fits[m][1]
                b0 = np.mean([r[m + "_cal"] for m in SIDE_MODELS])
                r["blend"] = fits["blend"][0] * b0 + fits["blend"][1]
                r["epa_tot_cal"] = fits["epa_tot"][0] * et(r) + fits["epa_tot"][1]
                r["pts_tot_cal"] = fits["pts_tot"][0] * r["pts_total"] + fits["pts_tot"][1]
                r["tot_blend"] = fits["tot_blend"][0] * (r["epa_tot_cal"] + r["pts_tot_cal"]) / 2 + fits["tot_blend"][1]
                r["side_sd"], r["tot_sd"] = side_sd, tot_sd
                r["calib"] = {k: [round(v[0], 3), round(v[1], 2)] for k, v in fits.items() if k not in ANGLES}
                r["calib"].update({"side_sd": round(side_sd, 2), "tot_sd": round(tot_sd, 2)})
                r["calib"]["n"] = len(past)
            else:
                for m in SIDE_MODELS + ANGLES:
                    r[m + "_cal"] = None
                r["blend"] = r["epa_tot_cal"] = r["pts_tot_cal"] = r["tot_blend"] = r["side_sd"] = r["tot_sd"] = None
        done += by_seq[sq]
    return rows
