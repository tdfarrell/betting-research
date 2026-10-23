#!/usr/bin/env python3
"""NBA game lines: DraftKings spread/total vs Ballwatcher's team ratings -> data/boards/nba_<date>.json.

  python scripts/regrade_nba.py              grade every upcoming game on the DK board (one slate per ET day)
  python scripts/regrade_nba.py backtest     replay past seasons vs DK closing lines -> data/nba/backtest_summary.json

Screens (both opponent adjusted, built only from games before the slate):
  eff   offensive / defensive points per 100 possessions + pace (possessions per game)
  pts   points scored / allowed per game (margin and total)
Each starts from last season's final rating, shrunk 35% toward average, worth PRIOR_GAMES games; current-season
games take over as they pile up (by ~25 games the prior is a small minority).
Home court and back-to-backs are added to both screens. Each screen is put on DraftKings' scale with recent
closing lines plus this slate's own lines (fit screen = a + b * line and invert, like the NFL and CFB boards),
so an edge can't be predicted from the size of the line. The average of the two is the BW number / BW total.
Grades (sides and totals): WATCH 0.3-0.6x the usual gap, TARGET 0.6-1.0x, CHECK above it (shown as a lean);
both screens must sit on the same side by 1+ pt for WATCH/TARGET. A line that moved 2+ against our side since
open drops a tier. A team's top-2 players (minutes) listed Out/Doubtful/Questionable/Day-To-Day -> FLAG (not
graded): the ratings were built with them playing and the market prices the news.
"""
import csv, datetime as dt, json, math, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from board_keep import merge_started
from picks import finalize_rows

D = "data/nba"
OUT = "data/boards"
ET = dt.timezone(dt.timedelta(hours=-4))
PRIOR_GAMES = 10.0     # last season's rating counts as this many games
PRIOR_SHRINK = 0.65    # ... after shrinking it toward average (rosters change)
B2B = 1.5              # points lost on the second night of a back-to-back
MOVE_AGAINST = 2.0
STAR_STATUS = ("out", "doubtful", "questionable", "day-to-day")
NBA_TEAMS = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
             "NO", "NY", "OKC", "ORL", "PHI", "PHX", "POR", "SA", "SAC", "TOR", "UTAH", "WSH"}   # skips All-Star games
DOWN = {"TARGET": "WATCH", "WATCH": "PASS", "PASS": "PASS", "FLAG": "FLAG", "CHECK": "CHECK"}

def num(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None

def rnd(x, n=1):
    return None if x is None else round(float(x), n)

def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")

def load():
    T = [r for r in csv.DictReader(open(f"{D}/team_games.csv"))]
    T = [r for r in T if r["team"] in NBA_TEAMS and r["opp"] in NBA_TEAMS]
    for r in T:
        for k in ("pts", "opp_pts", "poss", "home", "rest", "ot", "season"):
            r[k] = num(r[k])
    L = {r["game_id"]: r for r in csv.DictReader(open(f"{D}/lines_close.csv"))} if os.path.exists(f"{D}/lines_close.csv") else {}
    return T, L

# ------------------------------------------------------------------ ratings
def games_of(T):
    """team_games rows -> one record per game (home perspective)."""
    by = defaultdict(dict)
    for r in T:
        by[r["game_id"]][int(r["home"] or 0)] = r
    out = []
    for gid, s in by.items():
        if 0 in s and 1 in s:
            h, a = s[1], s[0]
            out.append({"id": gid, "season": int(h["season"]), "date": h["date"], "home": h["team"], "away": a["team"],
                        "hpts": h["pts"], "apts": a["pts"], "hposs": h["poss"], "aposs": a["poss"], "ot": h["ot"] or 0,
                        "hrest": h["rest"], "arest": a["rest"]})
    return sorted(out, key=lambda g: (g["date"], g["id"]))

def solve(rows, teams, prior, k, mu_prior=None, hca_prior=None, k_mu=60.0, k_hca=400.0):
    """Least squares y = mu + X_off[team] - X_def[opp] + hca*home with ridge pseudo-games toward prior.
    rows: (team, opp, home, y). prior: {team: (o, d)} in units of y around mu. Returns mu, hca, {team: (o, d)}."""
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams); P = 2 + 2 * n
    A, b = [], []
    for t, o, hm, y in rows:
        x = np.zeros(P); x[0] = 1; x[1] = hm; x[2 + idx[t]] = 1; x[2 + n + idx[o]] = -1
        A.append(x); b.append(y)
    mu0 = float(np.mean(b)) if b else 0.0
    for t in teams:
        po, pd = prior.get(t, (0.0, 0.0))
        w = math.sqrt(k)
        x = np.zeros(P); x[2 + idx[t]] = w; A.append(x); b.append(w * po)
        x = np.zeros(P); x[2 + n + idx[t]] = w; A.append(x); b.append(w * pd)
    # pin the mean of offense and defense ratings at zero, and keep mu/hca sensible with tiny ridge
    x = np.zeros(P); x[2:2 + n] = 10; A.append(x); b.append(0.0)
    x = np.zeros(P); x[2 + n:] = 10; A.append(x); b.append(0.0)
    x = np.zeros(P); x[0] = 0.01; A.append(x); b.append(0.01 * mu0)
    # league average and home court lean on last season until enough games pile up (a few dozen games can't pin them down)
    if mu_prior is not None:
        x = np.zeros(P); x[0] = math.sqrt(k_mu); A.append(x); b.append(math.sqrt(k_mu) * mu_prior)
    if hca_prior is not None:
        x = np.zeros(P); x[1] = math.sqrt(k_hca); A.append(x); b.append(math.sqrt(k_hca) * hca_prior)
    sol = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    return float(sol[0]), float(sol[1]), {t: (float(sol[2 + idx[t]]), float(sol[2 + n + idx[t]])) for t in teams}

def solve_sum(rows, teams, prior, k):
    """y = mu + P[team] + P[opp] (pace: both teams push it), ridge toward prior."""
    idx = {t: i for i, t in enumerate(teams)}; n = len(teams)
    A, b = [], []
    for t, o, y in rows:
        x = np.zeros(1 + n); x[0] = 1; x[1 + idx[t]] += 1; x[1 + idx[o]] += 1; A.append(x); b.append(y)
    mu0 = float(np.mean(b)) if b else 99.0
    for t in teams:
        x = np.zeros(1 + n); x[1 + idx[t]] = math.sqrt(k); A.append(x); b.append(math.sqrt(k) * prior.get(t, 0.0))
    x = np.zeros(1 + n); x[1:] = 10; A.append(x); b.append(0.0)
    x = np.zeros(1 + n); x[0] = 0.01; A.append(x); b.append(0.01 * mu0)
    sol = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    return float(sol[0]), {t: float(sol[1 + idx[t]]) for t in teams}

class Ratings:
    """Everything known before `date` in `season` (ESPN year)."""
    def __init__(self, G, season, date, prior_end=None):
        cur = [g for g in G if g["season"] == season and g["date"] < date]
        teams = sorted({g["home"] for g in G if g["season"] == season} | {g["away"] for g in G if g["season"] == season})
        self.teams, self.n_games = teams, defaultdict(int)
        for g in cur:
            self.n_games[g["home"]] += 1; self.n_games[g["away"]] += 1
        pe = prior_end or {}
        sh = lambda key: {t: (PRIOR_SHRINK * pe[t][key][0], PRIOR_SHRINK * pe[t][key][1]) for t in teams if t in pe and t != "_league"}
        shp = {t: PRIOR_SHRINK * pe[t]["pace"] for t in teams if t in pe and t != "_league"}
        k = PRIOR_GAMES if pe else 2.0
        # efficiency: points per 100 possessions (overtime possessions included; fine at this scale)
        eff_rows, pace_rows, pts_rows = [], [], []
        for g in cur:
            poss = ((g["hposs"] or 0) + (g["aposs"] or 0)) / 2 or 98
            per48 = poss * 48 / (48 + 5 * g["ot"])
            eff_rows += [(g["home"], g["away"], 1, 100 * g["hpts"] / poss), (g["away"], g["home"], 0, 100 * g["apts"] / poss)]
            pace_rows.append((g["home"], g["away"], per48))
            sc = 48 / (48 + 5 * g["ot"])
            pts_rows += [(g["home"], g["away"], 1, g["hpts"] * sc), (g["away"], g["home"], 0, g["apts"] * sc)]
        c = (prior_end or {}).get("_league", {})
        self.mu_e, self.hca_e, self.eff = solve(eff_rows, teams, sh("eff"), k, c.get("mu_e"), c.get("hca_e", 1.8))
        self.mu_p, self.pace = solve_sum(pace_rows, teams, shp, k)
        if c.get("mu_p") and len(cur) < 100:   # pace level early: blend toward last season's
            w = len(cur) / 100; self.mu_p = w * self.mu_p + (1 - w) * c["mu_p"]
        self.mu_s, self.hca_s, self.pts = solve(pts_rows, teams, sh("pts"), k, c.get("mu_s"), c.get("hca_s", 1.8))
        if not cur:   # nothing played yet: league-average constants from last season's shape
            self.mu_e, self.hca_e, self.mu_p, self.mu_s, self.hca_s = 114.0, 1.2, 99.0, 114.0, 1.2

    def export(self):
        out = {t: {"eff": self.eff[t], "pace": self.pace[t], "pts": self.pts[t]} for t in self.teams}
        out["_league"] = {"mu_e": self.mu_e, "hca_e": self.hca_e, "mu_s": self.mu_s, "hca_s": self.hca_s, "mu_p": self.mu_p}
        return out

    def project(self, h, a, hrest=None, arest=None):
        pace = self.mu_p + self.pace.get(h, 0) + self.pace.get(a, 0)
        oh, dh = self.eff.get(h, (0, 0)); oa, da = self.eff.get(a, (0, 0))
        e_h = self.mu_e + oh - da + self.hca_e; e_a = self.mu_e + oa - dh
        sh, dsh = self.pts.get(h, (0, 0)); sa, dsa = self.pts.get(a, (0, 0))
        p_h = self.mu_s + sh - dsa + self.hca_s; p_a = self.mu_s + sa - dsh
        b2b = B2B * ((1 if arest == 1 else 0) - (1 if hrest == 1 else 0))
        return {"pace": pace, "eff_margin": (e_h - e_a) * pace / 100 + b2b, "eff_total": (e_h + e_a) * pace / 100,
                "pts_margin": p_h - p_a + b2b, "pts_total": p_h + p_a, "b2b": b2b}

_PRIOR_CACHE = {}
def prior_for(G, season):
    if season not in _PRIOR_CACHE:
        prev = [g for g in G if g["season"] == season - 1]
        if not prev:
            _PRIOR_CACHE[season] = None
        else:
            nxt = (dt.date.fromisoformat(max(g["date"] for g in prev)) + dt.timedelta(days=1)).isoformat()
            _PRIOR_CACHE[season] = Ratings(G, season - 1, nxt, prior_for(G, season - 1)).export()
    return _PRIOR_CACHE[season]

# ------------------------------------------------------------------ calibration (same idea as the NFL board)
def match_fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or y.std() < 1e-9 or x.std() < 1e-9:
        return 1.0, 0.0
    b, a = np.polyfit(y, x, 1)
    if abs(b) < 0.15:
        b = np.sign(np.corrcoef(x, y)[0, 1] or 1) * x.std() / y.std(); a = x.mean() - b * y.mean()
    return float(1 / b), float(-a / b)

def calibrate(past, cur, n=160, min_n=60):
    """past: earlier rows with closing lines; cur: this slate's rows (their own DK lines join the fit)."""
    sample = [r for r in past if r.get("close") is not None][-max(n - len(cur), 40):] + [r for r in cur if r.get("close") is not None]
    if len(sample) < min_n:
        return None
    y = np.array([r["close"] for r in sample])
    f = {m: match_fit([r[m] for r in sample], y) for m in ("eff_margin", "pts_margin")}
    bl = np.array([np.mean([f[m][0] * r[m] + f[m][1] for m in ("eff_margin", "pts_margin")]) for r in sample])
    f["blend"] = match_fit(bl, y)
    side_sd = float(np.sqrt(np.mean((f["blend"][0] * bl + f["blend"][1] - y) ** 2)))
    ts = [r for r in sample if r.get("close_total") is not None]
    yt = np.array([r["close_total"] for r in ts])
    f.update({m: match_fit([r[m] for r in ts], yt) for m in ("eff_total", "pts_total")})
    tb = np.array([np.mean([f[m][0] * r[m] + f[m][1] for m in ("eff_total", "pts_total")]) for r in ts])
    f["tblend"] = match_fit(tb, yt)
    tot_sd = float(np.sqrt(np.mean((f["tblend"][0] * tb + f["tblend"][1] - yt) ** 2))) if len(ts) else None
    f["side_sd"], f["tot_sd"], f["n"] = side_sd, tot_sd, len(sample)
    return f

def apply_cal(r, f):
    """Home margin (+ = home better) and total on DK's scale."""
    for m in ("eff_margin", "pts_margin", "eff_total", "pts_total"):
        r[m + "_cal"] = f[m][0] * r[m] + f[m][1]
    r["number"] = f["blend"][0] * (r["eff_margin_cal"] + r["pts_margin_cal"]) / 2 + f["blend"][1]
    r["total_number"] = f["tblend"][0] * (r["eff_total_cal"] + r["pts_total_cal"]) / 2 + f["tblend"][1]
    r["side_sd"], r["tot_sd"] = f["side_sd"], f["tot_sd"]

def bands(sd):
    return 0.3 * sd, 0.6 * sd, 1.0 * sd

def grade(edge, screens, sd, min_each=1.0):
    """edge/screens > 0 = home (side) or over (total)."""
    if edge is None or sd is None:
        return "PASS"
    w, t, c = bands(sd)
    agree = [x for x in screens if x is not None and abs(x) >= min_each and (x > 0) == (edge > 0)]
    if abs(edge) >= c:
        return "CHECK"
    if abs(edge) >= t and len(agree) == 2:
        return "TARGET"
    if abs(edge) >= w and len(agree) == 2:
        return "WATCH"
    return "PASS"

# ------------------------------------------------------------------ backtest
def history_rows(G, L, seasons, since=""):
    rows = []
    for s in seasons:
        pri = prior_for(G, s)
        days = sorted({g["date"] for g in G if g["season"] == s and g["date"] >= since})
        for d in days:
            R = Ratings(G, s, d, pri)
            for g in (x for x in G if x["season"] == s and x["date"] == d):
                p = R.project(g["home"], g["away"], g["hrest"], g["arest"])
                ln = L.get(g["id"], {})
                hs, tot = num(ln.get("home_spread")), num(ln.get("total"))
                rows.append({"id": g["id"], "season": s, "date": d, "home": g["home"], "away": g["away"], **p,
                             "close": -hs if hs is not None else None, "close_total": tot,
                             "margin": g["hpts"] - g["apts"], "total": g["hpts"] + g["apts"],
                             "gp": min(R.n_games[g["home"]], R.n_games[g["away"]])})
    return rows

def backtest(seasons):
    T, L = load()
    G = games_of(T)
    rows = history_rows(G, L, range(min(seasons) - 1, max(seasons) + 1))
    done, out = [], []
    by_day = defaultdict(list)
    for r in rows:
        by_day[(r["season"], r["date"])].append(r)
    for key in sorted(by_day):
        cur = by_day[key]
        f = calibrate(done, cur)
        if f and key[0] in seasons:
            for r in cur:
                if r["close"] is None:
                    continue
                apply_cal(r, f)
                e = r["number"] - r["close"]
                es = [r["eff_margin_cal"] - r["close"], r["pts_margin_cal"] - r["close"]]
                r["side_grade"] = grade(e, es, r["side_sd"])
                r["side_pick"] = 1 if e > 0 else -1
                r["side_res"] = None if r["margin"] == r["close"] else (1 if (r["margin"] - r["close"]) * r["side_pick"] > 0 else 0)
                r["side_gap"] = abs(e)
                if r["close_total"] is not None and r["tot_sd"]:
                    et = r["total_number"] - r["close_total"]
                    ets = [r["eff_total_cal"] - r["close_total"], r["pts_total_cal"] - r["close_total"]]
                    r["tot_grade"] = grade(et, ets, r["tot_sd"], 1.5)
                    r["tot_pick"] = 1 if et > 0 else -1
                    r["tot_res"] = None if r["total"] == r["close_total"] else (1 if (r["total"] - r["close_total"]) * r["tot_pick"] > 0 else 0)
                out.append(r)
        done += cur
    def rec(rs, res):
        w = sum(1 for r in rs if r.get(res) == 1); l = sum(1 for r in rs if r.get(res) == 0)
        return {"w": w, "l": l, "n": len(rs), "pct": round(w / (w + l), 3) if w + l else None}
    # every graded game, for the confidence tables (scripts/confidence.py)
    import csv as _csv
    with open(f"{D}/backtest_games.csv", "w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(["season", "date", "side_edge", "side_sd", "side_res", "tot_edge", "tot_sd", "tot_res",
                                          "game_id", "number", "total_number", "close", "close_total", "margin", "total"])
        for r in out:
            w.writerow([r["season"], r["date"], rnd(r["number"] - r["close"], 2), rnd(r["side_sd"], 2), r.get("side_res"),
                        rnd(r["total_number"] - r["close_total"], 2) if r.get("tot_res") is not None else "", rnd(r.get("tot_sd"), 2) if r.get("tot_sd") else "", r.get("tot_res"),
                        r["id"], rnd(r["number"], 2), rnd(r.get("total_number"), 2), r["close"], r.get("close_total"), r["margin"], r["total"]])
    summ = {"seasons": [min(seasons), max(seasons)], "games": len(out), "line": "DraftKings close (ESPN)",
            "params": {"prior_games": PRIOR_GAMES, "prior_shrink": PRIOR_SHRINK, "b2b": B2B},
            "rmse": {k: rnd(math.sqrt(np.mean([(r[k] - r["margin"]) ** 2 for r in out])), 2) for k in ("close", "number", "eff_margin_cal", "pts_margin_cal")} if out else {},
            "tiers_vs_close": {"side": {}, "total": {}}, "by_season": {}, "side_by_gap": {}, "early_vs_late": {}}
    for t in ("WATCH", "TARGET", "CHECK"):
        summ["tiers_vs_close"]["side"][t] = rec([r for r in out if r.get("side_grade") == t], "side_res")
        summ["tiers_vs_close"]["total"][t] = rec([r for r in out if r.get("tot_grade") == t], "tot_res")
    for s in seasons:
        rs = [r for r in out if r["season"] == s]
        summ["by_season"][s] = {t: rec([r for r in rs if r.get("side_grade") == t], "side_res") for t in ("WATCH", "TARGET", "CHECK")}
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 99)):
        summ["side_by_gap"][f"{lo}-{hi if hi < 99 else '+'}"] = rec([r for r in out if lo <= r.get("side_gap", -1) < hi], "side_res")
    for lab, cond in (("first 15 games", lambda r: r["gp"] < 15), ("after 15 games", lambda r: r["gp"] >= 15)):
        rs = [r for r in out if cond(r)]
        summ["early_vs_late"][lab] = {t: rec([r for r in rs if r.get("side_grade") == t], "side_res") for t in ("WATCH", "TARGET", "CHECK")}
    os.makedirs(D, exist_ok=True)
    json.dump(summ, open(f"{D}/backtest_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))

# ------------------------------------------------------------------ live board
def players_by_team(season):
    """Minutes leaders per team this season (last season if <3 games)."""
    P = [r for r in csv.DictReader(open(f"{D}/player_games.csv"))]
    out = {}
    for s in (season, season - 1):
        agg = defaultdict(lambda: {"g": 0, "min": 0.0, "pts": 0.0, "name": "", "team": "", "id": ""})
        for r in P:
            if int(r["season"]) != s or r["dnp"] == "1":
                continue
            a = agg[(r["team"], r["player_id"])]
            a["g"] += 1; a["min"] += float(r["min"]); a["pts"] += float(r["pts"]); a["name"] = r["name"]; a["team"] = r["team"]; a["id"] = r["player_id"]
        by = defaultdict(list)
        for a in agg.values():
            by[a["team"]].append({"id": a["id"], "name": a["name"], "g": a["g"], "mpg": a["min"] / a["g"], "ppg": a["pts"] / a["g"]})
        for t, ps in by.items():
            if t not in out and max(p["g"] for p in ps) >= (3 if s == season else 1):
                out[t] = sorted(ps, key=lambda p: -p["mpg"] * min(p["g"], 10))[:8]
    return out

def injuries():
    if not os.path.exists(f"{D}/injuries.csv"):
        return {}
    return {r["player_id"]: r for r in csv.DictReader(open(f"{D}/injuries.csv")) if r.get("player_id")}

def team_card(t, R, G, season, date, L, P, INJ, name):
    rows = [g for g in G if g["season"] == season and g["date"] < date and t in (g["home"], g["away"])]
    res, w, l, aw, al = [], 0, 0, 0, 0
    for g in rows:
        home = g["home"] == t
        my, op = (g["hpts"], g["apts"]) if home else (g["apts"], g["hpts"])
        win = my > op; w += win; l += not win
        ln = L.get(g["id"], {}); hs = num(ln.get("home_spread"))
        ats = None
        if hs is not None:
            mine = hs if home else -hs
            m = my - op + mine
            ats = "P" if m == 0 else ("W" if m > 0 else "L")
            aw += ats == "W"; al += ats == "L"
        tl = num(ln.get("total")); tp = my + op
        ou = None if tl is None else ("P" if tp == tl else "O" if tp > tl else "U")
        res.append({"date": g["date"], "opp": g["away"] if home else g["home"], "site": "H" if home else "A",
                    "res": "W" if win else "L", "score": f"{int(my)}-{int(op)}", "ats": ats, "ou": ou, "line": hs if home else (-hs if hs is not None else None), "tl": tl})
    o, d = R.eff.get(t, (0, 0))
    return {"abbr": t, "name": name, "record": f"{w}-{l}", "ats": f"{aw}-{al}" if aw + al else "—", "games": len(rows),
            "ortg": rnd(R.mu_e + o), "drtg": rnd(R.mu_e - d), "net": rnd(o + d), "pace": rnd(R.mu_p + 2 * R.pace.get(t, 0)),
            "pts_net": rnd(sum(R.pts.get(t, (0, 0)))),
            "players": [{"name": p["name"], "mpg": rnd(p["mpg"]), "ppg": rnd(p["ppg"]), "g": p["g"],
                         "status": (INJ.get(p["id"]) or {}).get("status")} for p in P.get(t, [])],
            "injuries": [{"name": r["name"], "status": r["status"], "detail": r["detail"]} for r in INJ.values() if r["team"] == t],
            "results": res[-10:]}

def build_slate(day, games, G, L, season, past_rows, P, INJ, odds_pulled):
    R = Ratings(G, season, day, prior_for(G, season))
    rows = []
    last_date = defaultdict(str)
    for g in G:
        if g["season"] == season and g["date"] < day:
            last_date[g["home"]] = max(last_date[g["home"]], g["date"]); last_date[g["away"]] = max(last_date[g["away"]], g["date"])
    def rest(t):
        return (dt.date.fromisoformat(day) - dt.date.fromisoformat(last_date[t])).days if last_date.get(t) else None
    for b in games:
        h, a = b["home"], b["away"]
        hr, ar = rest(h), rest(a)
        p = R.project(h, a, hr, ar)
        hs, tot = num(b.get("home_spread")), num(b.get("total"))
        rows.append({"b": b, **p, "hrest": hr, "arest": ar, "close": -hs if hs is not None else None, "close_total": tot})
    f = calibrate(past_rows, rows)
    try:   # v3 game model inputs (scripts/nba_games_v3.py, games_live.py)
        import nba_games_v3 as NV3
        TF = NV3.live_team_feats(day, season, INJ)
    except Exception as e:
        print("v3 team features failed:", e); TF = None
    ranks = {t: i + 1 for i, t in enumerate(sorted(R.teams, key=lambda t: -sum(R.eff[t])))}
    teams, out = {}, []
    for r in rows:
        b = r["b"]; h, a = b["home"], b["away"]
        dk = {"home_spread": num(b.get("home_spread")), "home_spread_open": num(b.get("home_spread_open")),
              "home_spread_odds": b.get("home_spread_odds"), "away_spread_odds": b.get("away_spread_odds"),
              "total": num(b.get("total")), "total_open": num(str(b.get("total_open") or "").lstrip("ou")),
              "over_odds": b.get("over_odds"), "under_odds": b.get("under_odds"), "home_ml": b.get("home_ml"), "away_ml": b.get("away_ml")}
        model = {"pace": rnd(r["pace"]), "b2b": rnd(r["b2b"]), "eff_raw": rnd(r["eff_margin"]), "pts_raw": rnd(r["pts_margin"]),
                 "eff_total_raw": rnd(r["eff_total"]), "pts_total_raw": rnd(r["pts_total"])}
        side = {"grade": "PASS", "adjust": []}; total = {"grade": "PASS", "adjust": []}
        flags, notes = [], []
        if f:
            apply_cal(r, f)
            model.update({"number": rnd(-r["number"]), "eff_spread": rnd(-r["eff_margin_cal"]), "pts_spread": rnd(-r["pts_margin_cal"]),
                          "total_number": rnd(r["total_number"]), "eff_total": rnd(r["eff_total_cal"]), "pts_total": rnd(r["pts_total_cal"])})
            if dk["home_spread"] is not None:
                close = -dk["home_spread"]
                e = r["number"] - close
                es = [r["eff_margin_cal"] - close, r["pts_margin_cal"] - close]
                gr = grade(e, es, r["side_sd"])
                ps = "home" if e > 0 else "away"
                ln = dk["home_spread"] if ps == "home" else -dk["home_spread"]
                if dk["home_spread_open"] is not None and gr in ("TARGET", "WATCH"):
                    mv = (dk["home_spread"] - dk["home_spread_open"]) * (1 if ps == "home" else -1)
                    if mv >= MOVE_AGAINST:
                        side["adjust"].append(f"line moved {mv:+.1f} against since open"); gr = DOWN[gr]
                side.update({"edge": rnd(e), "edge_eff": rnd(es[0]), "edge_pts": rnd(es[1]), "grade": gr, "pick_side": ps,
                             "pick": f"{h if ps == 'home' else a} {ln:+g}".replace("+0", "pk") if ln else f"{h if ps == 'home' else a} pk",
                             "line": ln, "odds": dk["home_spread_odds"] if ps == "home" else dk["away_spread_odds"]})
            if dk["total"] is not None and r["tot_sd"]:
                e = r["total_number"] - dk["total"]
                es = [r["eff_total_cal"] - dk["total"], r["pts_total_cal"] - dk["total"]]
                gr = grade(e, es, r["tot_sd"], 1.5)
                ps = "over" if e > 0 else "under"
                if dk["total_open"] is not None and gr in ("TARGET", "WATCH"):
                    mv = (dk["total"] - dk["total_open"]) * (1 if ps == "over" else -1)
                    if mv >= MOVE_AGAINST * 1.5:
                        total["adjust"].append(f"total moved {mv:+.1f} against since open"); gr = DOWN[gr]
                total.update({"edge": rnd(e), "edge_eff": rnd(es[0]), "edge_pts": rnd(es[1]), "grade": gr, "pick_side": ps,
                              "pick": f"{'Over' if ps == 'over' else 'Under'} {dk['total']:g}", "line": dk["total"],
                              "odds": dk["over_odds"] if ps == "over" else dk["under_odds"]})
        # stars' status: the ratings assume they play
        for t in (a, h):
            for pl in P.get(t, [])[:2]:
                st = (INJ.get(pl["id"]) or {}).get("status") or ""
                if st and st.lower() in STAR_STATUS:
                    flags.append(f"{pl['name']} ({t}) {st}")
            for pl in P.get(t, [])[2:6]:
                st = (INJ.get(pl["id"]) or {}).get("status") or ""
                if st and st.lower() in STAR_STATUS:
                    notes.append(f"{pl['name']} ({t}, {pl['mpg']:.0f} min) {st}")
        if flags:
            for s_ in (side, total):
                if s_.get("grade") not in (None,):
                    s_["grade_before_flag"] = s_.get("grade"); s_["grade"] = "FLAG"
        if r["hrest"] == 1: notes.append(f"{h} on the second night of a back-to-back")
        if r["arest"] == 1: notes.append(f"{a} on the second night of a back-to-back")
        if (r["hrest"] or 0) >= 3: notes.append(f"{h} rested ({r['hrest'] - 1} days off)")
        if (r["arest"] or 0) >= 3: notes.append(f"{a} rested ({r['arest'] - 1} days off)")
        if R.n_games[h] < 5 or R.n_games[a] < 5:
            notes.append(f"Early season: ratings are mostly last season's ({min(R.n_games[h], R.n_games[a])} games played)")
        k = dt.datetime.fromisoformat(b["kickoff_utc"].replace("Z", "+00:00")).astimezone(ET)
        v3feats = NV3.live_feats(r, h, a, min(R.n_games[h], R.n_games[a]), TF) if TF is not None else None
        out.append({"id": b["game_id"], "kickoff_utc": b["kickoff_utc"], "kick_et": k.strftime("%a %-I:%M %p"), "day_et": day,
                    "away_abbr": a, "home_abbr": h, "away": a, "home": h, "away_name": b.get("away_name"), "home_name": b.get("home_name"),
                    "neutral": bool(b.get("neutral")), "venue": b.get("venue"), "rest": {"home": r["hrest"], "away": r["arest"]},
                    "dk": dk, "model": model, "side": side, "total": total, "flags": flags, "notes": notes,
                    "bands": [rnd(x) for x in bands(r["side_sd"])] if f else None,
                    "tbands": [rnd(x) for x in bands(r["tot_sd"])] if f and r["tot_sd"] else None})
        if v3feats:
            import games_live
            games_live.attach("nba", out[-1], v3feats)
            out[-1]["v3"]["inputs"] = {k: rnd(v, 2) for k, v in v3feats.items() if v is not None}
        for t, nm in ((h, b.get("home_name")), (a, b.get("away_name"))):
            teams[t] = team_card(t, R, G, season, day, L, P, INJ, nm)
            teams[t]["net_rank"] = ranks.get(t)
    calib = {k: [round(v[0], 3), round(v[1], 2)] for k, v in (f or {}).items() if isinstance(v, tuple)}
    if f:
        calib.update({"side_sd": rnd(f["side_sd"], 2), "tot_sd": rnd(f["tot_sd"], 2), "n": f["n"]})
    return out, teams, calib, R

def main():
    T, L = load()
    G = games_of(T)
    board = json.load(open("data/odds/latest_nba.json"))
    season = max(g["season"] for g in G)
    pre = [b for b in board["games"] if b.get("status") == "pre"]
    by_day = defaultdict(list)
    for b in pre:
        k = dt.datetime.fromisoformat(b["kickoff_utc"].replace("Z", "+00:00")).astimezone(ET)
        by_day[k.date().isoformat()].append(b)
    # calibration history: this season and last, every past game with its closing line
    # (only the last ~70 game days are needed: the fit uses the most recent 160 games)
    gdays = sorted({g["date"] for g in G})
    hist = history_rows(G, L, [season - 1, season], since=gdays[-70] if len(gdays) > 70 else "") if by_day else []
    P = players_by_team(season); INJ = injuries()
    os.makedirs(OUT, exist_ok=True)
    idx_path = f"{OUT}/nba_index.json"
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {"slates": []}
    slates = {s["date"]: s for s in idx.get("slates", [])}
    for day in sorted(by_day):
        rows, teams, calib, R = build_slate(day, by_day[day], G, L, season, [r for r in hist if r["date"] < day], P, INJ, board["pulled_utc"])
        finalize_rows(rows, "nba")   # every game gets a pick; PLAY / LEAN tags, the rest untagged; backtest confidence
        path = f"{OUT}/nba_{day}.json"
        rows, teams = merge_started(path, rows, teams)
        rows.sort(key=lambda r: (r["kickoff_utc"], r["id"]))
        doc = {"key": day, "date": day, "season": season, "league": "nba", "odds_pulled_utc": board["pulled_utc"],
               "stats_pulled_utc": json.load(open(f"{D}/_status.json")).get("pulled_utc") if os.path.exists(f"{D}/_status.json") else None,
               "generated_utc": now_iso(), "calibration": calib, "league_avg": {"ortg": rnd(R.mu_e), "pace": rnd(R.mu_p)},
               "games": rows, "teams": teams}
        json.dump(doc, open(path, "w"), separators=(",", ":"))
        g_ = [r for r in rows if not r.get("started")]
        slates[day] = {"date": day, "games": len(rows), "plays": sum(1 for r in rows for m in ("side", "total") if r[m].get("grade") == "PLAY"),
                       "leans": sum(1 for r in rows for m in ("side", "total") if r[m].get("grade") == "LEAN")}
        print(f"nba {day}: {len(rows)} games, {slates[day]['plays']} plays, {slates[day]['leans']} leans")
    # keep slates whose files exist, newest 21 days
    cutoff = (dt.datetime.now(ET).date() - dt.timedelta(days=21)).isoformat()
    idx = {"generated_utc": now_iso(), "odds_pulled_utc": board["pulled_utc"], "season": season,
           "slates": sorted([s for d, s in slates.items() if d >= cutoff and os.path.exists(f"{OUT}/nba_{d}.json")], key=lambda s: s["date"])}
    if os.path.exists(f"{D}/backtest_summary.json"):
        idx["backtest"] = json.load(open(f"{D}/backtest_summary.json"))
    json.dump(idx, open(idx_path, "w"), indent=1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        backtest([int(x) for x in sys.argv[2:]] or [2026, 2027])
    else:
        main()
