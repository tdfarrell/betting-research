#!/usr/bin/env python3
"""College basketball game lines: DraftKings spread/total vs Ballwatcher's ratings -> data/boards/cbb_<date>.json.

  python scripts/regrade_cbb.py               grade every upcoming D-I game on the DK board (one slate per ET day)
  python scripts/regrade_cbb.py backtest      replay 2025-26 vs closing lines -> data/cbb/backtest_summary.json

Scores only (no box scores for 360 teams), D-I vs D-I games, overtime scaled back to 40 minutes.
Two screens, both opponent adjusted with home court (0 at neutral sites):
  season   points scored / allowed, every game this season the same weight
  recent   the same, but each game back counts 0.93x (form, injuries, lineup changes show up sooner)
Each starts from last season's final rating shrunk 50% toward average (college rosters turn over), worth 6 games.
Calibration and tags as the NBA board: fit to recent closing lines + this slate, PLAY 0.6-1.0x the usual gap,
LEAN 0.3-0.6x, both screens on the same side; every game gets a pick (see scripts/picks.py).
"""
import csv, datetime as dt, json, math, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from board_keep import merge_started
from picks import finalize_rows
from regrade_nba import match_fit, bands, grade, rnd, num

D, OUT = "data/cbb", "data/boards"
ET = dt.timezone(dt.timedelta(hours=-4))
PRIOR_GAMES, PRIOR_SHRINK, DECAY = 6.0, 0.5, 0.93
HCA_PRIOR = 3.0
MOVE_AGAINST = 2.0
DOWN = {"PLAY": "LEAN", "LEAN": "PASS", "PASS": "PASS", "CHECK": "CHECK"}

def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")

def load():
    G = [r for r in csv.DictReader(open(f"{D}/games.csv"))]
    L = {r["game_id"]: r for r in csv.DictReader(open(f"{D}/lines_close.csv"))} if os.path.exists(f"{D}/lines_close.csv") else {}
    games = []
    for r in G:
        hs, as_ = num(r["home_score"]), num(r["away_score"])
        if hs is None or as_ is None:
            continue
        sp, tot = num(r.get("home_spread")), num(r.get("total"))
        if sp is None and r["game_id"] in L:
            sp, tot = num(L[r["game_id"]]["home_spread"]), num(L[r["game_id"]]["total"])
        games.append({"id": r["game_id"], "season": int(r["season"]), "date": r["date"], "h": r["home_id"], "a": r["away_id"],
                      "habbr": r["home"], "aabbr": r["away"], "hname": r["home_name"], "aname": r["away_name"],
                      "hpts": hs, "apts": as_, "neutral": r["neutral"] == "1", "conf": r["conf"] == "1", "ot": int(num(r["ot"]) or 0),
                      "close": -sp if sp is not None else None, "close_total": tot})
    # D-I only: teams with 8+ games in a season (non-D-I opponents drop out)
    cnt = defaultdict(int)
    for g in games:
        cnt[(g["season"], g["h"])] += 1; cnt[(g["season"], g["a"])] += 1
    games = [g for g in games if cnt[(g["season"], g["h"])] >= 8 and cnt[(g["season"], g["a"])] >= 8]
    return sorted(games, key=lambda g: (g["date"], g["id"]))

def solve_w(rows, teams, prior, k, hca_prior, k_hca=300.0):
    """y = mu + O[t] - D[o] + hca*home, weighted rows, ridge toward prior."""
    idx = {t: i for i, t in enumerate(teams)}; n = len(teams); P = 2 + 2 * n
    A, b = [], []
    for t, o, hm, y, w in rows:
        sw = math.sqrt(w)
        x = np.zeros(P); x[0] = sw; x[1] = hm * sw; x[2 + idx[t]] = sw; x[2 + n + idx[o]] = -sw
        A.append(x); b.append(y * sw)
    mu0 = float(np.mean([r[3] for r in rows])) if rows else 72.0
    for t in teams:
        po, pd = prior.get(t, (0.0, 0.0)); w = math.sqrt(k)
        x = np.zeros(P); x[2 + idx[t]] = w; A.append(x); b.append(w * po)
        x = np.zeros(P); x[2 + n + idx[t]] = w; A.append(x); b.append(w * pd)
    x = np.zeros(P); x[2:2 + n] = 10; A.append(x); b.append(0.0)
    x = np.zeros(P); x[2 + n:] = 10; A.append(x); b.append(0.0)
    x = np.zeros(P); x[0] = 0.01; A.append(x); b.append(0.01 * mu0)
    x = np.zeros(P); x[1] = math.sqrt(k_hca); A.append(x); b.append(math.sqrt(k_hca) * hca_prior)
    sol = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    return float(sol[0]), float(sol[1]), {t: (float(sol[2 + idx[t]]), float(sol[2 + n + idx[t]])) for t in teams}

class Ratings:
    def __init__(self, G, season, date, prior=None):
        cur = [g for g in G if g["season"] == season and g["date"] < date]
        teams = sorted({g["h"] for g in G if g["season"] == season} | {g["a"] for g in G if g["season"] == season})
        self.teams, self.n = teams, defaultdict(int)
        last = {}
        for g in cur:
            self.n[g["h"]] += 1; self.n[g["a"]] += 1
        pe = prior or {}
        pr = {t: (PRIOR_SHRINK * pe[t][0], PRIOR_SHRINK * pe[t][1]) for t in teams if t in pe}
        rows, rrows = [], []
        # recency: each team's games numbered backwards from its latest
        order = defaultdict(list)
        for g in cur:
            order[g["h"]].append(g["id"]); order[g["a"]].append(g["id"])
        age = {}
        for t, ids in order.items():
            for i, gid in enumerate(reversed(ids)):
                age[(t, gid)] = i
        for g in cur:
            sc = 40 / (40 + 5 * g["ot"]); hm = 0 if g["neutral"] else 1
            for t, o, home, y in ((g["h"], g["a"], hm, g["hpts"] * sc), (g["a"], g["h"], 0, g["apts"] * sc)):
                rows.append((t, o, home, y, 1.0))
                rrows.append((t, o, home, y, DECAY ** min(age.get((t, g["id"]), 0), age.get((o, g["id"]), 0))))
        k = PRIOR_GAMES if pe else 2.0
        self.mu, self.hca, self.r = solve_w(rows, teams, pr, k, HCA_PRIOR)
        self.mu_r, self.hca_r, self.rr = solve_w(rrows, teams, pr, k, HCA_PRIOR)
        if not cur:
            self.mu = self.mu_r = 72.0

    def export(self):
        return dict(self.r)

    def project(self, h, a, neutral):
        hm = 0 if neutral else 1
        oh, dh = self.r.get(h, (0, 0)); oa, da = self.r.get(a, (0, 0))
        ph, pa = self.mu + oh - da + self.hca * hm, self.mu + oa - dh
        roh, rdh = self.rr.get(h, (0, 0)); roa, rda = self.rr.get(a, (0, 0))
        rph, rpa = self.mu_r + roh - rda + self.hca_r * hm, self.mu_r + roa - rdh
        return {"eff_margin": ph - pa, "eff_total": ph + pa, "pts_margin": rph - rpa, "pts_total": rph + rpa}

_PRI = {}
def prior_for(G, season):
    if season not in _PRI:
        prev = [g for g in G if g["season"] == season - 1]
        if not prev:
            _PRI[season] = None
        else:
            nxt = (dt.date.fromisoformat(max(g["date"] for g in prev)) + dt.timedelta(days=1)).isoformat()
            _PRI[season] = Ratings(G, season - 1, nxt, prior_for(G, season - 1)).export()
    return _PRI[season]

def calibrate(past, cur, n=300, min_n=80):
    sample = [r for r in past if r.get("close") is not None][-max(n - len(cur), 60):] + [r for r in cur if r.get("close") is not None]
    if len(sample) < min_n:
        return None
    y = np.array([r["close"] for r in sample])
    f = {m: match_fit([r[m] for r in sample], y) for m in ("eff_margin", "pts_margin")}
    bl = np.array([np.mean([f[m][0] * r[m] + f[m][1] for m in ("eff_margin", "pts_margin")]) for r in sample])
    f["blend"] = match_fit(bl, y)
    f["side_sd"] = float(np.sqrt(np.mean((f["blend"][0] * bl + f["blend"][1] - y) ** 2)))
    ts = [r for r in sample if r.get("close_total") is not None]
    if len(ts) >= min_n // 2:
        yt = np.array([r["close_total"] for r in ts])
        f.update({m: match_fit([r[m] for r in ts], yt) for m in ("eff_total", "pts_total")})
        tb = np.array([np.mean([f[m][0] * r[m] + f[m][1] for m in ("eff_total", "pts_total")]) for r in ts])
        f["tblend"] = match_fit(tb, yt)
        f["tot_sd"] = float(np.sqrt(np.mean((f["tblend"][0] * tb + f["tblend"][1] - yt) ** 2)))
    else:
        f["tot_sd"] = None
    f["n"] = len(sample)
    return f

def apply_cal(r, f):
    for m in ("eff_margin", "pts_margin"):
        r[m + "_cal"] = f[m][0] * r[m] + f[m][1]
    r["number"] = f["blend"][0] * (r["eff_margin_cal"] + r["pts_margin_cal"]) / 2 + f["blend"][1]
    if f.get("tot_sd"):
        for m in ("eff_total", "pts_total"):
            r[m + "_cal"] = f[m][0] * r[m] + f[m][1]
        r["total_number"] = f["tblend"][0] * (r["eff_total_cal"] + r["pts_total_cal"]) / 2 + f["tblend"][1]
    r["side_sd"], r["tot_sd"] = f["side_sd"], f.get("tot_sd")

def history_rows(G, seasons, since=""):
    rows = []
    for s in seasons:
        pri = prior_for(G, s)
        days = sorted({g["date"] for g in G if g["season"] == s and g["date"] >= since})
        for d in days:
            R = Ratings(G, s, d, pri)
            for g in (x for x in G if x["season"] == s and x["date"] == d):
                rows.append({"id": g["id"], "season": s, "date": d, **R.project(g["h"], g["a"], g["neutral"]),
                             "close": g["close"], "close_total": g["close_total"], "margin": g["hpts"] - g["apts"],
                             "total": g["hpts"] + g["apts"], "gp": min(R.n[g["h"]], R.n[g["a"]])})
    return rows

def backtest(seasons=(2026,)):
    G = load()
    rows = history_rows(G, range(min(seasons), max(seasons) + 1))
    done, out = [], []
    by = defaultdict(list)
    for r in rows:
        by[(r["season"], r["date"])].append(r)
    for key in sorted(by):
        cur = by[key]; f = calibrate(done, cur)
        if f:
            for r in cur:
                if r["close"] is None:
                    continue
                apply_cal(r, f)
                e = r["number"] - r["close"]
                r["side_grade"] = grade(e, [r["eff_margin_cal"] - r["close"], r["pts_margin_cal"] - r["close"]], r["side_sd"])
                r["side_res"] = None if r["margin"] == r["close"] else int((r["margin"] - r["close"]) * (1 if e > 0 else -1) > 0)
                r["side_gap"] = abs(e)
                if r["close_total"] is not None and r.get("tot_sd"):
                    et = r["total_number"] - r["close_total"]
                    r["tot_grade"] = grade(et, [r["eff_total_cal"] - r["close_total"], r["pts_total_cal"] - r["close_total"]], r["tot_sd"], 1.5)
                    r["tot_res"] = None if r["total"] == r["close_total"] else int((r["total"] - r["close_total"]) * (1 if et > 0 else -1) > 0)
                out.append(r)
        done += cur
    def rec(rs, k):
        w = sum(1 for r in rs if r.get(k) == 1); l = sum(1 for r in rs if r.get(k) == 0)
        return {"w": w, "l": l, "n": len(rs), "pct": round(w / (w + l), 3) if w + l else None}
    NAME = {"TARGET": "PLAY", "WATCH": "LEAN"}
    # every graded game, for the confidence tables (scripts/confidence.py)
    import csv as _csv
    with open(f"{D}/backtest_games.csv", "w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(["season", "date", "side_edge", "side_sd", "side_res", "tot_edge", "tot_sd", "tot_res",
                                          "game_id", "number", "total_number", "close", "close_total", "margin", "total"])
        for r in out:
            w.writerow([r["season"], r["date"], rnd(r["number"] - r["close"], 2), rnd(r["side_sd"], 2), r.get("side_res"),
                        rnd(r["total_number"] - r["close_total"], 2) if r.get("tot_res") is not None else "", rnd(r.get("tot_sd"), 2) if r.get("tot_sd") else "", r.get("tot_res"),
                        r["id"], rnd(r["number"], 2), rnd(r.get("total_number"), 2), r["close"], r.get("close_total"), r["margin"], r["total"]])
    summ = {"seasons": [min(seasons), max(seasons)], "games": len(out), "line": "closing line (ESPN core odds)",
            "rmse": {k: rnd(math.sqrt(np.mean([(r[k] - r["margin"]) ** 2 for r in out])), 2) for k in ("close", "number")} if out else {},
            "tiers_vs_close": {"side": {}, "total": {}}, "side_by_gap": {}}
    for t in ("TARGET", "WATCH", "CHECK"):
        nm = NAME.get(t, "BIG GAP (untagged)")
        summ["tiers_vs_close"]["side"][nm] = rec([r for r in out if r.get("side_grade") == t], "side_res")
        summ["tiers_vs_close"]["total"][nm] = rec([r for r in out if r.get("tot_grade") == t], "tot_res")
    summ["tiers_vs_close"]["side"]["ALL"] = rec(out, "side_res")
    summ["tiers_vs_close"]["total"]["ALL"] = rec([r for r in out if r.get("tot_res") is not None], "tot_res")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 99)):
        summ["side_by_gap"][f"{lo}-{hi if hi < 99 else '+'}"] = rec([r for r in out if lo <= r.get("side_gap", -1) < hi], "side_res")
    os.makedirs(D, exist_ok=True)
    json.dump(summ, open(f"{D}/backtest_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))

def team_card(tid, R, G, season, date, ranks):
    rows = [g for g in G if g["season"] == season and g["date"] < date and tid in (g["h"], g["a"])]
    res, w, l, aw, al = [], 0, 0, 0, 0
    abbr, name = tid, tid
    for g in rows:
        home = g["h"] == tid
        abbr, name = (g["habbr"], g["hname"]) if home else (g["aabbr"], g["aname"])
        my, op = (g["hpts"], g["apts"]) if home else (g["apts"], g["hpts"])
        win = my > op; w += win; l += not win
        ats = None
        if g["close"] is not None:
            m = (my - op) - (g["close"] if home else -g["close"])
            ats = "P" if m == 0 else ("W" if m > 0 else "L"); aw += ats == "W"; al += ats == "L"
        res.append({"date": g["date"], "opp": g["aabbr"] if home else g["habbr"], "site": "N" if g["neutral"] else ("H" if home else "A"),
                    "res": "W" if win else "L", "score": f"{int(my)}-{int(op)}", "ats": ats})
    o, d = R.r.get(tid, (0, 0)); ro, rd = R.rr.get(tid, (0, 0))
    return {"id": tid, "abbr": abbr, "name": name, "record": f"{w}-{l}", "ats": f"{aw}-{al}" if aw + al else "—", "games": len(rows),
            "off": rnd(R.mu + o), "def": rnd(R.mu - d), "net": rnd(o + d), "recent_net": rnd(ro + rd), "net_rank": ranks.get(tid),
            "results": res[-10:]}

def main():
    G = load()
    board = json.load(open("data/odds/latest_cbb.json")) if os.path.exists("data/odds/latest_cbb.json") else {"games": [], "pulled_utc": None}
    season = max(g["season"] for g in G)
    today = dt.datetime.now(ET).date()
    if today.month >= 7:
        season = max(season, today.year + 1)   # before the opener, the new season has no games yet
    by_day = defaultdict(list)
    for b in board["games"]:
        if b.get("status") == "pre" and b.get("home_spread") not in (None, ""):
            by_day[dt.datetime.fromisoformat(b["kickoff_utc"].replace("Z", "+00:00")).astimezone(ET).date().isoformat()].append(b)
    idx_path = f"{OUT}/cbb_index.json"
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {"slates": []}
    slates = {s["date"]: s for s in idx.get("slates", [])}
    gdays = sorted({g["date"] for g in G})
    hist = history_rows(G, [season - 1, season], since=gdays[-45] if len(gdays) > 45 else "") if by_day else []
    # ESPN team id lookup for the board's abbreviations (board rows carry abbreviations and names)
    ab2id = {}
    for g in G:
        ab2id[g["habbr"]] = g["h"]; ab2id[g["aabbr"]] = g["a"]
    for day in sorted(by_day):
        R = Ratings(G, season, day, prior_for(G, season))
        ranks = {t: i + 1 for i, t in enumerate(sorted(R.teams or list(R.r), key=lambda t: -sum(R.r[t])))}
        cur = []
        for b in by_day[day]:
            h, a = ab2id.get(b["home"]), ab2id.get(b["away"])
            if not h or not a:
                continue
            p = R.project(h, a, bool(b.get("neutral")))
            hs, tot = num(b.get("home_spread")), num(b.get("total"))
            cur.append({"b": b, "h": h, "a": a, **p, "close": -hs if hs is not None else None, "close_total": tot})
        f = calibrate([r for r in hist if r["date"] < day], cur)
        rows, teams = [], {}
        for r in cur:
            b = r["b"]
            dk = {"home_spread": num(b.get("home_spread")), "home_spread_open": num(b.get("home_spread_open")),
                  "home_spread_odds": b.get("home_spread_odds"), "away_spread_odds": b.get("away_spread_odds"),
                  "total": num(b.get("total")), "total_open": num(str(b.get("total_open") or "").lstrip("ou")),
                  "over_odds": b.get("over_odds"), "under_odds": b.get("under_odds"), "home_ml": b.get("home_ml"), "away_ml": b.get("away_ml")}
            model = {"season_raw": rnd(r["eff_margin"]), "recent_raw": rnd(r["pts_margin"])}
            side, total = {"grade": "PASS", "adjust": []}, {"grade": "PASS", "adjust": []}
            if f:
                apply_cal(r, f)
                model.update({"number": rnd(-r["number"]), "season_spread": rnd(-r["eff_margin_cal"]), "recent_spread": rnd(-r["pts_margin_cal"])})
                if r.get("total_number") is not None:
                    model.update({"total_number": rnd(r["total_number"]), "season_total": rnd(r["eff_total_cal"]), "recent_total": rnd(r["pts_total_cal"])})
                if dk["home_spread"] is not None:
                    c = -dk["home_spread"]; e = r["number"] - c
                    gr = grade(e, [r["eff_margin_cal"] - c, r["pts_margin_cal"] - c], r["side_sd"])
                    gr = {"TARGET": "PLAY", "WATCH": "LEAN"}.get(gr, gr)
                    if dk["home_spread_open"] is not None and gr in ("PLAY", "LEAN"):
                        mv = (dk["home_spread"] - dk["home_spread_open"]) * (1 if e > 0 else -1)
                        if mv >= MOVE_AGAINST:
                            side["adjust"].append(f"line moved {mv:+.1f} against since open"); gr = DOWN[gr]
                    side.update({"edge": rnd(e), "edge_season": rnd(r["eff_margin_cal"] - c), "edge_recent": rnd(r["pts_margin_cal"] - c), "grade": gr})
                if dk["total"] is not None and r.get("tot_sd"):
                    e = r["total_number"] - dk["total"]
                    gr = grade(e, [r["eff_total_cal"] - dk["total"], r["pts_total_cal"] - dk["total"]], r["tot_sd"], 1.5)
                    gr = {"TARGET": "PLAY", "WATCH": "LEAN"}.get(gr, gr)
                    total.update({"edge": rnd(e), "edge_season": rnd(r["eff_total_cal"] - dk["total"]), "edge_recent": rnd(r["pts_total_cal"] - dk["total"]), "grade": gr})
            notes = []
            if b.get("neutral"): notes.append("Neutral site: no home court")
            if R.n[r["h"]] < 5 or R.n[r["a"]] < 5:
                notes.append(f"Early season: ratings lean on last season ({min(R.n[r['h']], R.n[r['a']])} games played); rosters turn over a lot in college")
            k = dt.datetime.fromisoformat(b["kickoff_utc"].replace("Z", "+00:00")).astimezone(ET)
            rows.append({"id": b["game_id"], "kickoff_utc": b["kickoff_utc"], "kick_et": k.strftime("%a %-I:%M %p"), "day_et": day,
                         "away_abbr": b["away"], "home_abbr": b["home"], "away": b["away"], "home": b["home"],
                         "away_name": b.get("away_name"), "home_name": b.get("home_name"), "neutral": bool(b.get("neutral")),
                         "conference_game": bool(b.get("conference_game")), "venue": b.get("venue"),
                         "dk": dk, "model": model, "side": side, "total": total, "flags": [], "notes": notes,
                         "bands": [rnd(x) for x in bands(r["side_sd"])] if f else None,
                         "tbands": [rnd(x) for x in bands(r["tot_sd"])] if f and r.get("tot_sd") else None})
            for tid, ab in ((r["h"], b["home"]), (r["a"], b["away"])):
                teams[ab] = team_card(tid, R, G, season, day, ranks)
        if not rows:
            continue
        finalize_rows(rows, "cbb")
        path = f"{OUT}/cbb_{day}.json"
        rows, teams = merge_started(path, rows, teams)
        calib = {k: [round(v[0], 3), round(v[1], 2)] for k, v in (f or {}).items() if isinstance(v, tuple)}
        if f:
            calib.update({"side_sd": rnd(f["side_sd"], 2), "tot_sd": rnd(f.get("tot_sd"), 2), "n": f["n"]})
        json.dump({"key": day, "date": day, "season": season, "league": "cbb", "odds_pulled_utc": board["pulled_utc"], "generated_utc": now_iso(),
                   "stats_pulled_utc": json.load(open(f"{D}/_status.json")).get("pulled_utc") if os.path.exists(f"{D}/_status.json") else None,
                   "calibration": calib, "games": rows, "teams": teams, "teams_rated": len(R.teams)}, open(path, "w"), separators=(",", ":"))
        slates[day] = {"date": day, "games": len(rows), "plays": sum(1 for g in rows for m in ("side", "total") if g[m].get("grade") == "PLAY"),
                       "leans": sum(1 for g in rows for m in ("side", "total") if g[m].get("grade") == "LEAN")}
        print(f"cbb {day}: {len(rows)} games, {slates[day]['plays']} plays, {slates[day]['leans']} leans")
    cutoff = (today - dt.timedelta(days=21)).isoformat()
    idx = {"generated_utc": now_iso(), "odds_pulled_utc": board["pulled_utc"], "season": season,
           "slates": sorted([s for d_, s in slates.items() if d_ >= cutoff and os.path.exists(f"{OUT}/cbb_{d_}.json")], key=lambda s: s["date"])}
    if os.path.exists(f"{D}/backtest_summary.json"):
        idx["backtest"] = json.load(open(f"{D}/backtest_summary.json"))
    os.makedirs(OUT, exist_ok=True)
    json.dump(idx, open(idx_path, "w"), indent=1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        backtest(tuple(int(x) for x in sys.argv[2:]) or (2026,))
    else:
        main()
