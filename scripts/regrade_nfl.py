#!/usr/bin/env python3
"""Grade NFL weeks vs DraftKings using EPA, Elo and points ratings (QB adjusted), plus matchup notes.

Usage:
  python scripts/regrade_nfl.py            grade every week with games on the DK board

Writes data/boards/nfl_<season>_wk<NN>.json + .csv, data/boards/nfl_index.json, and logs every
TARGET/WATCH the first time it appears in data/nfl/flags.csv, graded later on CLV vs the DK close.

Method (see the project's NFL handicapping checklist):
  Sides   Three CURRENT-SEASON screens (EPA, success rate, points; opponent adjusted, QB adjusted), each
          fitted to recent closing lines and averaged into the Ballwatcher number. Last season only seeds
          the ratings in Weeks 1-5 and fades to zero by Week 6.
  Angles  Last season's rating, multi-season Elo and the market rating are shown for context only;
          they never change a grade.
          Calibration is "fair" to favorites and dogs: an edge can't be predicted from the size of the line.
          Gap sizes are measured against how far the BW number has recently strayed from the close
          ("usual gap", ~4 pts early season, ~3 late), so tiers mean the same thing all season:
          WATCH  = 0.3-0.6 x usual gap and at least two screens agree (each 1+ pts, same side).
          TARGET = 0.6-1.0 x usual gap and at least two screens agree.
          CHECK  = more than the usual gap: the market usually knows something (injury, QB, news). Shown with
                   the model's lean, not as a pick: check the news first.
          Backtest 2022-25 vs the close: WATCH 91-65 (58%), TARGET 115-86 (57%), CHECK 160-215 (43%);
          favorites ~half of all flags. Every flag is logged and judged on CLV vs the DK close.
  Totals  EPA total (EPA x pace, dome) and points-rating total, put on the market's scale the same way;
          BW total = their average. TARGET = both 1.5+ same way and BW total 2-4.5 off DK;
          WATCH = both 1+ same way and 1.5-2 off; CHECK = 4.5+ off. Totals have NOT shown an edge in
          the backtest (TARGET 76-64, other bands ~50%): treat them as unproven until the CLV log says otherwise.
  Adjust  Line moved 2+ toward the other side since open: one tier down.
          Projected starting QB questionable/doubtful/out: not graded (FLAG).
          Overs with 15+ mph wind or rain and a falling total: PASS.
"""
import csv, datetime as dt, json, os, re, sys
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(__file__))
from picks import finalize_rows
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
import nfl_model as M
from board_keep import merge_started

ET = dt.timezone(dt.timedelta(hours=-4))  # EDT; good through early November
ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}
DOWN = {"TARGET": "WATCH", "WATCH": "PASS", "PASS": "PASS", "FLAG": "FLAG", "CHECK": "CHECK"}
GAP_CHECK_TOTAL = 4.5
MOVE_AGAINST = 2.0
FLAGS = "data/nfl/flags.csv"
FLAG_COLS = ["game_id", "season", "week", "kickoff_utc", "game", "market", "pick_side", "pick", "first_grade", "best_grade",
             "flagged_utc", "line_at_flag", "odds_at_flag", "our_number", "close_line", "clv", "home_score", "away_score",
             "result", "status"]

def num(x):
    try:
        return float(str(x).lstrip("ou"))
    except (TypeError, ValueError):
        return None

def rnd(x, n=1):
    return None if x is None else round(float(x), n)

def side(x):
    return "home" if x > 0 else "away"

def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")

# ---------------------------------------------------------------- data
data = M.load()
SCHED, TG, QG = data
CUR = max(g["season"] for g in SCHED if g["home_score"] not in ("", None))
ROWS, CTX = M.predict(CUR - 1, CUR, data, with_matchups=True)
M.calibrate(ROWS)
BY_ESPN = {r["espn"]: r for r in ROWS if r["season"] == CUR}
SCHED_BY_ESPN = {g["espn"]: g for g in SCHED if g["season"] == CUR}
STATUS = json.load(open("data/nfl/_status.json")) if os.path.exists("data/nfl/_status.json") else {}
INJ = [r for r in csv.DictReader(open("data/nfl/injuries.csv"))] if os.path.exists("data/nfl/injuries.csv") else []
INJ_BY_ID = {r["gsis_id"]: r for r in INJ if r.get("gsis_id")}
BACKTEST = json.load(open("data/nfl/backtest_summary.json")) if os.path.exists("data/nfl/backtest_summary.json") else None

def qb_status(qb_id):
    r = INJ_BY_ID.get(qb_id)
    if not r:
        return None
    st = (r.get("report_status") or "").strip()
    return st if st in ("Questionable", "Doubtful", "Out") else None

# ---------------------------------------------------------------- team cards
def season_rows(team, week, opp=False):
    return [r for r in TG if r["season"] == CUR and r["week"] < week and (r["opp"] if opp else r["team"]) == team]

def rate(rows, num_k, den_k):
    n = sum(r[num_k] for r in rows); d = sum(r[den_k] for r in rows)
    return n / d if d else None

def ranks(d, reverse=True):
    order = sorted(d, key=lambda t: d[t], reverse=reverse)
    return {t: k + 1 for k, t in enumerate(order)}

def week_context(season, week):
    c = CTX[(season, week)]
    er, pr, mr = c["epa"], c["pts"], c["mkt"]
    net = {t: (er["off"][t] - er["def"][t]) * M.P.plays for t in er["off"]}
    last = c["last"]
    last_net = {t: (last["off"][t] - last["def"][t]) * M.P.plays for t in last["off"]}
    srn = {t: (c["sr"]["off"][t] - c["sr"]["def"][t]) * 100 for t in c["sr"]["off"]}
    return {"c": c, "net": net, "last_net": last_net, "last_rank": ranks(last_net), "sr_net": srn, "sr_rank": ranks(srn), "net_rank": ranks(net), "off_rank": ranks(er["off"]), "def_rank": ranks(er["def"], reverse=False),
            "mkt_rank": ranks(mr["r"]), "elo_rank": ranks(c["elo"])}

def results(team, week):
    out = []
    for g in SCHED:
        if g["season"] != CUR or g["week"] >= week or g["home_score"] in ("", None) or team not in (g["home_team"], g["away_team"]):
            continue
        home = g["home_team"] == team
        pf, pa = (float(g["home_score"]), float(g["away_score"])) if home else (float(g["away_score"]), float(g["home_score"]))
        cl = M.fnum(g["spread_line"])
        ats = None
        if cl is not None:
            m = (pf - pa) - (cl if home else -cl)
            ats = "W" if m > 0 else "L" if m < 0 else "P"
        out.append({"wk": g["week"], "opp": g["away_team"] if home else g["home_team"],
                    "site": "N" if g["neutral"] else ("H" if home else "A"),
                    "res": "W" if pf > pa else "L" if pf < pa else "T", "score": f"{int(pf)}-{int(pa)}", "ats": ats,
                    "ou": (lambda tl: None if tl is None else ("P" if pf + pa == tl else "O" if pf + pa > tl else "U"))(M.fnum(g.get("total_line")))})
    return sorted(out, key=lambda r: r["wk"])

def team_card(t, name, week, W, qb_id):
    c, er, mt = W["c"], W["c"]["epa"], W["c"]["match"]
    o, d = season_rows(t, week), season_rows(t, week, opp=True)
    qv, mix, names = c["qb_val"], c["qb_mix"], c["qb_names"]
    res = results(t, week)
    return {
        "name": name, "abbr": t,
        "record": f"{sum(r['res'] == 'W' for r in res)}-{sum(r['res'] == 'L' for r in res)}" + (f"-{sum(r['res'] == 'T' for r in res)}" if any(r["res"] == "T" for r in res) else ""),
        "ats": f"{sum(r['ats'] == 'W' for r in res)}-{sum(r['ats'] == 'L' for r in res)}",
        "net": rnd(W["net"][t]), "net_rank": W["net_rank"][t], "last_net": rnd(W["last_net"].get(t)),
        "last_rank": W["last_rank"].get(t), "sr_net": rnd(W["sr_net"][t], 1), "sr_rank": W["sr_rank"][t], "off_rank": W["off_rank"][t], "def_rank": W["def_rank"][t],
        "elo": rnd(c["elo"].get(t), 0), "elo_rank": W["elo_rank"].get(t), "mkt": rnd(c["mkt"]["r"].get(t)), "mkt_rank": W["mkt_rank"].get(t),
        "qb": names.get(qb_id) or None, "qb_val": rnd(qv.get(qb_id), 3) if qb_id in qv else None,
        "off": {"epa": rnd(er["mu"] + er["off"][t], 3), "pass_epa": rnd(mt["pass"]["mu"] + mt["pass"]["off"][t], 3),
                "rush_epa": rnd(mt["rush"]["mu"] + mt["rush"]["off"][t], 3), "sr": rnd(rate(o, "success", "plays"), 3),
                "expl": rnd(rate(o, "explosive", "plays"), 3), "sack": rnd(rate(o, "sacks", "pass_plays"), 3),
                "pace": rnd(rate(o, "neutral_secs", "neutral_plays"), 1),
                "proe": rnd(rate(o, "pass_oe_sum", "pass_oe_n"), 1), "third": rnd(rate(o, "third_conv", "third_att"), 3),
                "rz": rnd(rate(o, "rz_td_drives", "rz_drives"), 3), "giveaways": int(sum(r["giveaways"] for r in o))},
        "def": {"epa": rnd(er["mu"] + er["def"][t], 3), "pass_epa": rnd(mt["pass"]["mu"] + mt["pass"]["def"][t], 3),
                "rush_epa": rnd(mt["rush"]["mu"] + mt["rush"]["def"][t], 3), "sr": rnd(rate(d, "success", "plays"), 3),
                "expl": rnd(rate(d, "explosive", "plays"), 3), "sack": rnd(rate(d, "sacks", "pass_plays"), 3),
                "third": rnd(rate(d, "third_conv", "third_att"), 3), "rz": rnd(rate(d, "rz_td_drives", "rz_drives"), 3),
                "takeaways": int(sum(r["giveaways"] for r in d))},
        "results": res,
    }

def notes(h, a, T, g, r):
    """Matchup flags. Adjusted EPA/play; defensive numbers = what offenses gain vs that D (higher = worse)."""
    out = []
    for off, de in ((h, a), (a, h)):
        O, D = T[off], T[de]
        for k, label, hi, lo in (("pass_epa", "pass", 0.20, -0.15), ("rush_epa", "run", 0.10, -0.18)):
            x, y = O["off"][k], D["def"][k]
            if x is None or y is None:
                continue
            if x + y >= hi:
                out.append(f"{off} {label} edge ({label} EPA {x:+.2f}; {de} allows {y:+.2f})")
            elif x + y <= lo:
                out.append(f"{de} {label} D shuts down {off} ({label} EPA {x:+.2f}; allows {y:+.2f})")
        if (O["off"]["sack"] or 0) >= 0.08 and (D["def"]["sack"] or 0) >= 0.08:
            out.append(f"{de} pass rush vs {off} protection (sacked on {O['off']['sack']:.0%}; {de} sacks {D['def']['sack']:.0%})")
        if O["off"]["rz"] is not None and D["def"]["rz"] is not None and O["off"]["rz"] + D["def"]["rz"] >= 1.35:
            out.append(f"{off} should finish drives (RZ TD {O['off']['rz']:.0%}; {de} allows {D['def']['rz']:.0%})")
    hr, ar = M.fnum(g["home_rest"], 7), M.fnum(g["away_rest"], 7)
    if hr >= 13 or ar >= 13:
        out.append(f"Off a bye: {', '.join(t for t, x in ((h, hr), (a, ar)) if x >= 13)}")
    elif abs(hr - ar) >= 3:
        out.append(f"Rest edge: {h if hr > ar else a} ({int(max(hr, ar))} days vs {int(min(hr, ar))})")
    if hr <= 4 and ar <= 4:
        out.append("Short week for both (Thursday)")
    if g["div_game"] == "1":
        out.append("Divisional game")
    for t, q, qid in ((h, r["qb_h"], g["home_qb_id"]), (a, r["qb_a"], g["away_qb_id"])):
        prev = last_qb(t, g["week"])
        if prev and qid and prev[0] != qid:
            out.append(f"QB change: {T[t]['qb'] or 'new starter'} projected for {t}; {prev[1]} started last game (rating {q:+.1f} pts)")
        elif abs(q) >= 1.0:
            out.append(f"QB adjustment {t} {q:+.1f} pts ({T[t]['qb']} vs the QBs behind {t}'s rating, incl. last season)")
    return out

def last_qb(team, week):
    """(qb_id, name) with the most dropbacks in the team's most recent game this season."""
    rows = [q for q in QG if q["team"] == team and q["season"] == CUR and q["week"] < week]
    if not rows:
        return None
    wk = max(q["week"] for q in rows)
    top = max((q for q in rows if q["week"] == wk), key=lambda q: q["dropbacks"])
    return top["qb_id"], top["qb"]

# ---------------------------------------------------------------- grading
def side_bands(sd):
    """Point gaps for WATCH / TARGET / CHECK, scaled to how far the BW number has recently strayed from
    the close (sd). Early season the ratings are looser, so the same gap means less."""
    sd = sd or 3.3
    return 0.3 * sd, 0.6 * sd, 1.0 * sd

def grade_side(e, es, sd=None):
    w, t, c = side_bands(sd)
    agree = [x for x in es if x is not None and abs(x) >= 1 and side(x) == side(e)]
    if abs(e) >= c:
        return "CHECK"
    if abs(e) >= t and len(agree) >= 2:
        return "TARGET"
    if abs(e) >= w and len(agree) >= 2:
        return "WATCH"
    return "PASS"

def grade_total(e1, e2):
    if e1 is None or e2 is None:
        return "PASS"
    t = (e1 + e2) / 2
    if abs(t) >= GAP_CHECK_TOTAL:
        return "CHECK"
    if e1 * e2 <= 0:
        return "PASS"
    if min(abs(e1), abs(e2)) >= 1.5 and abs(t) >= 2:
        return "TARGET"
    if min(abs(e1), abs(e2)) >= 1 and abs(t) >= 1.5:
        return "WATCH"
    return "PASS"

def wind_mph(s):
    m = re.search(r"(\d+)\s*mph", s or "", re.I)
    return int(m.group(1)) if m else None

def build(games, season, week):
    W = week_context(season, week)
    rows, teams, unmatched = [], {}, []
    for bg in games:
        r = BY_ESPN.get(bg["game_id"])
        g = SCHED_BY_ESPN.get(bg["game_id"])
        dk_hs, dk_tot = num(bg.get("home_spread")), num(bg.get("total"))
        if r is None or g is None or r.get("blend") is None:
            unmatched.append(f"{bg['away']} @ {bg['home']}"); continue
        if dk_hs is None or dk_tot is None:
            continue
        h, a = r["home"], r["away"]
        for t, nm, q, qn in ((h, bg["home_name"], g["home_qb_id"], g["home_qb_name"]), (a, bg["away_name"], g["away_qb_id"], g["away_qb_name"])):
            teams[t] = team_card(t, nm, week, W, q)
            teams[t]["qb"] = qn or teams[t]["qb"]
        dk_line = -dk_hs  # home margin DK implies
        e = r["blend"] - dk_line
        es = [r[m + "_cal"] - dk_line for m in M.SIDE_MODELS]
        flags, sadj, tadj = [], [], []
        for t, q in ((h, g["home_qb_id"]), (a, g["away_qb_id"])):
            st = qb_status(q)
            if st:
                flags.append(f"{t} QB {teams[t]['qb'] or ''} {st.lower()}".replace("  ", " "))
        sg = "FLAG" if flags else grade_side(e, es, r.get("side_sd"))
        side_d = {"edge": rnd(e), "edge_epa": rnd(es[0]), "edge_sr": rnd(es[1]), "edge_pts": rnd(es[2]), "grade": sg,
                  "pick": None, "adjust": [], "keys": []}
        open_hs = num(bg.get("home_spread_open"))
        if sg in ("TARGET", "WATCH", "CHECK"):
            hp = e > 0
            line = dk_hs if hp else -dk_hs
            side_d.update(pick=f"{bg['home' if hp else 'away']} {line:+g}", pick_side="home" if hp else "away", line=line,
                          odds=bg.get("home_spread_odds" if hp else "away_spread_odds"))
            if sg == "CHECK":
                side_d["adjust"].append(f"BW number is {abs(e):.1f} pts off DK (usual gap {r.get('side_sd') or 0:.1f}): big gaps usually mean news the model doesn't have. Lean only; check injuries/QB first")
            elif open_hs is not None:
                moved = dk_hs - open_hs  # + = toward away
                against = moved if hp else -moved
                if against >= MOVE_AGAINST:
                    side_d["adjust"].append(f"Line moved {against:g} pts toward the other side since open")
                    side_d["grade"] = DOWN[side_d["grade"]]
                    if side_d["grade"] == "PASS":
                        side_d["pick"] = None
        side_d["keys"] = [k for k in (3, 7) if min(abs(r["blend"]), abs(dk_line)) < k < max(abs(r["blend"]), abs(dk_line))
                          and r["blend"] * dk_line > 0]
        t1, t2 = r["epa_tot_cal"] - dk_tot, r["pts_tot_cal"] - dk_tot
        tg = "FLAG" if flags else grade_total(t1, t2)
        tavg = (t1 + t2) / 2
        tot_d = {"edge": rnd(tavg), "edge_epa": rnd(t1), "edge_pts": rnd(t2), "grade": tg, "pick": None, "adjust": []}
        if tg in ("TARGET", "WATCH", "CHECK"):
            over = tavg > 0
            tot_d.update(pick=f"{'Over' if over else 'Under'} {dk_tot:g}", pick_side="over" if over else "under", line=dk_tot,
                         odds=bg.get("over_odds" if over else "under_odds"))
            wx = (bg.get("weather") or "")
            wm = wind_mph(wx)
            topen = num(bg.get("total_open"))
            if tg == "CHECK":
                tot_d["adjust"].append(f"BW total is {abs(tavg):.1f} off DK: big gaps usually mean weather or news the model doesn't have. Lean only")
            elif over and ((wm and wm >= 15) or (any(k in wx.lower() for k in ("rain", "snow", "storm")) and topen and topen - dk_tot >= 3)):
                tot_d["adjust"].append(f"Weather: {wx}; models don't see it")
                tot_d["grade"], tot_d["pick"] = "PASS", None
        k = dt.datetime.fromisoformat(bg["kickoff_utc"].replace("Z", "+00:00")).astimezone(ET)
        rows.append({
            "id": bg["game_id"], "nflverse_id": r["game_id"], "kickoff_utc": bg["kickoff_utc"],
            "kick_et": k.strftime("%a %-I:%M %p"), "day_et": k.date().isoformat(),
            "away_abbr": bg["away"], "home_abbr": bg["home"], "away": a, "home": h, "away_name": bg["away_name"], "home_name": bg["home_name"],
            "neutral": bool(r["neutral"]), "venue": bg.get("venue"), "weather": bg.get("weather"), "temp_f": bg.get("temp_f"),
            "roof": g["roof"], "surface": g["surface"], "div_game": g["div_game"] == "1",
            "rest": {"home": M.fnum(g["home_rest"]), "away": M.fnum(g["away_rest"])},
            "dk": {"home_spread": dk_hs, "home_spread_open": open_hs, "home_spread_odds": bg.get("home_spread_odds"),
                   "away_spread_odds": bg.get("away_spread_odds"), "total": dk_tot, "total_open": num(bg.get("total_open")),
                   "over_odds": bg.get("over_odds"), "under_odds": bg.get("under_odds"), "home_ml": bg.get("home_ml"), "away_ml": bg.get("away_ml")},
            "model": {"number": rnd(-r["blend"]), "epa_spread": rnd(-r["epa_cal"]), "sr_spread": rnd(-r["sr_cal"]),
                      "pts_spread": rnd(-r["pts_cal"]), "last_spread": rnd(-r["last_cal"]), "elo_spread": rnd(-r["elo_cal"]),
                      "mkt_spread": rnd(-r["mkt_cal"]), "prior_weight": rnd(W["c"]["prior_weight"], 2),
                      "qb_adj": rnd(r["qb_adj"]), "epa_total": rnd(r["epa_tot_cal"]), "pts_total": rnd(r["pts_tot_cal"]),
                      "total_number": rnd((r["epa_tot_cal"] + r["pts_tot_cal"]) / 2)},
            "side": side_d, "total": tot_d, "flags": flags, "notes": [], "bands": [rnd(x) for x in side_bands(r.get("side_sd"))],
        })
        rows[-1]["notes"] = notes(h, a, teams, g, r)
        try:   # v3 game model (scripts/nfl_games_v3.py, games_live.py)
            import games_live
            dome = 1 if g.get("roof") in ("dome", "closed") else 0
            hr_, ar_ = M.fnum(g["home_rest"]) or 7, M.fnum(g["away_rest"]) or 7
            v3f = {k_: r.get(k_) for k_ in ("epa_cal", "sr_cal", "pts_cal", "last_cal", "elo_cal", "mkt_cal", "qb_adj", "blend", "epa_tot_cal", "pts_tot_cal")}
            v3f.update({"home_rest": hr_, "away_rest": ar_, "rest_diff": hr_ - ar_, "div_game": int(g.get("div_game") == "1"), "dome": dome,
                        "temp": 70 if dome else (M.fnum(g.get("temp")) or num(bg.get("temp_f")) or 60),
                        "wind": 0 if dome else (M.fnum(g.get("wind")) or wind_mph(bg.get("weather") or "") or 5), "week": week})
            games_live.attach("nfl", rows[-1], v3f)
            rows[-1]["v3"]["inputs"] = {k_: rnd(v_, 2) for k_, v_ in v3f.items() if v_ is not None}
        except Exception as e:
            print("v3 attach failed:", e)
    order = {"TARGET": 0, "WATCH": 1, "CHECK": 2, "PASS": 3, "FLAG": 4}
    rows.sort(key=lambda x: (min(order[x["side"]["grade"]], order[x["total"]["grade"]]), x["kickoff_utc"]))
    calib = next((r["calib"] for r in ROWS if r["season"] == season and r["week"] == week and r.get("calib")), None)
    return rows, teams, unmatched, calib

# ---------------------------------------------------------------- flag log + CLV
def load_flags():
    return list(csv.DictReader(open(FLAGS))) if os.path.exists(FLAGS) else []

def log_flags(rows, season, week, flags):
    have = {(f["game_id"], f["market"]): f for f in flags}
    rank = {"TARGET": 3, "WATCH": 2, "CHECK": 1}
    for r in rows:
        for mk in ("side", "total"):
            m = r[mk]
            if m["grade"] not in rank:
                continue
            f = have.get((r["id"], mk))
            if f is None:
                f = {"game_id": r["id"], "season": season, "week": week, "kickoff_utc": r["kickoff_utc"],
                     "game": f"{r['away_abbr']} @ {r['home_abbr']}", "market": mk, "pick_side": m["pick_side"], "pick": m["pick"],
                     "first_grade": m["grade"], "best_grade": m["grade"], "flagged_utc": now_iso(), "line_at_flag": m["line"],
                     "odds_at_flag": m.get("odds") or "", "our_number": r["model"]["number"] if mk == "side" else
                     (r["model"]["epa_total"] + r["model"]["pts_total"]) / 2, "status": "open"}
                flags.append(f); have[(r["id"], mk)] = f
            elif f["pick_side"] == m["pick_side"] and rank[m["grade"]] > rank.get(f["best_grade"], 0):
                f["best_grade"] = m["grade"]

def settle_flags(flags):
    """Close = last DK line pulled before kickoff (history_nfl.csv); result from the final score."""
    hist = defaultdict(list)
    if os.path.exists("data/odds/history_nfl.csv"):
        for h in csv.DictReader(open("data/odds/history_nfl.csv")):
            hist[h["game_id"]].append(h)
    latest = {g["game_id"]: g for g in json.load(open("data/odds/latest_nfl.json"))["games"]} if os.path.exists("data/odds/latest_nfl.json") else {}
    for f in flags:
        if f.get("status") == "final":
            continue
        H = hist.get(f["game_id"], [])
        pre = [h for h in H if h["status"] == "pre" and h["pulled_utc"] <= f["kickoff_utc"]] or [h for h in H if h["status"] == "pre"]
        post = [h for h in H if h["status"] == "post"]
        g = latest.get(f["game_id"])
        started = (g and g.get("status") != "pre") or post or (not g and dt.datetime.now(dt.timezone.utc).isoformat() > f["kickoff_utc"])
        if not started or not pre:
            continue
        c = pre[-1]; line = float(f["line_at_flag"])
        if f["market"] == "side":
            close = num(c["home_spread"]); close = close if f["pick_side"] == "home" else (-close if close is not None else None)
            clv = None if close is None else line - close     # + = we got more points than the close
        else:
            close = num(c["total"]); clv = None if close is None else (close - line if f["pick_side"] == "over" else line - close)
        f["close_line"], f["clv"] = close, rnd(clv)
        fin = post[-1] if post else (g if g and g.get("status") == "post" else None)
        if fin:
            hs, as_ = num(fin["home_score"]), num(fin["away_score"])
            f["home_score"], f["away_score"] = hs, as_
            if f["market"] == "side":
                m = (hs - as_) if f["pick_side"] == "home" else (as_ - hs)
                v = m + line
            else:
                v = (hs + as_ - line) * (1 if f["pick_side"] == "over" else -1)
            f["result"] = "W" if v > 0 else "L" if v < 0 else "P"
            f["status"] = "final"
        else:
            f["status"] = "closed"

def flag_record(flags):
    out = {}
    for grade in ("TARGET", "WATCH", "CHECK"):
        for mk in ("side", "total"):
            fs = [f for f in flags if f["best_grade"] == grade and f["market"] == mk and f.get("clv") not in ("", None)]
            clv = [float(f["clv"]) for f in fs]
            res = [f["result"] for f in fs if f.get("result")]
            out[f"{grade}_{mk}"] = {"n": len(fs), "avg_clv": rnd(sum(clv) / len(clv), 2) if clv else None,
                                    "beat_close": sum(1 for x in clv if x > 0), "lost_close": sum(1 for x in clv if x < 0),
                                    "w": res.count("W"), "l": res.count("L"), "p": res.count("P")}
    return out

# ---------------------------------------------------------------- write
def write(season, week, rows, teams, unmatched, calib, odds_pulled):
    key = f"{season}_wk{week:02d}"
    doc = {"key": key, "season": season, "week": week, "league": "nfl", "odds_pulled_utc": odds_pulled,
           "prior_weight": rnd(M.prior_weight(week), 2),
           "stats_pulled_utc": STATUS.get("pulled_utc"), "generated_utc": now_iso(), "calibration": calib,
           "unmatched": unmatched, "games": rows, "teams": teams}
    rows, teams = merge_started(f"data/boards/nfl_{key}.json", rows, teams)  # keep games that already kicked off
    doc["games"], doc["teams"] = rows, teams
    with open(f"data/boards/nfl_{key}.json", "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    with open(f"data/boards/nfl_{key}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kick_et", "game", "dk_home_spread", "dk_open", "bw_number", "epa", "success_rate", "pts", "angle_last_season",
                    "angle_elo", "angle_market", "side_grade", "side_pick",
                    "dk_total", "total_open", "bw_total", "epa_total", "pts_total", "total_grade", "total_pick", "flags", "adjustments", "notes"])
        for r in rows:
            w.writerow([r["kick_et"], f"{r['away_abbr']} @ {r['home_abbr']}", r["dk"]["home_spread"], r["dk"]["home_spread_open"],
                        r["model"]["number"], r["model"]["epa_spread"], r["model"]["sr_spread"], r["model"]["pts_spread"],
                        r["model"]["last_spread"], r["model"]["elo_spread"], r["model"]["mkt_spread"],
                        r["side"]["grade"], r["side"]["pick"] or "", r["dk"]["total"], r["dk"]["total_open"], r["model"]["total_number"], r["model"]["epa_total"],
                        r["model"]["pts_total"], r["total"]["grade"], r["total"]["pick"] or "", "; ".join(r["flags"]),
                        "; ".join(r["side"]["adjust"] + r["total"]["adjust"]), "; ".join(r["notes"])])

def main():
    os.makedirs("data/boards", exist_ok=True)
    board = json.load(open("data/odds/latest_nfl.json"))
    by_week = defaultdict(list)
    for bg in board["games"]:
        g = SCHED_BY_ESPN.get(bg["game_id"])
        if g and bg.get("status") == "pre":
            by_week[(g["season"], g["week"])].append(bg)
    flags = load_flags()
    index = {"generated_utc": now_iso(), "odds_pulled_utc": board["pulled_utc"], "stats_pulled_utc": STATUS.get("pulled_utc"),
             "slates": []}
    for (s, w) in sorted(by_week):
        rows, teams, unmatched, calib = build(by_week[(s, w)], s, w)
        if not rows:
            continue
        log_flags(rows, s, w, flags)          # old tier names still drive the flags.csv CLV log
        finalize_rows(rows, "nfl")            # every game gets a pick; PLAY / LEAN tags, the rest untagged; backtest confidence
        write(s, w, rows, teams, unmatched, calib, board["pulled_utc"])
        cnt = lambda k, gr: sum(1 for r in rows if r[k]["grade"] == gr)
        index["slates"].append({"key": f"{s}_wk{w:02d}", "season": s, "week": w, "label": f"Week {w}",
                                # first day of the week counts games that already kicked off (kept on the board)
                                "first_day": min(r.get("day_et") or "9999" for r in json.load(open(f"data/boards/nfl_{s}_wk{w:02d}.json"))["games"]),
                                "games": len(json.load(open(f"data/boards/nfl_{s}_wk{w:02d}.json"))["games"]),
                                "plays": cnt("side", "PLAY") + cnt("total", "PLAY"),
                                "leans": cnt("side", "LEAN") + cnt("total", "LEAN")})
        print(f"{s} wk{w}: {len(rows)} games, {index['slates'][-1]['plays']} plays, {index['slates'][-1]['leans']} leans; unmatched {unmatched or 'none'}")
    settle_flags(flags)
    with open(FLAGS, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FLAG_COLS, extrasaction="ignore"); wr.writeheader(); wr.writerows(flags)
    index["flag_record"] = flag_record(flags)
    if BACKTEST:
        index["backtest"] = {k: BACKTEST[k] for k in ("seasons", "games", "line", "rmse", "tiers_vs_close", "side_by_gap") if k in BACKTEST}
    if os.path.exists("data/boards/nfl_index.json"):  # keep past weeks listed
        old = json.load(open("data/boards/nfl_index.json"))
        have = {x["key"] for x in index["slates"]}
        for x in old.get("slates", []):
            f = f"data/boards/nfl_{x['key']}.json"
            if x["key"] not in have and os.path.exists(f):
                x["games"] = len(json.load(open(f)).get("games", []))   # includes games that already kicked off
                index["slates"].append(x)
    index["slates"].sort(key=lambda x: x["key"])
    with open("data/boards/nfl_index.json", "w") as f:
        json.dump(index, f, indent=1)

if __name__ == "__main__":
    main()
