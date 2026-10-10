#!/usr/bin/env python3
"""Grade CFB slates vs DraftKings using SP+, Elo, a per-play screen and EPA matchups.

Usage:
  python scripts/regrade.py                 grade every slate date on the DK board (next ~7 days)
  python scripts/regrade.py 2026-10-10      grade one date (US Eastern)

Writes data/boards/cfb_<date>.json + .csv and data/boards/index.json (read by the web app).

Method (see the project's handicapping checklist):
  Sides   SP+ margin = SP_home - SP_away + HFA;  Elo margin = dElo/25 + HFA.
          Each model is fitted to the slate's DK spreads first (they run more extreme than the
          market), so an edge = where DK breaks from its usual relationship to the model.
          PLAY = both >= 3 pts same side; LEAN = one >= 3 and other >= 1 same side; else PASS.
  Totals  SP+ total and a per-play total (yds/play off vs def x pace, from CFBD season stats),
          each de-biased to the slate's DK totals. PLAY = both >= 3 same way and one >= 5;
          LEAN = both >= 3 same way; else PASS.
  Adjust  Market moved >= 2 pts toward the other side since open: PLAY->LEAN, LEAN->PASS.
          Over picks where the total fell >= 5 and rain/showers are forecast: -> PASS (models don't see weather).
  Flags   Spreads >= 25 and option offenses (Army/Navy/Air Force) are shown but not graded.
"""
import csv, datetime as dt, json, os, sys, unicodedata

HFA, EDGE, ELO_PER_PT, MOVE_AGAINST = 2.5, 3.0, 25.0, 2.0
OPTION = {"Army", "Navy", "Air Force"}
WET = ("rain", "shower", "storm", "thunder", "snow")
ET = dt.timezone(dt.timedelta(hours=-4))  # EDT; good through early November

def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower().replace("'", "").replace(".", "").strip()

def load(name):
    d = json.load(open(f"data/cfbd/{name}.json"))
    return d["rows"], d.get("pulled_utc")

sp_rows, CFBD_PULLED = load("sp_ratings")
nat = next(r for r in sp_rows if r["team"] == "nationalAverages")
SP = {r["team"]: r for r in sp_rows if r["team"] != "nationalAverages" and r.get("rating") is not None}
ADV = {r["team"]: r for r in load("advanced_season")[0]}
PPA = {r["team"]: r for r in load("ppa_teams")[0]}
TAL = {r["team"]: r["talent"] for r in load("talent")[0]}

STATS = {}
for r in load("season_stats")[0]:
    STATS.setdefault(r["team"], {})[r["statName"]] = r["statValue"]

def per_play(t):
    s = STATS.get(t, {})
    try:
        plays = s["rushingAttempts"] + s["passAttempts"]
        oplays = s["rushingAttemptsOpponent"] + s["passAttemptsOpponent"]
        return {"ypp": s["totalYards"] / plays, "ypp_allowed": s["totalYardsOpponent"] / oplays,
                "pace": plays / s["games"]}
    except (KeyError, ZeroDivisionError):
        return None

PP = {t: v for t in STATS if (v := per_play(t))}
LG_YPP = sum(STATS[t]["totalYards"] for t in PP) / sum(STATS[t]["rushingAttempts"] + STATS[t]["passAttempts"] for t in PP)

ELO, RESULTS = {}, {}
for g in sorted(load("games")[0], key=lambda g: g["startDate"]):
    if g.get("completed") and g.get("homePoints") is not None:
        for s, o in (("home", "away"), ("away", "home")):
            if g.get(f"{s}PostgameElo") is not None:
                ELO[g[f"{s}Team"]] = g[f"{s}PostgameElo"]
            pf, pa = g[f"{s}Points"], g[f"{o}Points"]
            RESULTS.setdefault(g[f"{s}Team"], []).append({
                "wk": g["week"], "opp": g[f"{o}Team"], "site": "N" if g.get("neutralSite") else ("H" if s == "home" else "A"),
                "res": "W" if pf > pa else "L" if pf < pa else "T", "score": f"{pf}-{pa}"})

TEAMS = sorted(SP, key=lambda t: -len(t))
ALIASES = {"miami hurricanes": "Miami", "miami (oh) redhawks": "Miami (OH)"}

def cfbd_name(display):
    d = norm(display)
    if d in ALIASES:
        return ALIASES[d]
    for t in TEAMS:  # longest CFBD name that prefixes the ESPN display name
        n = norm(t)
        if d == n or d.startswith(n + " "):
            return t
    return None

def num(x):
    try:
        return float(str(x).lstrip("ou"))
    except (TypeError, ValueError):
        return None

def rnd(x, n=1):
    return None if x is None else round(x, n)

def side_of(gap):
    return "home" if gap > 0 else "away"

def grade_side(gs, ge):
    if ge is None:
        ge = 0.0
    if abs(gs) >= EDGE and abs(ge) >= EDGE and side_of(gs) == side_of(ge):
        return "PLAY"
    big, small = (gs, ge) if abs(gs) >= abs(ge) else (ge, gs)
    if abs(big) >= EDGE and abs(small) >= 1 and side_of(big) == side_of(small):
        return "LEAN"
    return "PASS"

def grade_total(g1, g2):
    if g2 is None or g1 * g2 <= 0 or min(abs(g1), abs(g2)) < 3:
        return "PASS"
    return "PLAY" if max(abs(g1), abs(g2)) >= 5 else "LEAN"

DOWN = {"PLAY": "LEAN", "LEAN": "PASS", "PASS": "PASS"}

def team_card(t):
    p, a = PPA.get(t, {}), ADV.get(t, {})
    po, pd, ao, ad = p.get("offense", {}), p.get("defense", {}), a.get("offense", {}), a.get("defense", {})
    pp = PP.get(t, {})
    s = STATS.get(t, {})
    return {
        "name": t, "conf": SP[t].get("conference"), "sp": rnd(SP[t]["rating"]), "sp_rank": SP[t].get("ranking"),
        "sp_off": rnd(SP[t]["offense"]["rating"]), "sp_off_rank": SP[t]["offense"].get("ranking"),
        "sp_def": rnd(SP[t]["defense"]["rating"]), "sp_def_rank": SP[t]["defense"].get("ranking"),
        "elo": rnd(ELO.get(t), 0), "talent": rnd(TAL.get(t), 0),
        "off": {"epa": rnd(po.get("overall"), 2), "rush_epa": rnd(po.get("rushing"), 2), "pass_epa": rnd(po.get("passing"), 2),
                "sr": rnd(ao.get("successRate"), 3), "expl": rnd(ao.get("explosiveness"), 2),
                "ppo": rnd(ao.get("pointsPerOpportunity"), 2), "stuff": rnd(ao.get("stuffRate"), 3),
                "havoc": rnd((ao.get("havoc") or {}).get("total"), 3), "ypp": rnd(pp.get("ypp"), 2), "pace": rnd(pp.get("pace"), 1)},
        "def": {"epa": rnd(pd.get("overall"), 2), "rush_epa": rnd(pd.get("rushing"), 2), "pass_epa": rnd(pd.get("passing"), 2),
                "sr": rnd(ad.get("successRate"), 3), "expl": rnd(ad.get("explosiveness"), 2),
                "ppo": rnd(ad.get("pointsPerOpportunity"), 2), "stuff": rnd(ad.get("stuffRate"), 3),
                "havoc": rnd((ad.get("havoc") or {}).get("total"), 3), "ypp": rnd(pp.get("ypp_allowed"), 2)},
        "turnovers": s.get("turnovers"), "takeaways": s.get("turnoversOpponent"), "games": s.get("games"),
        "third_down": rnd(s["thirdDownConversions"] / s["thirdDowns"], 3) if s.get("thirdDowns") else None,
        "third_down_allowed": rnd(s["thirdDownConversionsOpponent"] / s["thirdDownsOpponent"], 3) if s.get("thirdDownsOpponent") else None,
        "results": RESULTS.get(t, []),
    }

def notes(h, a):
    """Matchup flags. Defensive EPA/PPO = what offenses gain vs that D (higher = worse D)."""
    out = []
    for off, de in ((h, a), (a, h)):
        po, pd = PPA.get(off, {}).get("offense", {}), PPA.get(de, {}).get("defense", {})
        ao, ad = ADV.get(off, {}).get("offense", {}), ADV.get(de, {}).get("defense", {})
        for kind in ("rushing", "passing"):
            o, d = po.get(kind), pd.get(kind)
            if o is None or d is None:
                continue
            label = "run" if kind == "rushing" else "pass"
            if o + d >= 0.45:
                out.append(f"{off} {label} edge ({label} EPA {o:+.2f}; {de} allows {d:+.2f})")
            elif o + d <= -0.20:
                out.append(f"{de} {label} D shuts down {off} ({label} EPA {o:+.2f}; allows {d:+.2f})")
        hv = (ad.get("havoc") or {}).get("total")
        if hv is not None and hv >= 0.20:
            out.append(f"{de} havoc rate {hv:.0%}")
        o, d = ao.get("pointsPerOpportunity"), ad.get("pointsPerOpportunity")
        if o is not None and d is not None and o + d >= 8.8:
            out.append(f"{off} should finish drives ({o:.1f} pts/opp; {de} allows {d:.1f})")
    return out

def fit(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    return b, my - b * mx

def build_rows(games):
    rows, unmatched = [], []
    nd = nat["defense"]["rating"]
    for g in games:
        dk, tot = num(g.get("home_spread")), num(g.get("total"))
        if dk is None or tot is None:
            continue
        h, a = cfbd_name(g["home_name"]), cfbd_name(g["away_name"])
        if not h or not a or h not in SP or a not in SP:
            unmatched.append(f"{g['away_name']} @ {g['home_name']}")
            continue
        hfa = 0 if g.get("neutral") else HFA
        sp_m = SP[h]["rating"] - SP[a]["rating"] + hfa
        elo_m = (ELO[h] - ELO[a]) / ELO_PER_PT + hfa if h in ELO and a in ELO else None
        sp_tot = (SP[h]["offense"]["rating"] + SP[a]["defense"]["rating"] - nd) + (SP[a]["offense"]["rating"] + SP[h]["defense"]["rating"] - nd)
        pp_tot = None
        if h in PP and a in PP:
            eh = PP[h]["ypp"] + PP[a]["ypp_allowed"] - LG_YPP
            ea = PP[a]["ypp"] + PP[h]["ypp_allowed"] - LG_YPP
            pp_tot = (eh + ea) * (PP[h]["pace"] + PP[a]["pace"]) / 2 / 14.5
        flags = []
        if abs(dk) >= 25: flags.append("Big spread")
        if h in OPTION or a in OPTION: flags.append("Option offense")
        k = dt.datetime.fromisoformat(g["kickoff_utc"].replace("Z", "+00:00"))
        rows.append({
            "id": g["game_id"], "kickoff_utc": g["kickoff_utc"], "kick_et": k.astimezone(ET).strftime("%-I:%M %p"),
            "away_abbr": g["away"], "home_abbr": g["home"], "away": a, "home": h, "neutral": bool(g.get("neutral")),
            "conference_game": g.get("conference_game"), "venue": g.get("venue"), "weather": g.get("weather"), "temp_f": g.get("temp_f"),
            "dk": {"home_spread": dk, "home_spread_open": num(g.get("home_spread_open")),
                   "home_spread_odds": g.get("home_spread_odds"), "away_spread_odds": g.get("away_spread_odds"),
                   "total": tot, "total_open": num(g.get("total_open")), "over_odds": g.get("over_odds"), "under_odds": g.get("under_odds"),
                   "home_ml": g.get("home_ml"), "away_ml": g.get("away_ml")},
            "model": {"sp_margin": sp_m, "elo_margin": elo_m, "sp_total": sp_tot, "pp_total": pp_tot},
            "flags": flags, "notes": notes(h, a),
        })
    return rows, unmatched

CALIB_FILE, MIN_FIT = "data/boards/calibration.json", 15
DEFAULT_CALIB = {"spread_vs_sp": [0.73, -0.8], "spread_vs_elo": [0.8, -1.48], "sp_total_bias": 2.5, "pp_total_bias": -1.8}

def grade(rows):
    """Fit models to this slate's DK lines when it's big enough; small (midweek) slates
    reuse the fit from the most recent full slate so a 2-game Tuesday isn't graded on noise."""
    base = [r for r in rows if not r["flags"]]
    if len(base) >= MIN_FIT:
        bs, a_s = fit([-r["model"]["sp_margin"] for r in base], [r["dk"]["home_spread"] for r in base])
        eb = [r for r in base if r["model"]["elo_margin"] is not None]
        be, a_e = fit([-r["model"]["elo_margin"] for r in eb], [r["dk"]["home_spread"] for r in eb])
        sp_bias = sum(r["model"]["sp_total"] - r["dk"]["total"] for r in base) / len(base)
        pb = [r for r in base if r["model"]["pp_total"] is not None]
        pp_bias = sum(r["model"]["pp_total"] - r["dk"]["total"] for r in pb) / len(pb)
        calib = {"spread_vs_sp": [round(bs, 2), round(a_s, 2)], "spread_vs_elo": [round(be, 2), round(a_e, 2)],
                 "sp_total_bias": round(sp_bias, 1), "pp_total_bias": round(pp_bias, 1), "n": len(base), "source": "this slate"}
        with open(CALIB_FILE, "w") as f:
            json.dump({**calib, "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")}, f, indent=1)
    else:
        saved = json.load(open(CALIB_FILE)) if os.path.exists(CALIB_FILE) else dict(DEFAULT_CALIB, saved_utc=None)
        calib = {k: saved[k] for k in DEFAULT_CALIB}
        calib.update({"n": len(base), "source": f"last full slate ({saved.get('saved_utc') or 'default'})"})
    bs, a_s = calib["spread_vs_sp"]; be, a_e = calib["spread_vs_elo"]
    sp_bias, pp_bias = calib["sp_total_bias"], calib["pp_total_bias"]
    for r in rows:
        m, d = r["model"], r["dk"]
        sp_c = bs * -m["sp_margin"] + a_s
        elo_c = be * -m["elo_margin"] + a_e if m["elo_margin"] is not None else None
        gs = d["home_spread"] - sp_c
        ge = d["home_spread"] - elo_c if elo_c is not None else None
        sp_t = m["sp_total"] - sp_bias
        pp_t = m["pp_total"] - pp_bias if m["pp_total"] is not None else None
        g1, g2 = sp_t - d["total"], (pp_t - d["total"]) if pp_t is not None else None
        m.update({"sp_spread_cal": rnd(sp_c), "elo_spread_cal": rnd(elo_c), "sp_total_cal": rnd(sp_t), "pp_total_cal": rnd(pp_t),
                  "sp_margin": rnd(m["sp_margin"]), "elo_margin": rnd(m["elo_margin"]), "sp_total": rnd(m["sp_total"]), "pp_total": rnd(m["pp_total"])})
        side = {"edge_sp": rnd(gs), "edge_elo": rnd(ge), "grade": "FLAG" if r["flags"] else grade_side(gs, ge), "pick": None, "adjust": []}
        if side["grade"] in ("PLAY", "LEAN"):
            home_pick = gs > 0
            team = r["home_abbr"] if home_pick else r["away_abbr"]
            line = d["home_spread"] if home_pick else -d["home_spread"]
            side["pick"] = f"{team} {line:+g}"
            side["pick_side"] = "home" if home_pick else "away"
            if d["home_spread_open"] is not None:
                moved = d["home_spread"] - d["home_spread_open"]  # + = toward away team
                against = moved if home_pick else -moved
                if against >= MOVE_AGAINST:
                    side["adjust"].append(f"Line moved {against:g} pts toward the other side since open")
                    side["grade"] = DOWN[side["grade"]]
                    if side["grade"] == "PASS":
                        side["pick"] = None
        total = {"edge_sp": rnd(g1), "edge_pp": rnd(g2), "grade": "FLAG" if any(f == "Option offense" for f in r["flags"]) else grade_total(g1, g2),
                 "pick": None, "adjust": []}
        if total["grade"] in ("PLAY", "LEAN"):
            over = g1 > 0
            total["pick"] = f"{'Over' if over else 'Under'} {d['total']:g}"
            total["pick_side"] = "over" if over else "under"
            wet = any(w in (r.get("weather") or "").lower() for w in WET)
            if over and wet and d["total_open"] is not None and d["total_open"] - d["total"] >= 5:
                total["adjust"].append(f"Total fell {d['total_open'] - d['total']:g} with {r['weather'].lower()} forecast; models don't see weather")
                total["grade"], total["pick"] = "PASS", None
        r["side"], r["total"] = side, total
    order = {"PLAY": 0, "LEAN": 1, "PASS": 2, "FLAG": 3}
    rows.sort(key=lambda r: (min(order[r["side"]["grade"]], order[r["total"]["grade"]]), r["kickoff_utc"]))
    return calib

def write(day, rows, calib, unmatched, odds_pulled):
    os.makedirs("data/boards", exist_ok=True)
    teams = {t: team_card(t) for r in rows for t in (r["home"], r["away"])}
    doc = {"date": day, "league": "cfb", "odds_pulled_utc": odds_pulled, "cfbd_pulled_utc": CFBD_PULLED,
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
           "calibration": calib, "unmatched": unmatched, "games": rows, "teams": teams}
    with open(f"data/boards/cfb_{day}.json", "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    with open(f"data/boards/cfb_{day}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kick_et", "game", "dk_home_spread", "dk_open", "sp_cal", "elo_cal", "side_grade", "side_pick",
                    "dk_total", "total_open", "sp_total_cal", "pp_total_cal", "total_grade", "total_pick", "flags", "adjustments", "notes"])
        for r in rows:
            w.writerow([r["kick_et"], f"{r['away_abbr']} @ {r['home_abbr']}", r["dk"]["home_spread"], r["dk"]["home_spread_open"],
                        r["model"]["sp_spread_cal"], r["model"]["elo_spread_cal"], r["side"]["grade"], r["side"]["pick"] or "",
                        r["dk"]["total"], r["dk"]["total_open"], r["model"]["sp_total_cal"], r["model"]["pp_total_cal"],
                        r["total"]["grade"], r["total"]["pick"] or "", "; ".join(r["flags"]),
                        "; ".join(r["side"]["adjust"] + r["total"]["adjust"]), "; ".join(r["notes"])])
    return doc

def main(days=None):
    board = json.load(open("data/odds/latest_cfb.json"))
    by_day = {}
    for g in board["games"]:
        k = dt.datetime.fromisoformat(g["kickoff_utc"].replace("Z", "+00:00")).astimezone(ET)
        if g.get("status") == "pre":
            by_day.setdefault(k.date().isoformat(), []).append(g)
    days = days or sorted(by_day, key=lambda d: -sum(1 for g in by_day[d] if g.get("total") is not None))
    index = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
             "odds_pulled_utc": board["pulled_utc"], "cfbd_pulled_utc": CFBD_PULLED, "slates": []}
    for day in days:
        rows, unmatched = build_rows(by_day.get(day, []))
        if len(rows) < 2:
            continue
        calib = grade(rows)
        write(day, rows, calib, unmatched, board["pulled_utc"])
        count = lambda k, gr: sum(1 for r in rows if r[k]["grade"] == gr)
        index["slates"].append({"date": day, "games": len(rows),
                                "plays": count("side", "PLAY") + count("total", "PLAY"),
                                "leans": count("side", "LEAN") + count("total", "LEAN")})
        print(f"{day}: {len(rows)} games, {index['slates'][-1]['plays']} plays, {index['slates'][-1]['leans']} leans; unmatched {unmatched or 'none'}")
    # keep past slates listed so the app can still open them
    if os.path.exists("data/boards/index.json"):
        old = json.load(open("data/boards/index.json"))
        have = {s["date"] for s in index["slates"]}
        index["slates"] += [s for s in old.get("slates", []) if s["date"] not in have and os.path.exists(f"data/boards/cfb_{s['date']}.json")]
    index["slates"].sort(key=lambda s: s["date"])
    with open("data/boards/index.json", "w") as f:
        json.dump(index, f, indent=1)

if __name__ == "__main__":
    main(sys.argv[1:] or None)
