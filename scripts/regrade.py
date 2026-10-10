#!/usr/bin/env python3
"""Grade a CFB slate vs DraftKings using SP+, Elo and EPA/success-rate matchups.

Usage: python scripts/regrade.py 2026-10-10        (date in US Eastern)
Writes data/boards/cfb_<date>.csv and prints the graded board.

Projections (home perspective, negative spread = home favored, like DK):
  SP+   margin = SP_home - SP_away + HFA ; total = (offH + defA - natD) + (offA + defH - natD)
  Elo   margin = (Elo_home - Elo_away)/25 + HFA   (latest postgame Elo from CFBD games)
Calibration: SP+ and Elo run more extreme than the market and SP+ totals run high, so each
model is first fitted to the slate's DK lines (linear fit, big-spread/option games excluded).
Gaps are measured against the calibrated projection = where DK breaks from its usual pattern.
Grades (CFB edge threshold 3 pts, from the build plan):
  PLAY  both SP+ and Elo beat the DK number by >= 3 on the same side
  LEAN  one model >= 3 and the other >= 1 on the same side
  PASS  otherwise (or models point opposite ways)
  Totals: SP+ only; LEAN at >= 3, PLAY at >= 5. Option offenses and spreads >= 25 are flagged, not graded.
"""
import csv, datetime as dt, json, os, sys, unicodedata

HFA, EDGE, ELO_PER_PT = 2.5, 3.0, 25.0
OPTION = {"Army", "Navy", "Air Force"}

def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower().replace("'", "").replace(".", "").strip()

def load(name):
    return json.load(open(f"data/cfbd/{name}.json"))["rows"]

sp_rows = load("sp_ratings")
nat = next(r for r in sp_rows if r["team"] == "nationalAverages")
SP = {r["team"]: r for r in sp_rows if r["team"] != "nationalAverages" and r.get("rating") is not None}
ADV = {r["team"]: r for r in load("advanced_season")}
PPA = {r["team"]: r for r in load("ppa_teams")}
TAL = {r["team"]: r["talent"] for r in load("talent")}

ELO = {}
for g in sorted(load("games"), key=lambda g: g["startDate"]):
    if g.get("completed"):
        for side in ("home", "away"):
            if g.get(f"{side}PostgameElo") is not None:
                ELO[g[f"{side}Team"]] = g[f"{side}PostgameElo"]

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

def side(gap):
    return "home" if gap > 0 else "away"

def grade(gs, ge):
    if abs(gs) >= EDGE and abs(ge) >= EDGE and side(gs) == side(ge):
        return "PLAY"
    big, small = (gs, ge) if abs(gs) >= abs(ge) else (ge, gs)
    if abs(big) >= EDGE and abs(small) >= 1 and side(big) == side(small):
        return "LEAN"
    return "PASS"

def f(x, n=2):
    return None if x is None else round(x, n)

def notes(h, a):
    """Matchup flags in the style of the Action Network / CBS write-ups.
    Defensive EPA/PPO are what offenses gain against that defense (higher = worse defense),
    so a matchup's expected value is offense + defense-allowed."""
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
            out.append(f"{de} havoc {hv:.0%}")
        o, d = ao.get("pointsPerOpportunity"), ad.get("pointsPerOpportunity")
        if o is not None and d is not None and o + d >= 8.8:
            out.append(f"{off} should finish drives ({o:.1f} pts/opp; {de} allows {d:.1f})")
    return "; ".join(out)

def _fit(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    return b, my - b * mx

def calibrate(rows):
    base = [r for r in rows if not r["flags"]]
    dk = [r["dk_home_spread"] for r in base]
    bs, a_s = _fit([-r["sp_home_margin"] for r in base], dk)
    eb = [r for r in base if r["elo_home_margin"] is not None]
    be, a_e = _fit([-r["elo_home_margin"] for r in eb], [r["dk_home_spread"] for r in eb])
    tbias = sum(r["total_gap"] for r in base) / len(base)
    print(f"calibration: DK = {bs:.2f}*SP {a_s:+.2f} | DK = {be:.2f}*Elo {a_e:+.2f} | SP total bias {tbias:+.1f}")
    for r in rows:
        dk = r["dk_home_spread"]
        r["sp_cal_spread"] = f(bs * -r["sp_home_margin"] + a_s, 1)
        r["elo_cal_spread"] = f(be * -r["elo_home_margin"] + a_e, 1) if r["elo_home_margin"] is not None else None
        r["gap_sp"] = f(dk - r["sp_cal_spread"], 1)
        r["gap_elo"] = f(dk - r["elo_cal_spread"], 1) if r["elo_cal_spread"] is not None else 0.0
        r["sp_total_cal"] = f(r["sp_total"] - tbias, 1)
        r["total_gap"] = f(r["sp_total_cal"] - r["dk_total"], 1)
        tg = r["total_gap"]
        if r["side_grade"] != "FLAG":
            r["side_grade"] = grade(r["gap_sp"], r["gap_elo"])
            team = r["game"].split(" @ ")[1] if r["gap_sp"] > 0 else r["game"].split(" @ ")[0]
            line = dk if r["gap_sp"] > 0 else -dk
            r["side_pick"] = f"{team} {line:+g}" if r["side_grade"] in ("PLAY", "LEAN") else ""
        if r["total_grade"] != "FLAG":
            r["total_grade"] = "PLAY" if abs(tg) >= 5 else "LEAN" if abs(tg) >= 3 else "PASS"
            r["total_pick"] = (("Over " if tg > 0 else "Under ") + str(r["dk_total"])) if r["total_grade"] in ("PLAY", "LEAN") else ""

def main(day):
    board = json.load(open("data/odds/latest_cfb.json"))
    start = dt.datetime.fromisoformat(day + "T08:00:00+00:00")  # ~4am ET
    end = start + dt.timedelta(hours=24)
    rows, unmatched = [], []
    for g in board["games"]:
        k = dt.datetime.fromisoformat(g["kickoff_utc"].replace("Z", "+00:00"))
        if not (start <= k < end) or g.get("total") is None or g.get("home_spread") is None:
            continue
        h, a = cfbd_name(g["home_name"]), cfbd_name(g["away_name"])
        if not h or not a or h not in SP or a not in SP:
            unmatched.append(f"{g['away_name']} @ {g['home_name']}")
            continue
        hfa = 0 if g.get("neutral") else HFA
        dk = float(g["home_spread"])
        sp_m = SP[h]["rating"] - SP[a]["rating"] + hfa
        elo_m = (ELO[h] - ELO[a]) / ELO_PER_PT + hfa if h in ELO and a in ELO else None
        nd = nat["defense"]["rating"]
        sp_tot = (SP[h]["offense"]["rating"] + SP[a]["defense"]["rating"] - nd) + \
                 (SP[a]["offense"]["rating"] + SP[h]["defense"]["rating"] - nd)
        gap_sp = dk + sp_m            # >0 means home better than DK says
        gap_elo = dk + elo_m if elo_m is not None else 0.0
        tgap = sp_tot - float(g["total"])
        flags = []
        if abs(dk) >= 25: flags.append("big spread")
        if h in OPTION or a in OPTION: flags.append("option team")
        sgrade = "FLAG" if flags else grade(gap_sp, gap_elo)
        tgrade = "FLAG" if (h in OPTION or a in OPTION) else ("PLAY" if abs(tgap) >= 5 else "LEAN" if abs(tgap) >= 3 else "PASS")
        pick_team = (g["home"] if side(gap_sp) == "home" else g["away"])
        pick_line = dk if pick_team == g["home"] else -dk
        rows.append({
            "kick_et": (k - dt.timedelta(hours=4)).strftime("%a %I:%M%p"),
            "game": f"{g['away']} @ {g['home']}", "away": a, "home": h, "neutral": bool(g.get("neutral")),
            "dk_home_spread": dk, "dk_home_open": g.get("home_spread_open"), "dk_total": g["total"], "dk_total_open": g.get("total_open"),
            "sp_home_margin": f(sp_m, 1), "elo_home_margin": f(elo_m, 1), "sp_total": f(sp_tot, 1),
            "gap_sp": f(gap_sp, 1), "gap_elo": f(gap_elo, 1), "total_gap": f(tgap, 1),
            "side_grade": sgrade, "side_pick": f"{pick_team} {pick_line:+g}" if sgrade in ("PLAY", "LEAN") else "",
            "total_grade": tgrade, "total_pick": (("Over " if tgap > 0 else "Under ") + str(g["total"])) if tgrade in ("PLAY", "LEAN") else "",
            "sp_rank_away": SP[a].get("ranking"), "sp_rank_home": SP[h].get("ranking"),
            "talent_away": f(TAL.get(a), 0), "talent_home": f(TAL.get(h), 0),
            "weather": g.get("weather"), "temp_f": g.get("temp_f"), "flags": ", ".join(flags),
            "matchup_notes": notes(h, a),
        })
    calibrate(rows)
    os.makedirs("data/boards", exist_ok=True)
    out = f"data/boards/cfb_{day}.csv"
    if rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    order = {"PLAY": 0, "LEAN": 1, "PASS": 2, "FLAG": 3}
    rows.sort(key=lambda r: (min(order[r["side_grade"]], order[r["total_grade"]]), -max(abs(r["gap_sp"]), abs(r["total_gap"]))))
    print(f"{len(rows)} games graded; DK lines pulled {board['pulled_utc']}; unmatched: {unmatched or 'none'}")
    for r in rows:
        print(f"{r['kick_et']:>10} {r['game']:16} DK {r['dk_home_spread']:+5.1f} SPc {r['sp_cal_spread']:+6.1f} "
              f"Eloc {(r['elo_cal_spread'] if r['elo_cal_spread'] is not None else float('nan')):+6.1f} "
              f"| {r['side_grade']:4} {r['side_pick']:11} | T {r['dk_total']:5} SPc {r['sp_total_cal']:5} {r['total_grade']:4} {r['total_pick']:12} {r['flags']}")
    return out

if __name__ == "__main__":
    print("wrote", main(sys.argv[1] if len(sys.argv) > 1 else dt.date.today().isoformat()))
