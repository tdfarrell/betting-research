#!/usr/bin/env python3
"""Grade the Screener's own predictions once games are final, day by day.

Each run (after the graders):
 1. Snapshot  every game still before kickoff on a CFB / NFL board -> data/grades/screener_picks.csv (one row per
              game x market). The row keeps updating until kickoff, so what gets graded is the pick as it stood
              at kickoff. The DraftKings line when a game was FIRST flagged is kept too, for closing-line value.
              PASS games are kept with the model's lean (which side it liked) so "all games" accuracy can be shown.
 2. Props     every NFL prop card before kickoff -> data/grades/props_picks.csv (model % and season hit rate per tier).
 3. Settle    final scores from data/results/<league>.json (ESPN), else the DK history file; prop stats from the
              ESPN box score, else nflverse.
 4. Summary   data/grades/screener_results.json: season totals and each day, by tier, for the Results view.

  python scripts/grade_screener.py              normal run
  python scripts/grade_screener.py --backfill   rebuild snapshots from the repo's git history (needs history fetched)
"""
import csv, datetime as dt, glob, json, os, re, subprocess, sys, unicodedata
from collections import defaultdict

OUT = "data/grades"
PICKS, PROPS, SUMMARY = f"{OUT}/screener_picks.csv", f"{OUT}/props_picks.csv", f"{OUT}/screener_results.json"
ET = dt.timezone(dt.timedelta(hours=-4))
PICK_COLS = ["league", "slate", "day_et", "game_id", "kickoff_utc", "game", "away", "home", "market", "grade", "pick", "pick_side",
             "line", "odds", "model", "edge", "lean_side", "lean_line", "first_seen_utc", "last_seen_utc",
             "flag_grade", "flag_side", "flag_line", "flag_utc", "away_score", "home_score", "result", "lean_result", "clv", "status"]
PROP_COLS = ["season", "week", "slate", "day_et", "game_id", "kickoff_utc", "player_id", "name", "team", "opp", "role", "market",
             "tiers", "probs", "season_hr", "l3_hr", "disagree", "last_seen_utc", "actual", "status"]
FLAGGED = {"PLAY", "LEAN", "TARGET", "WATCH", "CHECK"}
TIER_ORDER = {"cfb": ["PLAY", "LEAN", "CHECK"], "nfl": ["TARGET", "WATCH", "CHECK"], "nba": ["TARGET", "WATCH", "CHECK"]}

def now_utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes").replace("+00:00", "Z")

def iso_le(a, b):
    """a <= b for ISO times that may end in Z or +00:00."""
    f = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return f(a) <= f(b)

def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def load_csv(path, cols):
    if not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        out[key_of(r, cols)] = r
    return out

def key_of(r, cols):
    return (r["league"], r["game_id"], r["market"]) if cols is PICK_COLS else (r["game_id"], r["player_id"], r["market"])

def save_csv(path, rows, cols):
    os.makedirs(OUT, exist_ok=True)
    rows = sorted(rows.values(), key=lambda r: (r["kickoff_utc"], r["game_id"], r.get("market", ""), r.get("player_id", "")))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

# ------------------------------------------------------------------ 1. line snapshots
def snapshot_board(doc, league, slate, picks, at):
    for g in doc.get("games", []):
        if g.get("started") or not g.get("kickoff_utc") or iso_le(g["kickoff_utc"], at):
            continue
        k = dt.datetime.fromisoformat(g["kickoff_utc"].replace("Z", "+00:00"))
        d, m = g["dk"], g["model"]
        for mk in ("side", "total"):
            s = g.get(mk) or {}
            grade = s.get("grade") or "PASS"
            if mk == "side":
                hs = d.get("home_spread")
                if hs is None:
                    continue
                edge = s.get("edge") if s.get("edge") is not None else s.get("edge_sp")
                lean = None if edge is None or edge == 0 else ("home" if edge > 0 else "away")
                lean_line = None if lean is None else (hs if lean == "home" else -hs)
                model = m.get("number") if league in ("nfl", "nba") else m.get("sp_spread_cal")
            else:
                tot = d.get("total")
                if tot is None:
                    continue
                edge = s.get("edge") if s.get("edge") is not None else s.get("edge_sp")
                lean = None if edge is None or edge == 0 else ("over" if edge > 0 else "under")
                lean_line = tot
                model = m.get("total_number") if league in ("nfl", "nba") else (
                    ((m.get("sp_total_cal") or 0) + (m.get("pp_total_cal") or 0)) / 2 if m.get("pp_total_cal") is not None else m.get("sp_total_cal"))
            key = (league, str(g["id"]), mk)
            r = picks.get(key) or {"league": league, "game_id": str(g["id"]), "market": mk, "first_seen_utc": at, "status": "open"}
            if r.get("status") == "final":
                continue
            side = s.get("pick_side") if grade in FLAGGED else None
            r.update({"slate": slate, "day_et": k.astimezone(ET).date().isoformat(), "kickoff_utc": g["kickoff_utc"],
                      "game": f"{g.get('away_abbr')} @ {g.get('home_abbr')}", "away": g.get("away_abbr"), "home": g.get("home_abbr"),
                      "grade": grade, "pick": s.get("pick") or "", "pick_side": side or "",
                      "line": s.get("line") if side else "", "odds": s.get("odds") or "",
                      "model": round(model, 1) if isinstance(model, (int, float)) else "", "edge": edge if edge is not None else "",
                      "lean_side": lean or "", "lean_line": lean_line if lean_line is not None else "", "last_seen_utc": at})
            if side and grade in FLAGGED:
                ln = s.get("line")
                if ln is None:
                    ln = (d.get("home_spread") if side == "home" else -d.get("home_spread")) if mk == "side" else d.get("total")
                r["line"] = ln
                if r.get("flag_side") != side:          # first flag (or the side flipped): line at that moment
                    r.update({"flag_grade": grade, "flag_side": side, "flag_line": ln, "flag_utc": at})
            picks[key] = r

# ------------------------------------------------------------------ 2. prop snapshots
def snapshot_props(doc, props, at):
    for c in doc.get("cards", []):
        if not c.get("kickoff_utc") or iso_le(c["kickoff_utc"], at):
            continue
        k = dt.datetime.fromisoformat(c["kickoff_utc"].replace("Z", "+00:00"))
        for mk, m in c.get("markets", {}).items():
            key = (str(c["game_id"]), c["id"], mk)
            r = props.get(key) or {"game_id": str(c["game_id"]), "player_id": c["id"], "market": mk, "status": "open"}
            if r.get("status") in ("final", "void"):
                continue
            fr = lambda x: f"{x[0]}/{x[1]}"
            r.update({"season": doc.get("season"), "week": doc.get("week"), "slate": doc.get("key"),
                      "day_et": k.astimezone(ET).date().isoformat(), "kickoff_utc": c["kickoff_utc"],
                      "name": c["name"], "team": c["team"], "opp": c["opp"], "role": c["role"],
                      "tiers": ",".join(str(t["t"]) for t in m["tiers"]), "probs": ",".join(f"{t['p']:.3f}" for t in m["tiers"]),
                      "season_hr": ",".join(fr(t["season"]) for t in m["tiers"]), "l3_hr": ",".join(fr(t["l3"]) for t in m["tiers"]),
                      "disagree": "|".join(m.get("disagree") or []), "last_seen_utc": at})
            props[key] = r

# ------------------------------------------------------------------ 3. settle
def finals(league):
    out = {}
    p = f"data/results/{league}.json"
    if os.path.exists(p):
        for g in json.load(open(p)).get("games", []):
            if g.get("final"):
                out[str(g["id"])] = (g["away"]["score"], g["home"]["score"], g)
    p = f"data/odds/history_{league}.csv"
    if os.path.exists(p):
        for h in csv.DictReader(open(p)):
            if h["status"] == "post" and h["game_id"] not in out and num(h["away_score"]) is not None:
                out[h["game_id"]] = (num(h["away_score"]), num(h["home_score"]), None)
    return out

def wlp(x):
    return "" if x is None else ("W" if x > 0 else "L" if x < 0 else "P")

def settle_picks(picks):
    F = {lg: finals(lg) for lg in ("cfb", "nfl", "nba")}
    for r in picks.values():
        if r.get("status") == "final":
            continue
        f = F.get(r["league"], {}).get(r["game_id"])
        if not f:
            continue
        a, h = f[0], f[1]
        r["away_score"], r["home_score"] = int(a), int(h)
        def res(side, line):
            line = num(line)
            if not side or line is None:
                return None
            if r["market"] == "side":
                margin = (h - a) if side == "home" else (a - h)
                return margin + line
            return (a + h - line) * (1 if side == "over" else -1)
        r["result"] = wlp(res(r.get("pick_side"), r.get("line")))
        r["lean_result"] = wlp(res(r.get("lean_side"), r.get("lean_line")))
        fl, cl = num(r.get("flag_line")), num(r.get("line"))
        if r.get("flag_side") and r.get("flag_side") == r.get("pick_side") and fl is not None and cl is not None:
            r["clv"] = round((fl - cl) if r["market"] == "side" else ((cl - fl) if r["pick_side"] == "over" else (fl - cl)), 1)
        r["status"] = "final"

def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()

PROP_STAT = {"rec_yds": ("rec_yds", "receiving_yards"), "rush_yds": ("rush_yds", "rushing_yards"), "rr_yds": ("rr_yds", None),
             "pass_yds": ("pass_yds", "passing_yards"), "rec": ("rec", "receptions"), "pass_td": ("pass_td", "passing_tds"), "atd": ("td", None)}

def settle_props(props):
    espn = {}
    if os.path.exists("data/results/nfl.json"):
        for g in json.load(open("data/results/nfl.json")).get("games", []):
            if g.get("final") and g.get("players"):
                espn[str(g["id"])] = {norm(n): p for n, p in g["players"].items()}
    nflv, teams_in = {}, set()
    if os.path.exists("data/nfl/player_games.csv"):
        for x in csv.DictReader(open("data/nfl/player_games.csv")):
            nflv[(x["season"], x["week"], x["player_id"])] = x
            teams_in.add((x["season"], x["week"], x["team"]))
    for r in props.values():
        if r.get("status") in ("final", "void"):
            continue
        val = None
        box = espn.get(r["game_id"])
        if box is not None:
            p = box.get(norm(r["name"]))
            ek, _ = PROP_STAT[r["market"]]
            if p is not None and not p.get("dnp"):
                val = p["s"].get(ek, 0)
            # not in the box score (inactive, or no touches): left for nflverse, which lists every player who played
        if val is None:
            x = nflv.get((str(r["season"]), str(r["week"]), r["player_id"]))
            if x:
                ek, nk = PROP_STAT[r["market"]]
                f = lambda c: num(x.get(c)) or 0
                val = {"rr_yds": f("rushing_yards") + f("receiving_yards"), "atd": f("rushing_tds") + f("receiving_tds")}.get(r["market"]) \
                    if nk is None else f(nk)
        if val is not None:
            r["actual"] = int(val) if float(val).is_integer() else val
            r["status"] = "final"
        elif (str(r["season"]), str(r["week"]), r["team"]) in teams_in:
            r["status"] = "void"   # his team's game is in nflverse but he isn't: didn't play (DraftKings voids these)

# ------------------------------------------------------------------ 4. summary
def rec_add(d, res, odds):
    if res == "W":
        d["w"] += 1
        o = num(odds) or -110
        d["units"] += (o / 100) if o > 0 else (100 / -o)
    elif res == "L":
        d["l"] += 1; d["units"] -= 1
    elif res == "P":
        d["p"] += 1

def blank():
    return {"w": 0, "l": 0, "p": 0, "units": 0.0, "clv_sum": 0.0, "clv_n": 0}

def finish(d):
    n = d["w"] + d["l"]
    d["pct"] = round(d["w"] / n, 3) if n else None
    d["units"] = round(d["units"], 2)
    d["clv"] = round(d.pop("clv_sum") / d["clv_n"], 2) if d["clv_n"] else None
    d.pop("clv_n")
    return d

def props_metrics(rows):
    calls = []
    for r in rows:
        if r.get("status") != "final":
            continue
        act = num(r["actual"])
        tiers = [num(t) for t in r["tiers"].split(",")]
        probs = [num(p) for p in r["probs"].split(",")]
        hrs = r["season_hr"].split(",")
        dis = r.get("disagree") or ""
        for t, p, hr in zip(tiers, probs, hrs):
            hit = int(act >= t)
            h, n = (int(x) for x in hr.split("/"))
            lab = f" {int(t)}+:" if r["market"] != "atd" else ":"
            calls.append({"p": p, "hit": hit, "hr": (h / n) if n else None, "dis": lab in dis, "market": r["market"]})
    if not calls:
        return None
    out = {"calls": len(calls), "players": len({(r["game_id"], r["player_id"]) for r in rows if r.get("status") == "final"})}
    out["brier_model"] = round(sum((c["p"] - c["hit"]) ** 2 for c in calls) / len(calls), 4)
    hc = [c for c in calls if c["hr"] is not None]
    out["brier_hit_rate"] = round(sum(((c["hr"] * 0.85 + 0.075) - c["hit"]) ** 2 for c in hc) / len(hc), 4) if hc else None
    bands = []
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        b = [c for c in calls if lo <= c["p"] < hi]
        if b:
            bands.append({"band": f"{int(lo * 100)}-{min(int(hi * 100), 100)}%", "n": len(b), "model": round(sum(c["p"] for c in b) / len(b), 3),
                          "hit": round(sum(c["hit"] for c in b) / len(b), 3)})
    out["calibration"] = bands
    strong = [c for c in calls if c["p"] >= 0.7]
    out["model_70"] = {"n": len(strong), "hit": sum(c["hit"] for c in strong)}
    dis = [c for c in calls if c["dis"] and c["hr"] is not None]
    out["disagreements"] = {"n": len(dis), "model_right": sum(1 for c in dis if (c["p"] >= .5) == bool(c["hit"])),
                            "hit_rate_right": sum(1 for c in dis if (c["hr"] >= .5) == bool(c["hit"]))}
    return out

def summarize(picks, props):
    out = {"generated_utc": now_utc(), "leagues": {}}
    for lg in ("cfb", "nfl", "nba"):
        rows = [r for r in picks.values() if r["league"] == lg]
        if not rows and lg == "nba":
            continue
        days = defaultdict(list)
        for r in rows:
            days[r["day_et"]].append(r)
        season = {"side": defaultdict(blank), "total": defaultdict(blank)}
        day_list = []
        for day in sorted(days):
            rs = days[day]
            fin = [r for r in rs if r["status"] == "final"]
            agg = {"side": defaultdict(blank), "total": defaultdict(blank)}
            for r in fin:
                for bucket in ((r["grade"],) if r["grade"] in FLAGGED and r["result"] else ()) + ("ALL",):
                    res, odds = (r["result"], r["odds"]) if bucket != "ALL" else (r["lean_result"], -110)
                    if not res:
                        continue
                    for A in (agg, season):
                        d = A[r["market"]][bucket]
                        rec_add(d, res, odds)
                        if bucket != "ALL" and r.get("clv") not in ("", None):
                            d["clv_sum"] += float(r["clv"]); d["clv_n"] += 1
            games = [{"g": r["game"], "id": r["game_id"], "m": r["market"], "grade": r["grade"], "pick": r["pick"],
                      "line": r["line"], "lean": r["lean_side"], "score": f"{r['away_score']}-{r['home_score']}" if r["status"] == "final" else "",
                      "res": r["result"], "lean_res": r["lean_result"], "clv": r["clv"], "flag_line": r["flag_line"]}
                     for r in sorted(rs, key=lambda r: (r["kickoff_utc"], r["game"], r["market"]))]
            day_list.append({"day": day, "games": len({r["game_id"] for r in rs}), "final": len({r["game_id"] for r in fin}),
                             "side": {k: finish(dict(v)) for k, v in agg["side"].items()},
                             "total": {k: finish(dict(v)) for k, v in agg["total"].items()}, "picks": games})
        out["leagues"][lg] = {"tiers": TIER_ORDER[lg], "season": {mk: {k: finish(dict(v)) for k, v in season[mk].items()} for mk in season},
                              "days": day_list[::-1]}
    pdays = defaultdict(list)
    for r in props.values():
        pdays[r["day_et"]].append(r)
    out["props"] = {"season": props_metrics(list(props.values())),
                    "days": [{"day": d, **(props_metrics(pdays[d]) or {"calls": 0}), "open": sum(1 for r in pdays[d] if r.get("status") == "open"),
                              "void": len({r["player_id"] for r in pdays[d] if r.get("status") == "void"})}
                             for d in sorted(pdays, reverse=True)]}
    return out

# ------------------------------------------------------------------ runs
def boards():
    for p in glob.glob("data/boards/cfb_*.json"):
        yield "cfb", re.search(r"cfb_(.+)\.json", p).group(1), p
    for p in glob.glob("data/boards/nfl_*.json"):
        if "index" not in p:
            yield "nfl", re.search(r"nfl_(.+)\.json", p).group(1), p
    for p in glob.glob("data/boards/nba_*.json"):
        if "index" not in p:
            yield "nba", re.search(r"nba_(.+)\.json", p).group(1), p

def run():
    picks, props = load_csv(PICKS, PICK_COLS), load_csv(PROPS, PROP_COLS)
    at = now_utc()
    for lg, slate, p in boards():
        snapshot_board(json.load(open(p)), lg, slate, picks, at)
    for p in glob.glob("data/props/nfl_props_*.json"):
        snapshot_props(json.load(open(p)), props, at)
    settle_picks(picks); settle_props(props)
    save_csv(PICKS, picks, PICK_COLS); save_csv(PROPS, props, PROP_COLS)
    summ = summarize(picks, props)
    with open(SUMMARY, "w") as f:
        json.dump(summ, f, separators=(",", ":"))
    n_open = sum(1 for r in picks.values() if r["status"] != "final")
    print(f"screener grades: {len(picks)} game-markets ({n_open} open), {len(props)} prop rows")

def backfill(since):
    """Replay every committed version of the boards / prop files since `since` (a commit) through the snapshot step."""
    picks, props = load_csv(PICKS, PICK_COLS), load_csv(PROPS, PROP_COLS)
    log = subprocess.run(["git", "log", "--reverse", "--format=%H %cI", f"{since}..HEAD", "--", "data/boards", "data/props"],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    for line in filter(None, log):
        sha, when = line.split(" ", 1)
        at = dt.datetime.fromisoformat(when).astimezone(dt.timezone.utc).isoformat(timespec="minutes").replace("+00:00", "Z")
        files = subprocess.run(["git", "ls-tree", "--name-only", sha, "data/boards/", "data/props/"], capture_output=True, text=True).stdout.split()
        for f in files:
            m = re.search(r"data/boards/(cfb|nfl|nba)_(.+)\.json$", f)
            if m and "index" not in f:
                doc = json.loads(subprocess.run(["git", "show", f"{sha}:{f}"], capture_output=True, text=True).stdout)
                snapshot_board(doc, m.group(1), m.group(2), picks, at)
            elif re.search(r"data/props/nfl_props_.+\.json$", f):
                doc = json.loads(subprocess.run(["git", "show", f"{sha}:{f}"], capture_output=True, text=True).stdout)
                snapshot_props(doc, props, at)
    save_csv(PICKS, picks, PICK_COLS); save_csv(PROPS, props, PROP_COLS)
    print(f"backfill: {len(picks)} game-markets, {len(props)} prop rows")

if __name__ == "__main__":
    if "--backfill" in sys.argv:
        backfill(sys.argv[sys.argv.index("--backfill") + 1])
    run()
