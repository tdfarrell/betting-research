#!/usr/bin/env python3
"""Prop pick backtest against real sportsbook lines (data/props_hist/, from scripts/props_history_pull.py).

For every historical player line, the model is rebuilt with only what was known before the game (same code as the
live cards), it picks over or under exactly like scripts/picks.py, and the pick is graded on the box score.
Output: win rate and ROI by edge band, a threshold scan, and the confidence table the live picks use.

  python scripts/props_backtest.py nba 2026 [2025]     -> data/props_hist/nba_backtest_picks.csv.gz
  python scripts/props_backtest.py nfl 2024 2025       -> data/props_hist/nfl_backtest_picks.csv.gz
  python scripts/props_backtest.py summarize           -> data/grades/prop_confidence.json (+ printed scan)

NBA lines carry prices (edge = model chance - no-vig chance). NFL lines don't (edge = model chance - 50%,
graded at -110). Lines are ESPN BET's where ESPN no longer serves DraftKings' for that game.
"""
import csv, gzip, json, math, os, sys
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from picks import novig_over, implied

H = "data/props_hist"
SHRINK = 80.0

def american_profit(o):
    try:
        x = float(str(o).replace("+", ""))
    except (TypeError, ValueError):
        x = -110.0
    if not x:
        x = -110.0
    return x / 100 if x > 0 else 100 / -x

def load_lines(lg, seasons):
    """One MAIN line per (game, player, market). ESPN BET lists alternate lines in the same market, so:
    priced (NBA): the line with both an over and an under price whose no-vig chance is closest to 50%;
    unpriced (NFL): the line listed most often (main lines carry both sides), ties to the middle line."""
    out = []
    for s in seasons:
        p = f"{H}/{lg}_{s}.csv.gz"
        if not os.path.exists(p):
            continue
        g = defaultdict(lambda: defaultdict(dict))
        meta = {}
        cnt = defaultdict(lambda: defaultdict(int))
        for r in csv.DictReader(gzip.open(p, "rt")):
            k = (r["game_id"], r["athlete_id"], r["market"])
            ln = float(r["line"])
            meta[k] = (r["date"], r["provider"])
            cnt[k][ln] += 1
            if r["side"] in ("over", "under") and r["price"] not in ("", None):
                g[k][ln].setdefault(r["side"], r["price"])
        for k, lines in cnt.items():
            best = None
            for ln, sides in g[k].items():
                if "over" in sides and "under" in sides:
                    nv = novig_over(sides)
                    if nv is not None and (best is None or abs(nv - .5) < best[0]):
                        best = (abs(nv - .5), ln, sides)
            if best and best[0] <= 0.15:
                rec = {"line": best[1], "over": best[2]["over"], "under": best[2]["under"]}
            elif lg == "nfl":
                top = max(lines.values())
                cands = sorted(ln for ln, c in lines.items() if c == top)
                rec = {"line": cands[len(cands) // 2], "over": "", "under": ""}
            else:
                continue
            out.append({"game_id": k[0], "athlete_id": k[1], "market": k[2], "date": meta[k][0], "provider": meta[k][1],
                        "season": int(s), "n_lines": len(lines), **rec})
    return out

# ------------------------------------------------------------------ NBA
def nba(seasons):
    import nba_props as NP
    pg, tg = NP.load()
    lines = load_lines("nba", seasons)
    LC = {r["game_id"]: r for r in csv.DictReader(open("data/nba/lines_close.csv"))}
    box = {(r.game_id, r.player_id): r for r in pg.itertuples()}
    by_date = defaultdict(list)
    for r in lines:
        by_date[(r["season"], r["date"])].append(r)
    rng = np.random.default_rng(11)
    out = []
    first = {S: min(d for (s_, d) in by_date if s_ == S) for S in {k[0] for k in by_date}}
    have_prior = set(pg.season.unique())
    for (S, date) in sorted(by_date):
        # a season with no prior season in the data: wait two weeks so the model has this season's games
        if S - 1 not in have_prior and (pd.Timestamp(date) - pd.Timestamp(first[S])).days < 14:
            continue
        day = NP.Day(pg, tg, S, date)
        cache = {}
        for r in by_date[(S, date)]:
            x = box.get((r["game_id"], r["athlete_id"]))
            if x is None or x.dnp == 1 or x.min <= 0:
                continue   # didn't play: DraftKings voids
            key = r["athlete_id"]
            if key not in cache:
                lc = LC.get(r["game_id"], {})
                try:
                    hs, tot = float(lc.get("home_spread")), float(lc.get("total"))
                    spread = hs if x.home == 1 else -hs; impl = tot / 2 - spread / 2
                except (TypeError, ValueError):
                    spread = impl = None
                p = NP.project(day, r["athlete_id"], NP.pos3(x.pos), x.team, x.opp, spread, impl)
                if p and not all(np.isfinite(v) and v < 5 for v in p["rate"].values()):
                    p = None
                cache[key] = NP.simulate(p, rng, 2000) if p else None
            sims = cache[key]
            if sims is None or r["market"] not in sims:
                continue
            line = float(r["line"])
            p_over = float((sims[r["market"]] > line).mean())
            nv = novig_over(r)
            if nv is None:
                continue
            over = p_over >= nv
            actual = getattr(x, r["market"])
            if actual == line:
                continue
            win = int((actual > line) == over)
            odds = r["over"] if over else r["under"]
            out.append([S, date, r["game_id"], r["athlete_id"], r["market"], line, round(p_over, 3), round(nv, 3),
                        "over" if over else "under", round(abs(p_over - nv), 3), round(p_over if over else 1 - p_over, 3),
                        odds, win, round(american_profit(odds) if win else -1.0, 3), r["provider"]])
        if len(out) and len(by_date) and (len(out) % 5000 < 60):
            print(f"  {date}: {len(out)} picks")
    cols = ["season", "date", "game_id", "player", "market", "line", "p_over", "novig_over", "side", "edge", "chance", "odds", "win", "profit", "provider"]
    df = pd.DataFrame(out, columns=cols)
    df.to_csv(f"{H}/nba_backtest_picks.csv.gz", index=False, compression="gzip")
    print(f"nba: {len(df)} graded picks")

# ------------------------------------------------------------------ NFL
def nfl(seasons):
    import nfl_props as F
    lines = load_lines("nfl", seasons)
    ids = {r["espn_id"]: r["gsis_id"] for r in csv.DictReader(open(f"{H}/nfl_player_ids.csv"))}
    seasons = [int(s) for s in seasons]
    D = F.setup(range(min(seasons), max(seasons) + 1))
    sched = D["sched"]
    by_espn = {str(r.espn).split(".")[0]: r for r in sched.itertuples() if not pd.isna(r.espn)}
    all_roles = {}
    for S in range(min(seasons) - 1, max(seasons) + 1):
        all_roles.update(F.season_roles(D, S, 19, prior_only=True))
    pgi = {(r.game_id, r.player_id): r for r in D["pg"].itertuples()}
    by_week = defaultdict(list)
    for r in lines:
        g = by_espn.get(r["game_id"])
        pid = ids.get(r["athlete_id"])
        if g is None or not pid:
            continue
        by_week[(int(g.season), int(g.week))].append((r, g, pid))
    rng = np.random.default_rng(13)
    out = []
    coefs = {}
    for (S, W) in sorted(by_week):
        if S not in coefs:
            coefs[S] = F.fit_coefs(D, S); D["role_means"] = F.role_means(D, S, all_roles)
        wk = D["wk_cache"].get((S, W)) or F.Week(D, S, W)
        cache = {}
        for r, g, pid in by_week[(S, W)]:
            x = pgi.get((g.game_id, pid))
            role = all_roles.get((S, W, pid))
            if x is None or not role:
                continue
            if pid not in cache:
                home = x.team == g.home_team
                sp, tot = g.spread_line, g.total_line
                if pd.isna(sp) or pd.isna(tot):
                    cache[pid] = None
                else:
                    impl = tot / 2 + (sp / 2 if home else -sp / 2); spr = -sp if home else sp
                    p = F.project(wk, D, pid, x.position, role, x.team, x.opp, impl, spr, coefs[S])
                    cache[pid] = (p, F.simulate(p, rng, 2000))
            if not cache[pid]:
                continue
            p, sims = cache[pid]
            mk, line = r["market"], float(r["line"])
            if mk == "pass_td":
                lam = p["lam_ptd"]; k = math.floor(line) + 1
                p_over = 1 - sum(math.exp(-lam) * lam ** j / math.factorial(j) for j in range(k))
            elif mk in sims:
                p_over = float((sims[mk] > line).mean())
            else:
                continue
            actual = getattr(x, F.STAT[mk]) if F.STAT[mk] != "rr_yds" else x.rr_yds
            if actual == line:
                continue
            over = p_over >= 0.5
            win = int((actual > line) == over)
            out.append([S, W, g.game_id, pid, mk, line, round(p_over, 3), "", "over" if over else "under",
                        round(abs(p_over - 0.5), 3), round(max(p_over, 1 - p_over), 3), "-110", win, round(0.909 if win else -1.0, 3), r["provider"]])
        print(f"  {S} wk{W}: {len(out)} picks")
    cols = ["season", "date", "game_id", "player", "market", "line", "p_over", "novig_over", "side", "edge", "chance", "odds", "win", "profit", "provider"]
    df = pd.DataFrame(out, columns=cols)
    df.to_csv(f"{H}/nfl_backtest_picks.csv.gz", index=False, compression="gzip")
    print(f"nfl: {len(df)} graded picks")

# ------------------------------------------------------------------ summary + thresholds
def scan(df, key):
    rows = []
    for lo in np.arange(0.0, 0.30, 0.01):
        m = df[df[key] >= lo]
        if len(m) < 30:
            break
        rows.append({"min": round(float(lo), 2), "n": int(len(m)), "win": round(float(m.win.mean()), 3), "roi": round(float(m.profit.mean()), 3),
                     "per_day": round(len(m) / max(1, df.date.nunique()), 1)})
    return rows

def bands_table(df, key, edges):
    out = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else 9
        m = df[(df[key] >= lo) & (df[key] < hi)]
        n, w = len(m), float(m.win.sum())
        out.append({"lo": lo, "hi": None if hi >= 9 else hi, "n": int(n), "w": int(w), "raw": round(w / n, 3) if n else None,
                    "conf": round((w + 0.5 * SHRINK) / (n + SHRINK), 3), "roi": round(float(m.profit.mean()), 3) if n else None})
    return out

def summarize():
    out = {"shrink_games": SHRINK, "tables": {}, "scan": {}}
    for lg, key in (("nba", "edge"), ("nfl", "chance")):
        p = f"{H}/{lg}_backtest_picks.csv.gz"
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        edges = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20] if lg == "nba" else [0.5, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75]
        cedges = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.78, 0.8, 0.85]
        out["tables"][lg] = {"key": "chance", "games": int(df.game_id.nunique()), "picks": int(len(df)), "days": int(df.date.nunique()),
                             "sides": {sd: bands_table(df[df.side == sd], "chance", cedges) for sd in ("over", "under")},
                             "bands_" + key: bands_table(df, key, edges),
                             "by_market": {mk: {"n": int(len(g)), "win": round(float(g.win.mean()), 3), "roi": round(float(g.profit.mean()), 3)}
                                           for mk, g in df.groupby("market")}}
        out["scan"][lg] = scan(df, key)
        print(lg, out["tables"][lg]["picks"], "picks over", out["tables"][lg]["days"], "days")
        for sd, bs in out["tables"][lg]["sides"].items():
            for b in bs:
                print("  ", sd, b)
        for s in out["scan"][lg]:
            print("   >=", s)
    json.dump(out, open("data/grades/prop_confidence.json", "w"), indent=1)

if __name__ == "__main__":
    if sys.argv[1] == "nba":
        nba([int(x) for x in sys.argv[2:]])
    elif sys.argv[1] == "nfl":
        nfl([int(x) for x in sys.argv[2:]])
    else:
        summarize()
