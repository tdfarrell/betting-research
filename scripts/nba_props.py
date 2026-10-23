#!/usr/bin/env python3
"""NBA player prop cards: DraftKings lines (with prices), hit rates, game logs with splits, and a model.

  python scripts/nba_props.py                    cards for every upcoming slate day -> data/props/nba_props_<date>.json
  python scripts/nba_props.py backtest 2026      calibration test on a past season -> data/props/nba_backtest_summary.json

Markets: points, rebounds, assists, threes, PRA, P+R, P+A, R+A, steals, blocks.
Model (per player, per game), all from games before the slate:
  minutes    recent games weighted toward the latest (each game back counts 0.85x), blended with last season's
             average worth 3 games; starters lose a little in expected blowouts (|DK spread| > 7)
  rates      per-minute rates this season, shrunk toward last season's (250 minutes' worth) and toward the
             position average when there's little history
  matchup    what this opponent allows per minute to the position this season, vs the league (shrunk by games
             played), and the expected pace of the game; points also lean toward DK's implied team total
  sim        4,000 games: minutes ~ normal, each stat Poisson with a gamma-spread rate (over-dispersed),
             combos summed from the same simulated game
Hit rates: L3/L5/L10/L20, this season, last season, H2H come from the game log on each card (worked out in the page).
"""
import csv, datetime as dt, json, math, os, sys
from collections import defaultdict
import numpy as np
import pandas as pd

D, OUT = "data/nba", "data/props"
ET = dt.timezone(dt.timedelta(hours=-4))
STATS = ["pts", "reb", "ast", "tpm", "stl", "blk"]
COMBO = {"pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"], "pa": ["pts", "ast"], "ra": ["reb", "ast"]}
MARKETS = ["pts", "reb", "ast", "tpm", "pra", "pr", "pa", "ra", "stl", "blk"]
TIERS = {"pts": [10, 15, 20, 25, 30, 35, 40], "reb": [4, 6, 8, 10, 12, 14], "ast": [2, 4, 6, 8, 10, 12],
         "tpm": [1, 2, 3, 4, 5, 6], "pra": [15, 20, 25, 30, 35, 40, 45, 50], "pr": [15, 20, 25, 30, 35, 40],
         "pa": [15, 20, 25, 30, 35, 40], "ra": [5, 10, 15, 20], "stl": [1, 2, 3], "blk": [1, 2, 3]}
LABEL = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "tpm": "Threes", "pra": "Pts + Reb + Ast", "pr": "Pts + Reb",
         "pa": "Pts + Ast", "ra": "Reb + Ast", "stl": "Steals", "blk": "Blocks"}
DISP = {"pts": 14.0, "reb": 32.0, "ast": 28.0, "tpm": 10.0, "stl": 12.0, "blk": 8.0}   # gamma shape (larger = steadier), tuned on 2025-26
K_MIN = 250.0          # minutes of last season's rate blended into this season's
K_POS = 120.0          # minutes of the position average blended in
PRIOR_GAMES_MIN = 2.0  # last season's minutes average counts as 2 games
DECAY = 0.85
MIN_SD = (0.14, 1.5)   # minutes spread: sd = a * projected + b
RATE_SCALE = {"pts": 1.03}   # the 2025-26 backtest ran ~3% short on points
N_SIM = 4000
GLOG_COLS = ["season", "date", "team", "opp", "home", "rest", "starter", "min", "pts", "reb", "ast", "tpm", "stl", "blk",
             "pra", "pr", "pa", "ra", "out"]

NBA_TEAMS = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
             "NO", "NY", "OKC", "ORL", "PHI", "PHX", "POR", "SA", "SAC", "TOR", "UTAH", "WSH"}   # skips All-Star games
def pos3(p):
    p = (p if isinstance(p, str) else "").upper()
    return "C" if p.startswith("C") else "G" if p.startswith(("G", "PG", "SG")) else "F" if p else "F"

def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")

def rnd(x, n=1):
    try:
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)
    except (TypeError, ValueError):
        return None

def fair(p):
    if p is None or p <= .005 or p >= .995:
        return None
    return int(round(-100 * p / (1 - p))) if p >= .5 else int(round(100 * (1 - p) / p))

def load():
    pg = pd.read_csv(f"{D}/player_games.csv", dtype={"player_id": str, "game_id": str})
    pg = pg[pg.team.isin(NBA_TEAMS) & pg.opp.isin(NBA_TEAMS)]
    for c in ["min", "pts", "reb", "ast", "tpm", "stl", "blk", "tov", "dnp", "starter", "home", "season"]:
        pg[c] = pd.to_numeric(pg[c], errors="coerce").fillna(0)
    for k, parts in COMBO.items():
        pg[k] = sum(pg[p] for p in parts)
    pg["pos3"] = pg["pos"].map(pos3)
    tg = pd.read_csv(f"{D}/team_games.csv", dtype={"game_id": str})
    tg = tg[tg.team.isin(NBA_TEAMS) & tg.opp.isin(NBA_TEAMS)]
    for c in ["poss", "pts", "season", "rest", "home"]:
        tg[c] = pd.to_numeric(tg[c], errors="coerce")
    return pg, tg

class Day:
    """What's known before `day` in season S."""
    def __init__(self, pg, tg, S, day):
        self.S, self.day = S, day
        played = pg[(pg.dnp == 0) & (pg["min"] > 0)]
        self.cur = played[(played.season == S) & (played.date < day)]
        # this season's box-score rows incl. coach's-decision DNPs (0 minutes): minutes and who's on the roster
        allc = pg[(pg.season == S) & (pg.date < day)]
        cd = allc[(allc.dnp == 1) & allc.dnp_reason.fillna("").str.upper().str.startswith("COACH")]
        self.min_rows = self.cur   # DK voids props when a player doesn't play, so minutes are "if he plays"
        self.on_team = allc.sort_values("date").groupby("player_id").team.last().to_dict()
        self.team_games = allc.groupby("team").game_id.nunique().to_dict()
        self.pri = played[played.season == S - 1]
        base = self.cur if len(self.cur) > 2000 else pd.concat([self.pri, self.cur])
        # position per-minute averages (players with 10+ minutes)
        b = base[base["min"] >= 10]
        self.pos_rate = {p: {s: g[s].sum() / g["min"].sum() for s in STATS} for p, g in b.groupby("pos3")}
        # player totals
        agg = lambda df: df.groupby("player_id").agg(g=("game_id", "nunique"), mins=("min", "sum"), **{s: (s, "sum") for s in STATS})
        self.C, self.P = agg(self.cur), agg(self.pri)
        # opponent vs position per minute this season, as a factor vs the league
        cb = self.cur[self.cur["min"] >= 10]
        lg = {p: {s: g[s].sum() / g["min"].sum() for s in STATS} for p, g in cb.groupby("pos3")} if len(cb) else self.pos_rate
        self.dvp = {}
        for (opp, p), g in cb.groupby(["opp", "pos3"]):
            n = g.game_id.nunique(); w = n / (n + 10.0)
            self.dvp[(opp, p)] = {s: 1 + w * ((g[s].sum() / g["min"].sum()) / lg[p][s] - 1) if lg.get(p, {}).get(s) else 1.0 for s in STATS}
            self.dvp[(opp, p)]["n"] = n
        # per-game allowed to position (for the card's rank column)
        self.dvp_pg = {}
        for (opp, p), g in cb.groupby(["opp", "pos3"]):
            n = g.game_id.nunique()
            self.dvp_pg[(opp, p)] = {s: g[s].sum() / n for s in STATS + list(COMBO)} | {"n": n}
        # pace (possessions per game), this season with last season as a 10-game prior
        t = tg[(tg.season == S) & (tg.date < day)]; tp = tg[tg.season == S - 1]
        lgp = (t.poss.mean() if len(t) > 60 else tp.poss.mean()) or 99.0
        self.lg_pace = lgp
        self.pace = {}
        for team in set(t.team) | set(tp.team):
            a, b_ = t[t.team == team].poss, tp[tp.team == team].poss
            self.pace[team] = ((a.sum() + 10 * (b_.mean() if len(b_) else lgp)) / (len(a) + 10))
        self.team_pts = {team: (g.pts.mean()) for team, g in (t if len(t) > 60 else pd.concat([tp, t])).groupby("team")}

    def minutes(self, pid, pos, spread=None):
        g = self.min_rows[self.min_rows.player_id == pid].sort_values("date")
        prior = self.P.loc[pid, "mins"] / self.P.loc[pid, "g"] if pid in self.P.index and self.P.loc[pid, "g"] > 0 else None
        if len(g) == 0 and prior is None:
            return None, 0.0
        w = np.array([DECAY ** i for i in range(len(g))][::-1])
        num = float((g["min"].values * w).sum()); den = float(w.sum())
        if prior is not None:
            num += PRIOR_GAMES_MIN * prior; den += PRIOR_GAMES_MIN
        m = num / den
        gp = g[g["min"] > 0]
        if len(gp) >= 3:
            start_rate = float(gp.tail(5).starter.mean())
        else:   # few or no games this season: blend in last season's starts
            last = self.pri[self.pri.player_id == pid].sort_values("date").tail(20)
            n_c, n_p = len(gp), len(last)
            start_rate = float((gp.starter.sum() + last.starter.mean() * (5 - n_c) * (n_p > 0)) / max(1, n_c + (5 - n_c) * (n_p > 0))) if n_c + n_p else 0.0
        if spread is not None and m >= 28 and abs(spread) > 7:
            m *= 1 - 0.01 * (abs(spread) - 7)
        return m, start_rate

    def rate(self, pid, pos, s):
        pr = self.pos_rate.get(pos, {}).get(s, 0.0)
        prior, km = pr, K_POS
        if pid in self.P.index and self.P.loc[pid, "mins"] > 0:
            pm = self.P.loc[pid, "mins"]
            prior = (self.P.loc[pid, s] + pr * K_POS) / (pm + K_POS); km = K_MIN
        if pid in self.C.index and self.C.loc[pid, "mins"] > 0:
            return (self.C.loc[pid, s] + prior * km) / (self.C.loc[pid, "mins"] + km)
        return prior

def project(day, pid, pos, team, opp, spread, implied):
    mins, start_rate = day.minutes(pid, pos, spread)
    if mins is None:
        return None
    pace = ((day.pace.get(team, day.lg_pace) + day.pace.get(opp, day.lg_pace)) / 2) / day.lg_pace
    dv = day.dvp.get((opp, pos), {})
    mu = {}
    for s in STATS:
        r = day.rate(pid, pos, s) * dv.get(s, 1.0) * pace
        if s in RATE_SCALE:
            r *= RATE_SCALE[s]
        if s == "pts" and implied and day.team_pts.get(team):
            r *= 1 + 0.5 * (implied / day.team_pts[team] - 1)
        mu[s] = r
    return {"mins": mins, "start_rate": start_rate, "rate": mu, "pace": pace, "dvp": {s: rnd(dv.get(s, 1.0), 3) for s in STATS}}

def simulate(p, rng, n=N_SIM, disp=None):
    d = disp or DISP
    m = np.clip(rng.normal(p["mins"], MIN_SD[0] * p["mins"] + MIN_SD[1], n), 0, 48)
    out = {}
    for s in STATS:
        lam = p["rate"][s] * m * rng.gamma(d[s], 1 / d[s], n)
        out[s] = rng.poisson(lam)
    for k, parts in COMBO.items():
        out[k] = sum(out[x] for x in parts)
    out["min"] = m
    return out

def ge_compact(v):
    """P(stat >= k) in thousandths, only where it's between 0.1% and 99.9%: {"ge0": first k, "ge": [...]}."""
    ks = np.arange(0, int(v.max()) + 2)
    p = np.array([(v >= k).mean() for k in ks])
    keep = np.where((p < 0.999) & (p > 0.001))[0]
    if not len(keep):
        return {"ge0": 0, "ge": []}
    return {"ge0": int(keep[0]), "ge": [int(round(1000 * x)) for x in p[keep[0]:keep[-1] + 1]]}

def tier_probs(sims, mk):
    return [float((sims[mk] >= t).mean()) for t in TIERS[mk]]

# ------------------------------------------------------------------ cards
def hits(vals, tiers):
    return [[int(sum(1 for v in vals if v >= t)), len(vals)] for t in tiers]

def disagreement(mk, tiers, probs, season, l3):
    out = []
    for t, p, (h, n), (h3, n3) in zip(tiers, probs, season, l3):
        if n >= 5 and abs(h / n - p) >= 0.30 and (h / n - 0.5) * (p - 0.5) < 0:
            out.append(f" {t}+: hit {h}/{n} this season, model {p:.0%}")
        elif n3 == 3 and ((h3 == 3 and p < 0.3) or (h3 == 0 and p > 0.7)):
            out.append(f" {t}+: {h3}/3 last 3, model {p:.0%}")
    return out

def team_mates(day, pg, team):
    g = day.cur[day.cur.team == team]
    if g.game_id.nunique() < 2:
        g = pd.concat([day.pri[day.pri.team == team], g])
    a = g.groupby("player_id").agg(mpg=("min", "mean"), g=("game_id", "nunique"), name=("name", "last"), pos=("pos3", "last"))
    a = a[a.g >= 2].sort_values("mpg", ascending=False).head(7)
    return [{"id": pid, "name": r["name"], "pos": r["pos"], "mpg": rnd(r.mpg)} for pid, r in a.iterrows()]

def game_log(pg, pid, S, day, mates, team_games_by, rest_map):
    rows = pg[(pg.player_id == pid) & (pg.season >= S - 1) & (pg.date < day) & (pg.dnp == 0) & (pg["min"] > 0)].sort_values("date")
    out = []
    for x in rows.itertuples():
        played = team_games_by.get((x.game_id, x.team), set())
        on_team = team_games_by.get(("season", int(x.season), x.team), set())
        mask = 0
        for i, m in enumerate(mates):
            if m["id"] != pid and m["id"] in on_team and m["id"] not in played:
                mask |= 1 << i
        out.append([int(x.season), x.date, x.team, x.opp, int(x.home), rest_map.get((x.game_id, x.team)), int(x.starter), int(round(x.min)),
                    int(x.pts), int(x.reb), int(x.ast), int(x.tpm), int(x.stl), int(x.blk), int(x.pra), int(x.pr), int(x.pa), int(x.ra), mask])
    return out

def build_cards(pg, tg, S, day_str, games, lines, inj):
    day = Day(pg, tg, S, day_str)
    rng = np.random.default_rng(int(day_str.replace("-", "")))
    played = pg[(pg.dnp == 0) & (pg["min"] > 0) & (pg.season >= S - 1)]
    tgb = {k: set(v) for k, v in played.groupby(["game_id", "team"]).player_id}
    for (s, t), v in played.groupby(["season", "team"]).player_id:
        tgb[("season", int(s), t)] = set(v)
    rest_of = {}
    last = played[played.season == S].groupby("team").date.max().to_dict()
    for t, d in last.items():
        rest_of[t] = (dt.date.fromisoformat(day_str) - dt.date.fromisoformat(d)).days
    # rest per game row for the log (team_games.rest)
    rest_map = {(r.game_id, r.team): (None if pd.isna(r.rest) else int(r.rest)) for r in tg.itertuples()}
    latest = pg[pg.season >= S - 1].sort_values("date").groupby("player_id").last()
    mates_all, cards, dvp_rank = {}, [], {}
    # defense vs position ranks: per-game allowed to the position this season (30 = allows the most)
    for p in ("G", "F", "C"):
        for mk in MARKETS:
            vals = [(opp, v[mk]) for (opp, pp), v in day.dvp_pg.items() if pp == p]
            vals.sort(key=lambda x: x[1])
            for i, (opp, v) in enumerate(vals):
                dvp_rank[(opp, p, mk)] = {"rank": i + 1, "of": len(vals), "avg": rnd(v), "n": day.dvp_pg[(opp, p)]["n"]}
    for b in games:
        for team, opp, home in ((b["home"], b["away"], True), (b["away"], b["home"], False)):
            hs, tot = b.get("home_spread"), b.get("total")
            try:
                spread = float(hs) * (1 if home else -1); implied = float(tot) / 2 - spread / 2
            except (TypeError, ValueError):
                spread = implied = None
            mates = team_mates(day, pg, team); mates_all[team] = mates
            roster = latest[(latest.team == team)]
            if day.team_games.get(team, 0) >= 1:   # once a team has played, only players in its box scores this season
                roster = roster[[day.on_team.get(pid) == team for pid in roster.index]]
            cand = []
            for pid, r in roster.iterrows():
                st = (inj.get(pid) or {}).get("status")
                if st and st.lower() == "out":
                    continue
                pos = pos3(r.pos)
                p = project(day, pid, pos, team, opp, spread, implied)
                if not p:
                    continue
                has_line = pid in lines and any(v.get("game_id") == str(b["game_id"]) for v in lines[pid].values())
                if p["mins"] >= 16 or has_line:
                    cand.append((pid, r, pos, p, st, has_line))
            cand.sort(key=lambda x: -x[3]["mins"])
            for pid, r, pos, p, st, has_line in cand[:10]:
                sims = simulate(p, rng)
                g = day.cur[day.cur.player_id == pid].sort_values("date")
                l3 = g.tail(3)
                dv = day.cur[(day.cur.opp == opp) & (day.cur.pos3 == pos) & (day.cur["min"] >= 20)].sort_values("date")
                mkts, flags = {}, []
                for mk in MARKETS:
                    tiers = TIERS[mk]
                    probs = tier_probs(sims, mk)
                    sv, l3v, dvv = g[mk].tolist(), l3[mk].tolist(), dv[mk].tolist()
                    hs_, h3, hd = hits(sv, tiers), hits(l3v, tiers), hits(dvv, tiers)
                    dis = disagreement(mk, tiers, probs, hs_, h3)
                    flags += [f"{LABEL[mk]}{x}" for x in dis]
                    mkts[mk] = {"tiers": [{"t": t, "l3": a, "season": s_, "def": c, "p": round(pr, 3), "fair": fair(pr)} for t, a, s_, c, pr in zip(tiers, h3, hs_, hd, probs)],
                                "avg": rnd(np.mean(sv)) if sv else None, "l3_vals": [int(v) for v in l3v], "l3_opps": l3.opp.tolist(),
                                "season_vals": [int(v) for v in sv], "def_avg": rnd(np.mean(dvv)) if dvv else None, "def_n": len(dvv),
                                "proj": rnd(float(np.median(sims[mk]))), "mean": rnd(float(np.mean(sims[mk]))), "disagree": dis,
                                **ge_compact(sims[mk]),
                                "dvp": dvp_rank.get((opp, pos, mk))}
                k = dt.datetime.fromisoformat(b["kickoff_utc"].replace("Z", "+00:00")).astimezone(ET)
                glog = game_log(pg, pid, S, day_str, mates, tgb, rest_map)
                cards.append({"id": pid, "name": r["name"], "pos": pos, "role": pos, "starter": p["start_rate"] >= 0.5,
                              "team": team, "opp": opp, "home": home, "game_id": str(b["game_id"]), "kick_et": k.strftime("%a %-I:%M %p"),
                              "kickoff_utc": b["kickoff_utc"], "headshot": f"https://a.espncdn.com/i/headshots/nba/players/full/{pid}.png",
                              "status": st, "games": int(len(g)), "implied": rnd(implied), "spread": spread, "rest": rest_of.get(team),
                              "usage": {"min_pg": rnd(g["min"].mean()) if len(g) else None, "start_rate": rnd(p["start_rate"], 2),
                                        "last_min": int(g["min"].iloc[-1]) if len(g) else None},
                              "model": {"mins": rnd(p["mins"]), "pace": rnd(p["pace"], 3), "dvp": p["dvp"],
                                        "per36": {s: rnd(p["rate"][s] * 36) for s in STATS}},
                              "prior": {"weight": rnd(PRIOR_GAMES_MIN / (PRIOR_GAMES_MIN + len(g)), 2), "dropped": []},
                              "markets": mkts, "flags": flags, "glog": glog})
    return cards, mates_all

def main():
    pg, tg = load()
    board = json.load(open("data/odds/latest_nba.json"))
    S = int(pg.season.max())
    L = json.load(open(f"{OUT}/dk_lines_nba.json")).get("lines", {}) if os.path.exists(f"{OUT}/dk_lines_nba.json") else {}
    inj = {r["player_id"]: r for r in csv.DictReader(open(f"{D}/injuries.csv"))} if os.path.exists(f"{D}/injuries.csv") else {}
    by_day = defaultdict(list)
    for b in board["games"]:
        if b.get("status") == "pre":
            by_day[dt.datetime.fromisoformat(b["kickoff_utc"].replace("Z", "+00:00")).astimezone(ET).date().isoformat()].append(b)
    idx_path = f"{OUT}/nba_index.json"
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {"days": []}
    days = {d["key"]: d for d in idx.get("days", [])}
    ahead = sorted(by_day)[:3]   # DK posts player lines a day or so out; later slates get cards as they come up
    for old in sorted(by_day)[3:]:
        if os.path.exists(f"{OUT}/nba_props_{old}.json"):
            os.remove(f"{OUT}/nba_props_{old}.json")
    for day in ahead:
        cards, mates = build_cards(pg, tg, S, day, by_day[day], L, inj)
        path = f"{OUT}/nba_props_{day}.json"
        if os.path.exists(path):   # keep players whose game already started
            try:
                have = {(c["id"], c["game_id"]) for c in cards}
                for c in json.load(open(path)).get("cards", []):
                    if (c["id"], c["game_id"]) not in have and c.get("kickoff_utc", "9999") <= now_iso():
                        c["started"] = True; cards.append(c)
            except Exception:
                pass
        doc = {"key": day, "date": day, "season": S, "league": "nba", "generated_utc": now_iso(), "odds_pulled_utc": board["pulled_utc"],
               "tiers": TIERS, "labels": LABEL, "glog_cols": GLOG_COLS, "mates": mates, "cards": cards}
        json.dump(doc, open(path, "w"), separators=(",", ":"))
        days[day] = {"key": day, "date": day, "players": len(cards), "flags": sum(1 for c in cards if c["flags"])}
        print(f"nba props {day}: {len(cards)} players")
    cutoff = (dt.datetime.now(ET).date() - dt.timedelta(days=14)).isoformat()
    idx = {"generated_utc": now_iso(), "days": sorted([d for k, d in days.items() if k >= cutoff and os.path.exists(f"{OUT}/nba_props_{k}.json")], key=lambda d: d["key"])}
    if os.path.exists(f"{OUT}/nba_backtest_summary.json"):
        idx["backtest"] = json.load(open(f"{OUT}/nba_backtest_summary.json")).get("headline")
    json.dump(idx, open(idx_path, "w"), indent=1)

# ------------------------------------------------------------------ backtest
def backtest(S, step=1, disp=None):
    pg, tg = load()
    rows = []
    days = sorted(pg[(pg.season == S)].date.unique())
    rng = np.random.default_rng(7)
    for i, d in enumerate(days):
        if i < 1 or i % step:
            continue
        day = Day(pg, tg, S, d)
        today = pg[(pg.season == S) & (pg.date == d) & (pg.dnp == 0) & (pg["min"] > 0)]
        for x in today.itertuples():
            pos = pos3(x.pos)
            p = project(day, x.player_id, pos, x.team, x.opp, None, None)
            if not p or p["mins"] < 16:
                continue
            sims = simulate(p, rng, 1000, disp)
            g = day.cur[day.cur.player_id == x.player_id]
            for mk in ("pts", "reb", "ast", "tpm", "pra"):
                sv = g[mk].values
                for t, pr in zip(TIERS[mk], tier_probs(sims, mk)):
                    rows.append({"date": d, "mk": mk, "t": t, "p": pr, "hit": int(getattr(x, mk) >= t),
                                 "hr_season": (sv >= t).mean() if len(sv) >= 3 else np.nan,
                                 "hr_l10": (sv[-10:] >= t).mean() if len(sv) >= 3 else np.nan})
    df = pd.DataFrame(rows)
    out = {"season": S, "predictions": len(df), "markets": {}}
    for mk, m in df.groupby("mk"):
        h = m.dropna(subset=["hr_season"])
        cal = m.groupby(pd.cut(m.p, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])).agg(n=("hit", "size"), model=("p", "mean"), actual=("hit", "mean"))
        out["markets"][mk] = {"n": len(m), "brier_model": round(float(((h.p - h.hit) ** 2).mean()), 4),
                              "brier_season_hr": round(float(((h.hr_season * .85 + .075 - h.hit) ** 2).mean()), 4),
                              "brier_l10_hr": round(float(((h.hr_l10 * .85 + .075 - h.hit) ** 2).mean()), 4),
                              "calibration": [{"bin": str(b), "n": int(r.n), "model": rnd(r.model, 3), "actual": rnd(r.actual, 3)} for b, r in cal.iterrows() if r.n]}
    h = df.dropna(subset=["hr_season"])
    out["headline"] = {"season": S, "predictions": len(df), "brier_model": round(float(((h.p - h.hit) ** 2).mean()), 4),
                       "brier_season_hr": round(float(((h.hr_season * .85 + .075 - h.hit) ** 2).mean()), 4),
                       "brier_l10_hr": round(float(((h.hr_l10 * .85 + .075 - h.hit) ** 2).mean()), 4)}
    return out

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        S = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
        summ = backtest(S, step=int(sys.argv[3]) if len(sys.argv) > 3 else 2)
        json.dump(summ, open(f"{OUT}/nba_backtest_summary.json", "w"), indent=1)
        print(json.dumps(summ["headline"], indent=1))
        for mk, v in summ["markets"].items():
            print(mk, {k: v[k] for k in v if k != "calibration"}, [(c["model"], c["actual"], c["n"]) for c in v["calibration"]])
    else:
        main()
