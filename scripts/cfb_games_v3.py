#!/usr/bin/env python3
"""College football game lines v3: accuracy-first margin and total predictions, walk-forward by week, then
compared with the lines (scripts/games_v3_common.py).

Past SP+ is only published as end-of-season ratings (it would leak results), so the backtestable inputs are:
pre-game Elo (CFBD), home / neutral, conference game, and this season's opponent-adjusted scoring ratings built
from earlier weeks only (ridge on points for / against, prior = last season's ratings shrunk toward average).

  python scripts/cfb_games_v3.py      -> data/cfbd/v3_preds.csv + report
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(__file__))
import games_v3_common as V

def ratings_before(past, teams, prior, k=3.0):
    """Offense / defense points ratings: pts = mu + O[t] - D[o] + hfa*home, ridge toward prior."""
    idx = {t: i for i, t in enumerate(teams)}; n = len(teams)
    rows, y, w = [], [], []
    for r in past.itertuples():
        for t, o, hm, pts in ((r.home, r.away, 0 if r.neutral else 1, r.home_points), (r.away, r.home, 0, r.away_points)):
            x = np.zeros(1 + 2 * n); x[0] = hm; x[1 + idx[t]] = 1; x[1 + n + idx[o]] = -1
            rows.append(x); y.append(pts); w.append(1.0)
    for t in teams:
        po, pd_ = prior.get(t, (0.0, 0.0))
        x = np.zeros(1 + 2 * n); x[1 + idx[t]] = 1; rows.append(x); y.append(po); w.append(k)
        x = np.zeros(1 + 2 * n); x[1 + n + idx[t]] = 1; rows.append(x); y.append(pd_); w.append(k)
    X, y = np.array(rows), np.array(y, float)
    mu = np.average(y[:len(past) * 2]) if len(past) else 28.0
    m = Ridge(alpha=1.0, fit_intercept=False).fit(X, y - np.r_[np.full(len(past) * 2, mu), np.zeros(2 * n)], sample_weight=w)
    c = m.coef_
    return mu, c[0], {t: (c[1 + idx[t]], c[1 + n + idx[t]]) for t in teams}

def build():
    h = pd.read_csv("data/cfbd/history_games.csv", dtype={"id": str})
    h = h[(h.home_div == "fbs") & (h.away_div == "fbs") & h.home_points.notna()].copy()
    h["neutral"] = h.neutral.fillna(0).astype(int)
    out = []
    prior = {}
    for s in sorted(h.season.unique()):
        hs = h[h.season == s]
        teams = sorted(set(hs.home) | set(hs.away))
        end = None
        for w in sorted(hs.week.unique()):
            past, cur = hs[hs.week < w], hs[hs.week == w]
            mu, hfa, R = ratings_before(past, teams, prior)
            for r in cur.itertuples():
                oh, dh = R.get(r.home, (0, 0)); oa, da = R.get(r.away, (0, 0))
                ph = mu + oh - da + (0 if r.neutral else hfa); pa = mu + oa - dh
                out.append({"id": r.id, "season": s, "week": w, "date": str(r.start)[:10], "home": r.home, "away": r.away,
                            "pts_margin": ph - pa, "pts_total": ph + pa, "elo_diff": (r.home_elo or 0) - (r.away_elo or 0),
                            "neutral": r.neutral, "conf_game": int(bool(r.conference_game)), "gp": len(past) / max(1, len(teams)) * 2,
                            "margin": r.home_points - r.away_points, "total": r.home_points + r.away_points,
                            "close": -r.spread if pd.notna(r.spread) else np.nan, "close_total": r.total if pd.notna(r.total) else np.nan})
        mu, hfa, R = ratings_before(hs, teams, prior)
        prior = {t: (0.5 * o, 0.5 * d) for t, (o, d) in R.items()}
    return pd.DataFrame(out)

FEATS = ["pts_margin", "pts_total", "elo_diff", "neutral", "conf_game", "gp", "week"]

if __name__ == "__main__":
    d = build()
    d.to_csv("data/cfbd/v3_games.csv", index=False)
    p = V.walk(d, FEATS, min_train=800, period="W")
    p.to_csv("data/cfbd/v3_preds.csv", index=False)
    V.report(p, "CFB", old=("pts_margin", "pts_total"))
    h = pd.read_csv("data/cfbd/history_games.csv", dtype={"id": str})
    ref = pd.DataFrame({"game_id": h.id, "ref_home_spread": h.spread, "ref_total": h.total, "refopen_home_spread": h.spread_open, "refopen_total": h.total_open})
    V.tiers(p, "cfb", which=("refopen", "ref", "open", "close"), extra_lines=ref, min_train=500)

# ------------------------------------------------------------------ live (scripts/regrade.py -> games_live.attach)
_LIVE = {}
def live_ratings(day):
    """This season's opponent-adjusted scoring ratings from completed FBS games before `day`
    (data/cfbd/games.json), prior = last season's final ratings shrunk 50%, as in training."""
    if day in _LIVE:
        return _LIVE[day]
    import json
    G = json.load(open("data/cfbd/games.json"))
    rows = [r for r in G["rows"] if r.get("completed") and r.get("homeClassification") == "fbs" and r.get("awayClassification") == "fbs"
            and r.get("homePoints") is not None and str(r.get("startDate", ""))[:10] < day]
    season = G.get("year")
    cur = pd.DataFrame([{"home": r["homeTeam"], "away": r["awayTeam"], "neutral": int(bool(r.get("neutralSite"))),
                         "home_points": r["homePoints"], "away_points": r["awayPoints"], "week": r["week"]} for r in rows])
    h = pd.read_csv("data/cfbd/history_games.csv")
    h = h[(h.home_div == "fbs") & (h.away_div == "fbs") & h.home_points.notna() & (h.season == season - 1)].copy()
    h["neutral"] = h.neutral.fillna(0).astype(int)
    tl = sorted(set(h.home) | set(h.away))
    _, _, Rl = ratings_before(h, tl, {})
    prior = {t: (0.5 * o, 0.5 * d) for t, (o, d) in Rl.items()}
    teams = sorted(set(cur.home) | set(cur.away) | set(prior)) if len(cur) else sorted(prior)
    mu, hfa, R = ratings_before(cur, teams, prior)
    out = {"mu": mu, "hfa": hfa, "R": R, "gp": len(cur) / max(1, len(set(cur.home) | set(cur.away))) * 2 if len(cur) else 0,
           "week": int(cur.week.max()) + 1 if len(cur) else 1}
    _LIVE[day] = out
    return out

def live_feats(day, home, away, neutral, conf_game, elo_h, elo_a):
    L = live_ratings(day)
    oh, dh = L["R"].get(home, (0, 0)); oa, da = L["R"].get(away, (0, 0))
    ph = L["mu"] + oh - da + (0 if neutral else L["hfa"]); pa = L["mu"] + oa - dh
    return {"pts_margin": ph - pa, "pts_total": ph + pa, "elo_diff": (elo_h or 0) - (elo_a or 0) if elo_h and elo_a else 0,
            "neutral": int(bool(neutral)), "conf_game": int(bool(conf_game)), "gp": L["gp"], "week": L["week"]}
