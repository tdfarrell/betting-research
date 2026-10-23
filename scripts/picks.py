"""Shared pick rules for every sport (decided Oct 23, 2026).

Every spread and total on a board gets a pick: the side the model's number favors vs DraftKings.
Picks are tagged PLAY or LEAN when they clear that sport's bands; everything else is an untagged PICK
(including the old CHECK / big-gap games and FLAG games: they keep their side, graded like the rest).

Props: every player with a DraftKings over/under line gets an over or under pick (the side the model favors
  vs DK's no-vig chance, or vs 50% when DK shows no prices). Tags come from the prop backtest (see prop_pick):
  NFL unders only, PLAY at 78%+ model chance, LEAN 70-78%. NBA props and NFL overs are untagged for now.
The page (screener/index.html) applies the same prop rule, so tags match what gets graded.
"""
import json, math, os

TAG = {"PLAY": "PLAY", "TARGET": "PLAY", "LEAN": "LEAN", "WATCH": "LEAN"}

def tag_of(grade):
    return TAG.get(grade or "", "PICK")

def _pick_text(abbr, line):
    return f"{abbr} pk" if line == 0 else f"{abbr} {line:+g}"

def finalize_market(g, mk):
    """Give a board market (g['side'] or g['total']) a pick and a PLAY / LEAN / PICK tag."""
    s, d = g.get(mk) or {}, g.get("dk") or {}
    edge = s.get("edge")
    if edge is None:
        return
    old = s.get("grade") or "PASS"
    tag = tag_of(old)
    if mk == "side":
        hs = d.get("home_spread")
        if hs is None:
            return
        home = edge >= 0
        line = hs if home else -hs
        s.update(pick_side="home" if home else "away", line=line,
                 pick=_pick_text(g.get("home_abbr") if home else g.get("away_abbr"), line),
                 odds=d.get("home_spread_odds") if home else d.get("away_spread_odds"))
    else:
        tot = d.get("total")
        if tot is None:
            return
        over = edge >= 0
        s.update(pick_side="over" if over else "under", line=tot, pick=f"{'Over' if over else 'Under'} {tot:g}",
                 odds=d.get("over_odds") if over else d.get("under_odds"))
    if tag == "PICK" and old != "PASS":
        s["why_untagged"] = {"CHECK": "Big gap vs DraftKings: usually news the model doesn't have",
                             "FLAG": "; ".join(g.get("flags") or []) or "Flagged"}.get(old, old)
    elif tag == "PICK" and s.get("adjust"):
        s["why_untagged"] = "; ".join(s["adjust"])
    s["was"] = old
    s["grade"] = tag

# ------------------------------------------------------------------ confidence (scripts/confidence.py)
_CONF = None
def _conf_tables():
    global _CONF
    if _CONF is None:
        p = os.path.join(os.path.dirname(__file__), "..", "data", "grades", "confidence.json")
        try:
            _CONF = json.load(open(p)).get("tables", {})
        except Exception:
            _CONF = {}
    return _CONF

def conf_for(league, mk, x):
    """Backtest win rate (shrunk) for a pick whose gap is x (z units or points, per the table)."""
    t = _conf_tables().get(f"{league}_{mk}")
    if not t or x is None:
        return None
    for b in t["bands"]:
        if x >= b["lo"] and (b["hi"] is None or x < b["hi"]):
            return {"conf": b["conf"], "conf_n": b["n"], "conf_raw": b["raw"]}
    return None

def _gap(g, mk, league, usual):
    e = (g.get(mk) or {}).get("edge")
    if e is None:
        return None
    t = _conf_tables().get(f"{league}_{mk}") or {}
    if t.get("unit") == "pts":
        return abs(e)
    if league == "cfb":
        return abs(e) / usual if usual else None
    b = g.get("bands") if mk == "side" else g.get("tbands")
    sd = b[2] if b and len(b) > 2 and b[2] else None
    return abs(e) / sd if sd else None

def finalize_rows(rows, league=None, usual=None):
    """usual: CFB's slate-level usual gap for sides (calibration bands_side[1])."""
    for g in rows:
        if g.get("started"):
            continue
        for mk in ("side", "total"):
            finalize_market(g, mk)
            if league:
                c = conf_for(league, mk, _gap(g, mk, league, usual))
                if c:
                    g[mk].update(c)
    order = {"PLAY": 0, "LEAN": 1, "PICK": 2}
    rows.sort(key=lambda g: (min(order.get(g["side"].get("grade"), 3), order.get(g["total"].get("grade"), 3)), g.get("kickoff_utc", "")))
    return rows

# ------------------------------------------------------------------ props
def implied(o):
    try:
        x = float(str(o).replace("+", ""))
    except (TypeError, ValueError):
        return None
    if not x:
        return None
    return -x / (-x + 100) if x < 0 else 100 / (x + 100)

def novig_over(rec):
    a, b = implied(rec.get("over")), implied(rec.get("under"))
    return a / (a + b) if a is not None and b is not None else None

def model_over(m, mk, line):
    """Model chance the stat goes OVER a .5 line, same order of sources as the page's modelOver()."""
    k = math.floor(line) + 1
    for t in m.get("tiers", []):
        if t["t"] == k:
            return t["p"]
    q = m.get("q")
    if q:
        n = len(q) - 1
        if line < q[0]:
            return 1.0
        if line >= q[n]:
            return 0.0
        i = 0
        while i < n and q[i + 1] <= line:
            i += 1
        frac = (line - q[i]) / (q[i + 1] - q[i]) if q[i + 1] > q[i] else 0
        return 1 - (i + frac) / n
    if m.get("ge") is not None and m.get("ge0") is not None:
        if k < m["ge0"]:
            return 1.0
        j = k - m["ge0"]
        return m["ge"][j] / 1000 if j < len(m["ge"]) else 0.0
    if m.get("ge") is not None:
        return m["ge"][k] if k < len(m["ge"]) else 0.0
    if mk == "pass_td" and m.get("proj") is not None:
        lam = m["proj"]
        return 1 - sum(math.exp(-lam) * lam ** j / math.factorial(j) for j in range(k))
    return None

# Prop tags, from the backtest against real sportsbook lines (scripts/props_backtest.py, Oct 23 2026):
#  NBA (2024-26, 107k main lines): the model adds almost nothing beyond the line itself (win 50.7%, ROI -5.6% at
#      every edge size), so NBA props are picked and graded but never tagged until the model earns it.
#  NFL (2024-25, 15k lines, graded at -110): overs never beat the line (49.7%); unders did, more so the surer
#      the model: 70-78% model chance won ~57%, 78%+ won ~63% in both seasons. So PLAY = under at 78%+,
#      LEAN = under at 70-78%. Overs are picked and graded, untagged.
NBA_MK = {"pts", "reb", "ast", "tpm", "pra", "pr", "pa", "ra", "stl", "blk"}
NFL_UNDER = {"play": 0.78, "lean": 0.70}
PRICED = {"play": None, "lean": None}     # kept for reference: no priced (NBA) prop is tagged
UNPRICED = NFL_UNDER

def prop_pick(m, mk, rec):
    """rec = DK line record {line, open, over, under}. Returns the pick dict or None."""
    line = rec.get("line")
    if line is None:
        return None
    p = model_over(m, mk, float(line))
    if p is None:
        return None
    nv = novig_over(rec)
    over = p >= (nv if nv is not None else 0.5)
    chance = p if over else 1 - p
    edge = abs(p - nv) if nv is not None else chance - 0.5
    why = None
    if mk in NBA_MK:
        tag = "PICK"
        why = "NBA prop model hasn't beaten the line in the backtest yet: picked and graded, not tagged"
    elif over:
        tag = "PICK"
        why = "Overs haven't beaten the line in the backtest: picked and graded, not tagged"
    else:
        tag = "PLAY" if chance >= NFL_UNDER["play"] else "LEAN" if chance >= NFL_UNDER["lean"] else "PICK"
    out = {"side": "over" if over else "under", "line": float(line), "p_over": round(p, 3), "chance": round(chance, 3),
           "novig_over": round(nv, 3) if nv is not None else None, "edge": round(edge, 3), "tag": tag,
           "odds": rec.get("over" if over else "under")}
    if why:
        out["why_untagged"] = why
    return out
