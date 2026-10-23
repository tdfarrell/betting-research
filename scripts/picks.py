"""Shared pick rules for every sport (decided Oct 23, 2026).

Every spread and total on a board gets a pick: the side the model's number favors vs DraftKings.
Picks are tagged PLAY or LEAN when they clear that sport's bands; everything else is an untagged PICK
(including the old CHECK / big-gap games and FLAG games: they keep their side, graded like the rest).

Props (rebuilt Oct 23, see prop_pick): NFL only, from the recalibrated model; PLAY 60%+, LEAN 57-60%, otherwise
  no pick. NBA props have no model pick (it didn't beat the line).
The page (screener/index.html) applies the same prop rule, so tags match what gets graded.
"""
import json, math, os

# Game-line picks recorded and shown per sport (screener GAME_MODEL mirrors this). Since the v3 rebuild every game
# gets a Pick from the v3 model with its calibrated chance; no Lean / Play tags on game lines.
GAME_MODELS = {"cfb": True, "nfl": True, "nba": True, "cbb": True}

TAG = {"PLAY": "PLAY", "TARGET": "PLAY", "LEAN": "LEAN", "WATCH": "LEAN"}

def tag_of(grade):
    return TAG.get(grade or "", "PICK")

def _pick_text(abbr, line):
    return f"{abbr} pk" if line == 0 else f"{abbr} {line:+g}"

def finalize_market(g, mk):
    """Give a board market (g['side'] or g['total']) a pick. Since the Oct 23 rebuild, game lines get Picks only
    (no Lean / Play tags): the side the v3 model's calibrated chance favors (g['v3'], scripts/games_live.py),
    with that chance shown for context. Boards without v3 fall back to the old model's side."""
    s, d = g.setdefault(mk, {}), g.get("dk") or {}
    v = g.get("v3") or {}
    old = s.get("grade") or "PASS"
    if mk == "side":
        hs = d.get("home_spread")
        if hs is None:
            return
        if v.get("margin") is not None:
            gap = v["margin"] + hs                      # + = model likes the home side vs DK
            p = v.get("p_home")
            home = (p > 0.5) if p is not None and p != 0.5 else gap >= 0
            ch = (p if home else 1 - p) if p is not None else 0.5
            s["edge"], s["chance"] = round(gap, 1), round(ch, 3)
        elif s.get("edge") is not None:
            home = s["edge"] >= 0
        else:
            return
        line = hs if home else -hs
        s.update(pick_side="home" if home else "away", line=line,
                 pick=_pick_text(g.get("home_abbr") if home else g.get("away_abbr"), line),
                 odds=d.get("home_spread_odds") if home else d.get("away_spread_odds"))
    else:
        tot = d.get("total")
        if tot is None:
            return
        if v.get("total") is not None:
            gap = v["total"] - tot
            p = v.get("p_over")
            over = (p > 0.5) if p is not None and p != 0.5 else gap >= 0
            ch = (p if over else 1 - p) if p is not None else 0.5
            s["edge"], s["chance"] = round(gap, 1), round(ch, 3)
        elif s.get("edge") is not None:
            over = s["edge"] >= 0
        else:
            return
        s.update(pick_side="over" if over else "under", line=tot, pick=f"{'Over' if over else 'Under'} {tot:g}",
                 odds=d.get("over_odds") if over else d.get("under_odds"))
    s["was"] = old
    s["grade"] = "PICK"
    s.pop("why_untagged", None)

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
            if g[mk].get("chance") is not None:
                g[mk]["conf"] = g[mk]["chance"]       # the page's "confidence" = the v3 model's calibrated chance
    rows.sort(key=lambda g: g.get("kickoff_utc", ""))
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

# Props after the Oct 23 rebuild (claude/model-research-and-redesign.md):
#  NBA: the model couldn't beat the book's line in walk-forward tests (107k real lines, 2024-26; best versions -3% to
#       +2% ROI, nothing consistent), so per the rule "a model that doesn't beat the line doesn't ship", NBA props get
#       no model pick at all. The cards keep lines, hit rates and game logs.
#  NFL: the simulation's chance is recalibrated per market (data/models/nfl_props_calib.json, scripts/nfl_props_calib.py).
#       Walk-forward (passing TDs excluded), calibrated 60%+ won 62.4%, 57-60% won 57.2%, in both 2024 and 2025.
#       PLAY 60%+, LEAN 57-60%; below that, no pick. Passing TDs never picked (priced far from even).
NBA_MK = {"pts", "reb", "ast", "tpm", "pra", "pr", "pa", "ra", "stl", "blk"}
_NFL_CAL = None
def nfl_cal():
    global _NFL_CAL
    if _NFL_CAL is None:
        try:
            _NFL_CAL = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "models", "nfl_props_calib.json")))
        except Exception:
            _NFL_CAL = {}
    return _NFL_CAL

def calibrated_over(mk, p):
    c = (nfl_cal().get("markets") or {}).get(mk)
    if not c:
        return None
    p = min(max(p, 0.01), 0.99)
    return 1 / (1 + math.exp(-(c["a"] + c["b"] * math.log(p / (1 - p)))))

_NBA_CFG = None
def nba_cfg():
    global _NBA_CFG
    if _NBA_CFG is None:
        try:
            _NBA_CFG = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "models", "nba_props_v3.json")))
        except Exception:
            _NBA_CFG = {}
    return _NBA_CFG

NFL_LEAN = 0.60   # calibrated chance; walk-forward 62.4%
def prop_pick(m, mk, rec):
    """rec = DK line record {line, open, over, under}. Every line gets a pick (the side the calibrated chance favors);
    LEAN when the chance clears the bar: NBA 56%+ (v3 model, walk-forward 57.4%), NFL 60%+ (62.4%).
    Passing TDs: pick shown, never a Lean (priced far from even)."""
    line = rec.get("line")
    if line is None:
        return None
    if mk in NBA_MK:
        cfg = nba_cfg()
        if not cfg or not m.get("v3q"):
            return None
        import nba_props_live as NL
        x = NL.pick(cfg, mk, m["v3q"], float(line))
        p, chance, over, tag = x["p_over"], x["chance"], x["side"] == "over", x["tag"]
        p_raw = None
    else:
        p_raw = model_over(m, mk, float(line))
        if p_raw is None:
            return None
        p = calibrated_over(mk, p_raw)
        if p is None:
            return None
        over = p >= 0.5
        chance = p if over else 1 - p
        tag = "LEAN" if chance >= NFL_LEAN and mk not in (nfl_cal().get("no_pick") or []) else "PICK"
    nv = novig_over(rec)
    out = {"side": "over" if over else "under", "line": float(line), "p_over": round(p, 3), "p_raw": None if p_raw is None else round(p_raw, 3),
           "chance": round(chance, 3), "novig_over": round(nv, 3) if nv is not None else None, "edge": round(chance - 0.5, 3), "tag": tag,
           "odds": rec.get("over" if over else "under")}
    return out
