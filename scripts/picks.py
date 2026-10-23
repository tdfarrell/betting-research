"""Shared pick rules for every sport (decided Oct 23, 2026).

Every spread and total on a board gets a pick: the side the model's number favors vs DraftKings.
Picks are tagged PLAY or LEAN when they clear that sport's bands; everything else is an untagged PICK
(including the old CHECK / big-gap games and FLAG games: they keep their side, graded like the rest).

Props: every player with a DraftKings over/under line gets an over or under pick.
  With prices (NBA): edge = model chance - DK no-vig chance for the side the model prefers.
      PLAY 8%+, LEAN 4-8%, untagged below 4% and at 15%+ (gaps that big are usually news the model lacks).
  Without prices (NFL): model chance for the side it prefers.
      PLAY 62%+, LEAN 56-62%, untagged below 56% and at 75%+.
The page (screener/index.html) applies the same prop rule, so tags match what gets graded.
"""
import math

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

def finalize_rows(rows):
    for g in rows:
        if g.get("started"):
            continue
        for mk in ("side", "total"):
            finalize_market(g, mk)
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

PRICED = {"play": 0.08, "lean": 0.04, "too_big": 0.15}
UNPRICED = {"play": 0.62, "lean": 0.56, "too_big": 0.75}

def prop_pick(m, mk, rec):
    """rec = DK line record {line, open, over, under}. Returns the pick dict or None."""
    line = rec.get("line")
    if line is None:
        return None
    p = model_over(m, mk, float(line))
    if p is None:
        return None
    nv = novig_over(rec)
    if nv is not None:
        over = p >= nv
        edge = abs(p - nv)
        tag = "PICK" if edge >= PRICED["too_big"] else "PLAY" if edge >= PRICED["play"] else "LEAN" if edge >= PRICED["lean"] else "PICK"
        why = "Big gap vs DraftKings: usually minutes, role or injury news the model doesn't have" if edge >= PRICED["too_big"] else None
        chance = p if over else 1 - p
    else:
        over = p >= 0.5
        chance = p if over else 1 - p
        edge = chance - 0.5
        tag = "PICK" if chance >= UNPRICED["too_big"] else "PLAY" if chance >= UNPRICED["play"] else "LEAN" if chance >= UNPRICED["lean"] else "PICK"
        why = "Model far from DraftKings' line: usually news the model doesn't have" if chance >= UNPRICED["too_big"] else None
    out = {"side": "over" if over else "under", "line": float(line), "p_over": round(p, 3), "chance": round(chance, 3),
           "novig_over": round(nv, 3) if nv is not None else None, "edge": round(edge, 3), "tag": tag,
           "odds": rec.get("over" if over else "under")}
    if why:
        out["why_untagged"] = why
    return out
