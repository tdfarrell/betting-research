"""Keep games that have already kicked off on a slate's board.

The graders only see games that haven't started, so without this a Saturday board would lose its noon games
by the evening. Rows for started games are carried over from the previous version of the file, frozen as they
were at kickoff and marked "started": true; their team cards are carried over too."""
import datetime as dt, json, os

def merge_started(path, rows, teams):
    if not os.path.exists(path):
        return rows, teams
    try:
        old = json.load(open(path))
    except Exception:
        return rows, teams
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    have = {r["id"] for r in rows}
    kept = []
    for r in old.get("games", []):
        if r["id"] not in have and r.get("kickoff_utc", "9999") <= now:
            r["started"] = True
            kept.append(r)
            for t in (r.get("home"), r.get("away")):
                if t in old.get("teams", {}) and t not in teams:
                    teams[t] = old["teams"][t]
    return rows + kept, teams
