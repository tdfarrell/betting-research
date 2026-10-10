# betting-research

Automated data pulls and graded slates for CFB/NFL betting research. **DraftKings lines only.**

Web app: **ballwatcher.com/screener/** (source: `screener/index.html`; sport tabs CFB · NFL, NBA/CBB to come). `CFB-screener/` now redirects there.

| Data | Source | Schedule | Files |
|---|---|---|---|
| DK spreads, totals, moneylines (open + current) | ESPN public scoreboard | every 30 min | `data/odds/latest_{cfb,nfl}.json`, `history_*.csv` |
| SP+, success rate, EPA, explosiveness, season stats, results, talent | CollegeFootballData.com | ~6am / 6pm ET | `data/cfbd/*.json` |
| NFL play-by-play (summarized per team-game / QB-game), schedule, projected QBs, injuries | nflverse | ~6am / 6pm ET | `data/nfl/*.csv` |
| Graded boards | `scripts/regrade.py` (CFB), `scripts/regrade_nfl.py` (NFL) | after every pull | `data/boards/` |
| NFL flag log + CLV vs DK close | `scripts/regrade_nfl.py` | after every pull | `data/nfl/flags.csv` |

The CFBD key lives only in the repo secret `CFBD_KEY`. Run manually: Actions → "Pull betting data" → Run workflow.
Caveats: ESPN feed is unofficial and may change; main lines only. Always confirm price in the DK app.

## NFL model (current season first)
- Grades come from three **current-season** ratings (EPA/play, success rate, points; opponent and QB adjusted), fitted to recent closing lines and averaged into the BW number. Last season only seeds Weeks 1–5 and fades to zero by Week 6.
- Last season's rating, Elo and the market rating are **angles**: shown, never graded.
- `python scripts/nfl_backtest.py 2022 2025` → `data/nfl/backtest_summary.json`. No tier has beaten the closing line in backtests, so TARGET/WATCH flags are tracked on CLV in `data/nfl/flags.csv`.
- One-time: Actions → "Backfill NFL DraftKings history" pulls past DK open/close lines from ESPN (`data/nfl/dk_history.csv`) for an early-line test.
