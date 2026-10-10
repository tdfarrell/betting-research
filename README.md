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

## NFL player props (hit rates first)
- `python scripts/nfl_props.py` → `data/props/nfl_props_<season>_wk<NN>.json` + `data/props/index.json` (built with the twice-daily stats pull and Sunday 11:17am ET).
- Markets and tiers: receiving / rushing yds 25-40-50-…-100, rush+rec 40-…-110-125, passing yds 180-…-300, receptions 2-8, passing TDs 1-3, anytime TD.
- Each tier shows L3 and season hit rates, the opponent's rate allowed to the same role (roles set entering each game), and the model probability + fair odds. Flags tiers where hit rate and model point opposite ways.
- `python scripts/nfl_props.py backtest 2023 2025` → `data/props/backtest_summary.json` (calibration and Brier scores vs hit-rate baselines).
- Player-level inputs come from `data/nfl/player_games.csv` (nflverse weekly stats + red-zone touches + snap counts) and `data/nfl/roster_current.csv`.
