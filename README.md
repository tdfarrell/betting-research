# betting-research

Automated data pulls for CFB/NFL betting research. **DraftKings lines only.**

| Data | Source | Schedule | Files |
|---|---|---|---|
| DK spreads, totals, moneylines (open + current) | ESPN public scoreboard | every 30 min | `data/odds/latest_{cfb,nfl}.json`, `history_*.csv` |
| SP+, success rate, EPA, explosiveness, season stats, results, talent | CollegeFootballData.com | ~6am / 6pm ET | `data/cfbd/*.json` |

The CFBD key lives only in the repo secret `CFBD_KEY`. Run manually: Actions → "Pull betting data" → Run workflow.
Caveats: ESPN feed is unofficial and may change; main lines only. Always confirm price in the DK app.
