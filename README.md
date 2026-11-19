# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aprtama929-sudo/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aprtama929-sudo/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A deterministic, standard-library-only Python forecaster for the make.fun bounty "Forecast how many GTA VI players earn each trophy in week one". For every PlayStation 5 trophy of Grand Theft Auto VI it reads the Exophase earn-rate history recorded up to the case checkpoint and writes five quantiles (5/25/50/75/95) of the rate at the target, scored by pinball loss on the logit scale. Per trophy, five anchored median models on the logit scale (persistence, a log-time trend, its damped variant, an hours trend, and an exponential-saturation fit) are blended by an internal backtest over four span cuts, with a spread built from backtest scatter, cross-trophy pooling and model disagreement.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=j0mxz08v&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
