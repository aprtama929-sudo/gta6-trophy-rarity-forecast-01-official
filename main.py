"""Week-one GTA VI trophy rarity forecaster (Python 3.13, stdlib only).

Reads one case file (makefun/gta6-trophies-case/v1) per run from the input
directory and writes five quantiles (5/25/50/75/95) of every trophy's Exophase
earn rate at the case target, scored by pinball loss on the logit scale.

Method, per trophy, all on the logit scale:

* Five median models, each anchored at the last observed rate so a model can
  only move the forecast by the change its slope implies over the horizon:
    1. persistence (no move),
    2. least-squares line of logit(rate) against ln(1 + hours since launch),
    3. the same line with its slope damped by 0.6,
    4. least-squares line of logit(rate) against plain hours since launch,
    5. exponential saturation in rate space: a ceiling picked from a grid by
       least squares on ln(ceiling - rate) against time, then the gap to the
       ceiling decays exponentially from the last rate.
  A model's parameters are fitted once per data prefix; scoring a horizon
  afterwards is a handful of arithmetic per model, which keeps a 358-read
  final case far inside the 8-second cap.
* The five are blended by an internal backtest: the series is cut at 40/55/70/
  85 percent of its span, each model is refit on the prefix and scored on the
  reads after the cut with errors normalised per square-root day; weights are
  proportional to 1 / (mean error + 0.15)^2. The blended move away from the
  last rate is shrunk to 85 percent, which guards against ceiling kinks the
  series has not shown yet.
* The spread comes from the backtest scatter per square-root day, shrunk
  toward the pooled cross-trophy scatter and toward the 0.5 random-walk
  scale, combined with how far the five models disagree, scaled by the
  square root of the horizon and clipped to [0.30, 2.5] logits. A history
  shorter than 12 hours gets a 1.0-logit floor: one Exophase refresh can
  rebase launch-night rates by more than that.

Everything is deterministic: no randomness, no clock reads, files processed
in name order. Any unexpected failure inside a case falls back to the plain
persistence forecast so a case can never come out invalid.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ZVALS = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

CUTS = (0.40, 0.55, 0.70, 0.85)
DAMP = 0.6
SHRINK = 0.85
ERR_PAD = 0.15
MAX_MOVE = 3.0          # cap on any single model's logit move over the horizon
SPREAD_LO = 0.30
SPREAD_HI = 2.5
SHORT_SPAN_H = 12.0
SHORT_FLOOR = 1.0
MAX_FIT_POINTS = 120    # thin long series before fitting (keeps the last read)
MAX_SCORE_POINTS = 40   # thin backtest suffixes before scoring
SAT_GRID = 25           # ceiling candidates for the saturation fit


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(v):
    if v >= 0:
        return 100.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(moment):
    return (moment - LAUNCH).total_seconds() / 3600.0


def thin(points, cap):
    """Evenly thin a list to at most cap entries, always keeping the last."""
    if len(points) <= cap:
        return list(points)
    step = (len(points) - 1) / float(cap - 1)
    picked = [points[int(round(i * step))] for i in range(cap - 1)]
    picked.append(points[-1])
    out = []
    for p in picked:
        if not out or p is not out[-1]:
            out.append(p)
    return out


def ols_slope(xs, ys):
    """Least-squares slope of ys on xs; 0.0 when it cannot be estimated."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        sxx += dx * dx
        sxy += dx * (y - my)
    if sxx <= 1e-12:
        return 0.0
    return sxy / sxx


def cap_move(delta):
    return max(-MAX_MOVE, min(MAX_MOVE, delta))


class FittedModels:
    """The five models' parameters for one data prefix, fitted once.

    Predictions at any later time are O(1) per model: two slope reads, one
    exponential for the saturation model, and the anchored last logit.
    """

    __slots__ = ("last_t", "last_rate", "last_logit", "slope_log", "slope_hrs", "sat")

    def __init__(self, fit):
        self.last_t, self.last_rate = fit[-1]
        self.last_logit = logit(self.last_rate)
        xs_log = [math.log1p(t) for t, _ in fit]
        ys = [logit(r) for _, r in fit]
        self.slope_log = ols_slope(xs_log, ys)
        self.slope_hrs = ols_slope([t for t, _ in fit], ys)
        self.sat = self._fit_saturation(fit)

    @staticmethod
    def _fit_saturation(fit):
        """Best (ceiling, decay slope) by LS on ln(ceiling - rate), or None."""
        if len(fit) < 3:
            return None
        top = max(r for _, r in fit)
        if top >= 99.9:
            return None
        t0 = fit[0][0]
        best = None  # (sse, ceiling, slope)
        for i in range(SAT_GRID):
            frac = (i + 1) / SAT_GRID
            ceiling = top + 0.02 + (99.99 - top) * frac * frac
            xs = []
            ys = []
            ok = True
            for t, r in fit:
                gap = ceiling - r
                if gap <= 0.005:
                    ok = False
                    break
                xs.append(t - t0)
                ys.append(math.log(gap))
            if not ok:
                continue
            slope = ols_slope(xs, ys)
            if slope >= 0:
                continue
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            sse = 0.0
            for x, y in zip(xs, ys):
                resid = y - (my + slope * (x - mx))
                sse += resid * resid
            if best is None or sse < best[0]:
                best = (sse, ceiling, slope)
        if best is None:
            return None
        return best[1], best[2]

    def predict(self, t_target):
        base = self.last_logit
        d_log = cap_move(self.slope_log * (math.log1p(t_target) - math.log1p(self.last_t)))
        d_hrs = cap_move(self.slope_hrs * (t_target - self.last_t))
        if self.sat is None:
            p_sat = base
        else:
            ceiling, slope = self.sat
            gap_last = ceiling - self.last_rate
            rate_at = ceiling - gap_last * math.exp(slope * (t_target - self.last_t))
            p_sat = logit(min(99.99, max(0.01, rate_at)))
        return [base, base + d_log, base + DAMP * d_log, base + d_hrs, p_sat]


class TrophySeries:
    """One trophy's usable observations: (hours since launch, rate percent)."""

    def __init__(self, obs):
        self.obs = obs  # non-empty, time-ordered
        self.last_t, self.last_rate = obs[-1]
        self.last_logit = logit(self.last_rate)
        self.span_h = obs[-1][0] - obs[0][0] if len(obs) > 1 else 0.0

    def backtest(self):
        """Per-model (mean |err|, rms err) per sqrt-day, plus pooled squares."""
        empty = ([None] * 5, [None] * 5, [])
        if len(self.obs) < 6 or self.span_h <= 0:
            return empty
        abs_sum = [0.0] * 5
        sq_sum = [0.0] * 5
        counts = [0] * 5
        pooled_sq = []
        t_start = self.obs[0][0]
        for frac in CUTS:
            cut_t = t_start + frac * self.span_h
            prefix = [p for p in self.obs if p[0] <= cut_t]
            suffix = [p for p in self.obs if p[0] > cut_t]
            if len(prefix) < 3 or not suffix:
                continue
            fitted = FittedModels(thin(prefix, MAX_FIT_POINTS))
            anchor_t = prefix[-1][0]
            for t_obs, rate in thin(suffix, MAX_SCORE_POINTS):
                days = max((t_obs - anchor_t) / 24.0, 1.0 / 1440.0)
                scale = math.sqrt(days)
                actual = logit(rate)
                preds = fitted.predict(t_obs)
                for m in range(5):
                    err = abs(preds[m] - actual) / scale
                    abs_sum[m] += err
                    sq_sum[m] += err * err
                    counts[m] += 1
                    pooled_sq.append(err * err)
        if not counts[0]:
            return empty
        mean_abs = [abs_sum[m] / counts[m] for m in range(5)]
        rms = [math.sqrt(sq_sum[m] / counts[m]) for m in range(5)]
        return mean_abs, rms, pooled_sq

    def forecast(self, t_target, horizon_days, pooled_sigma, bt=None):
        fitted = FittedModels(thin(self.obs, MAX_FIT_POINTS))
        preds = fitted.predict(t_target)
        mean_abs, rms, _ = bt if bt is not None else self.backtest()
        if mean_abs[0] is None:
            weights = [1.0 / 5.0] * 5
            own_sigma = 0.5
        else:
            raw = [1.0 / (mean_abs[m] + ERR_PAD) ** 2 for m in range(5)]
            total = sum(raw)
            weights = [w / total for w in raw]
            own_sigma = sum(weights[m] * rms[m] for m in range(5))
        blended = sum(weights[m] * preds[m] for m in range(5))
        median_logit = self.last_logit + SHRINK * (blended - self.last_logit)

        sigma_day = 0.6 * own_sigma + 0.4 * pooled_sigma
        sigma_day = 0.8 * sigma_day + 0.2 * 0.5
        disagree = math.sqrt(sum(weights[m] * (preds[m] - blended) ** 2 for m in range(5)))
        spread = math.sqrt((sigma_day * math.sqrt(max(horizon_days, 0.0))) ** 2 + (0.5 * disagree) ** 2)
        if self.span_h < SHORT_SPAN_H:
            spread = max(spread, SHORT_FLOOR)
        spread = min(SPREAD_HI, max(SPREAD_LO, spread))
        return [round4(inv_logit(median_logit + z * spread)) for z in ZVALS]


def persistence_rows(case):
    last = case["history"][-1]
    horizon_days = case["horizon_minutes"] / 1440.0
    spread = 0.5 * math.sqrt(max(horizon_days, 0.0))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit(rate if rate is not None else 0.01)
        rows.append({"id": trophy["id"], "q": [round4(inv_logit(lg + z * spread)) for z in ZVALS]})
    return rows


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels")
    history = case["history"]
    if not history:
        raise ValueError("no history")
    t_target = hours_since_launch(parse_time(case["target"]))
    horizon_days = case["horizon_minutes"] / 1440.0

    series = []
    last_rates = []
    for i in range(len(case["trophies"])):
        obs = []
        for read in history:
            rate = read["rates"][i]
            if rate is not None:
                obs.append((hours_since_launch(parse_time(read["at"])), float(rate)))
        series.append(TrophySeries(obs) if obs else None)
        if obs:
            last_rates.append(obs[-1][1])

    # Pooled backtest scatter across trophies (per sqrt-day), for shrinkage.
    pooled = []
    per_series_bt = []
    for s in series:
        if s is None:
            per_series_bt.append(None)
            continue
        bt = s.backtest()
        per_series_bt.append(bt)
        pooled.extend(bt[2])
    pooled_sigma = math.sqrt(sum(pooled) / len(pooled)) if pooled else 0.5

    fallback_rate = sorted(last_rates)[len(last_rates) // 2] if last_rates else 1.0
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        s = series[i]
        if s is None:
            lg = logit(fallback_rate)
            q = [round4(inv_logit(lg + z * 1.5)) for z in ZVALS]
        else:
            q = s.forecast(t_target, horizon_days, pooled_sigma, per_series_bt[i])
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = {"trophies": persistence_rows(case)}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
