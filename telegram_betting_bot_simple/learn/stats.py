"""Statistics for the learning loop, in numpy and the standard library only (no scipy).

- scores: log loss and Brier (binary), ranked probability score, count log score, discrete CRPS, the
  non-randomised PIT histogram for counts and the variance ratio (Czado, Gneiting & Held 2009);
- distributions: Student t tail (regularised incomplete beta), a vectorised log-gamma (Lanczos), the negative
  binomial pmf/cdf with ``size`` (variance = mean + mean^2 / size);
- paired tests: Diebold-Mariano on daily sums with a Newey-West variance and the Harvey-Leybourne-Newbold
  correction, the day-block bootstrap, Holm's step-down adjustment and a one-sided CUSUM (Page 1954);
- calibration: CORP reliability (isotonic regression by pool-adjacent-violators) with a consistency band from
  resampling a perfectly calibrated forecaster, and the calibration slope and intercept with 95% ranges (Cox 1958);
- fitting: Newton's method with a numeric Hessian and the lower-confidence-bound step (``ai/learning.py``'s
  "keep only what's 95% sure").
"""

from __future__ import annotations

import datetime
import math
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

P_FLOOR = 1e-4  # engine/consensus.P_FLOOR: chances are kept inside [P_FLOOR, 1 - P_FLOOR] before logs
Z95 = 1.959963984540054
SURE = 1.645  # ai/learning.SURE: standard errors taken off a fitted move
_N = NormalDist()


# --- dates ---------------------------------------------------------------------------------------------------------

def signed(x: float) -> str:
    """A change in log loss for the page and the ledger: 4 decimals with its sign ("+0.0004"), or 2 significant
    digits when it is that small (-0.000037, not "-0.0000")."""
    x = float(x)
    if x == 0 or abs(x) >= 0.00005:
        return f"{x:+.4f}"
    digits = max(4, 1 - int(math.floor(math.log10(abs(x)))))
    return f"{x:+.{digits}f}"


def day_of(ts: float) -> str:
    """The UTC calendar day of an epoch time, "2026-10-28"."""
    return datetime.datetime.fromtimestamp(float(ts), datetime.timezone.utc).strftime("%Y-%m-%d")


def week_of(ts: float) -> str:
    """The ISO week (Monday to Sunday, UTC) of an epoch time, "2026-W44"."""
    year, week, _ = datetime.datetime.fromtimestamp(float(ts), datetime.timezone.utc).isocalendar()
    return f"{year}-W{week:02d}"


def week_bounds(week: str) -> Tuple[float, float]:
    """(start, end) epoch seconds of an ISO week "2026-W44": Monday 00:00 UTC to the next Monday."""
    year, number = week.split("-W")
    monday = datetime.datetime.fromisocalendar(int(year), int(number), 1).replace(tzinfo=datetime.timezone.utc)
    return monday.timestamp(), monday.timestamp() + 7 * 86400


# --- scores --------------------------------------------------------------------------------------------------------

def clip_p(p: Any) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), P_FLOOR, 1 - P_FLOOR)


def log_loss(p: Any, y: Any) -> np.ndarray:
    """Per-game log loss of the chance ``p`` of an event that happened (``y`` 1) or not (0)."""
    p, y = clip_p(p), np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(p: Any, y: Any) -> np.ndarray:
    """Per-game (p - y)^2 (for two outcomes the same as engine/consensus.score's half-sum Brier)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    return (p - y) ** 2


def rps(p: Sequence[float], outcome_index: int) -> float:
    """Ranked probability score of ordered outcomes (home, draw, away): mean squared gap of the cumulative chances."""
    k = len(p)
    cum_p = np.cumsum(p)[:-1]
    cum_o = np.cumsum([1.0 if i == outcome_index else 0.0 for i in range(k)])[:-1]
    return float(np.sum((cum_p - cum_o) ** 2) / (k - 1))


def logit(p: Any) -> Any:
    p = clip_p(p)
    return np.log(p / (1 - p))


def sigmoid(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


# --- distributions -------------------------------------------------------------------------------------------------

_LANCZOS = (0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
            12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)


def lgamma(x: Any) -> np.ndarray:
    """log Gamma(x) for x >= 0.5, vectorised (Lanczos, g = 7: about 1e-15 relative)."""
    x = np.asarray(x, dtype=float) - 1.0
    a = np.full_like(x, _LANCZOS[0])
    for i in range(1, 9):
        a = a + _LANCZOS[i] / (x + i)
    t = x + 7.5
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * np.log(t) - t + np.log(a)


def nb_logpmf(y: Any, mu: Any, size: Any) -> np.ndarray:
    """log P(Y = y) for a negative binomial with mean ``mu`` and ``size`` r (variance mu + mu^2 / r)."""
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    r = np.asarray(size, dtype=float)
    return (lgamma(y + r) - lgamma(r) - lgamma(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))


def poisson_logpmf(y: Any, mu: Any) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-12)
    return y * np.log(mu) - mu - lgamma(y + 1)


def nb_pmf_table(mu: Any, size: Any, kmax: int) -> np.ndarray:
    """P(Y = k) for k = 0..kmax, one row per forecast: shape (n, kmax + 1)."""
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    size = np.broadcast_to(np.asarray(size, dtype=float), mu.shape)
    k = np.arange(kmax + 1, dtype=float)
    return np.exp(nb_logpmf(k[None, :], mu[:, None], size[:, None]))


def count_scores(pmf: np.ndarray, y: Any, bins: int = 10) -> Dict[str, Any]:
    """Scores of count forecasts given as a pmf table (n, K): mean log score (lower is better), mean discrete CRPS,
    the non-randomised PIT histogram (``bins`` densities, about 1 each when calibrated: a U shape means too
    confident, a hump too cautious), the variance ratio (about 1), plus the per-game log score and CRPS."""
    pmf = np.asarray(pmf, dtype=float)
    y = np.asarray(y, dtype=int)
    n, size = pmf.shape
    if np.any(y >= size) or np.any(y < 0):
        raise ValueError("an observed count lies outside the pmf table")
    rows = np.arange(n)
    ls = -np.log(np.maximum(pmf[rows, y], 1e-300))
    cdf = np.cumsum(pmf, axis=1)
    k = np.arange(size)
    crps = np.sum((cdf - (y[:, None] <= k[None, :])) ** 2, axis=1)
    upper = cdf[rows, y]
    lower = np.where(y > 0, cdf[rows, np.maximum(y - 1, 0)], 0.0)
    edges = np.linspace(0, 1, bins + 1)
    width = np.maximum(upper - lower, 1e-12)
    pit = np.zeros(bins)
    for j in range(bins):
        overlap = np.clip(np.minimum(upper, edges[j + 1]) - np.maximum(lower, edges[j]), 0, None)
        pit[j] = np.mean(overlap / width)
    mean = pmf @ k
    var = pmf @ (k ** 2) - mean ** 2
    ratio = float(np.mean((y - mean) ** 2) / max(np.mean(var), 1e-12))
    return {"n": int(n), "log_score": float(ls.mean()), "crps": float(crps.mean()), "pit": (pit * bins).tolist(),
            "variance_ratio": ratio, "per_game_log_score": ls, "per_game_crps": crps}


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """The regularised incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_cdf(t: float, df: float) -> float:
    """P(T <= t) for Student's t with ``df`` degrees of freedom."""
    if not math.isfinite(t):
        return 1.0 if t > 0 else 0.0
    if df <= 0:
        raise ValueError("df must be positive")
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def norm_cdf(x: float) -> float:
    return _N.cdf(x)


def t_ppf(q: float, df: float) -> float:
    """The ``q`` quantile of Student's t with ``df`` degrees of freedom (bisection on ``t_cdf``)."""
    if not 0 < q < 1:
        raise ValueError("q must be inside (0, 1)")
    if q == 0.5:
        return 0.0
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > q:
        lo *= 2
    while t_cdf(hi, df) < q:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return (lo + hi) / 2


# --- paired tests --------------------------------------------------------------------------------------------------

def daily(d: Any, days: Sequence[Any]) -> Tuple[List[Any], np.ndarray, np.ndarray]:
    """(sorted days, the sum of ``d`` per day, the count per day)."""
    d = np.asarray(d, dtype=float)
    keys = sorted(set(days))
    index = {key: i for i, key in enumerate(keys)}
    pos = np.fromiter((index[day] for day in days), dtype=int, count=len(days))
    sums = np.bincount(pos, weights=d, minlength=len(keys))
    counts = np.bincount(pos, minlength=len(keys)).astype(float)
    return keys, sums, counts


def newey_west(x: np.ndarray, lags: int) -> float:
    """Long-run variance of a series (Bartlett weights); falls back to the plain variance when not positive."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    gamma0 = float(np.dot(x, x) / n)
    lrv = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        lrv += 2 * (1 - lag / (lags + 1)) * float(np.dot(x[lag:], x[:-lag]) / n)
    return lrv if lrv > 0 else gamma0


def diebold_mariano(d: Any, days: Sequence[Any], lags: int = 2, h: int = 1) -> Dict[str, Any]:
    """Diebold-Mariano on daily sums of per-game score differences ``d`` (challenger minus champion; negative is
    better), Newey-West variance with ``lags`` days, the Harvey-Leybourne-Newbold factor
    sqrt((T + 1 - 2h + h(h-1)/T) / T) and Student t with T - 1 degrees of freedom. ``p`` is one-sided for "the
    challenger is better" (mean below 0); ``p_worse`` for the other side. None-valued with fewer than 3 days."""
    keys, sums, _ = daily(d, days)
    T = len(keys)
    out: Dict[str, Any] = {"days": T, "n": int(np.size(d)), "mean_daily": None, "dm": None, "stat": None, "p": None,
                           "p_worse": None}
    if T < 3:
        return out
    mean = float(np.mean(sums))
    lrv = newey_west(sums, lags)
    if lrv <= 0:
        return out
    dm = mean / math.sqrt(lrv / T)
    stat = dm * math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    out.update({"mean_daily": mean, "dm": dm, "stat": stat, "p": t_cdf(stat, T - 1), "p_worse": 1 - t_cdf(stat, T - 1)})
    return out


def block_bootstrap(d: Any, days: Sequence[Any], reps: int = 2000, seed: int = 0,
                    level: float = 0.95) -> Dict[str, Any]:
    """Mean per-game ``d`` and its percentile range from resampling whole days with replacement."""
    d = np.asarray(d, dtype=float)
    n = int(d.size)
    if n == 0:
        return {"n": 0, "days": 0, "mean": None, "lo": None, "hi": None}
    keys, sums, counts = daily(d, days)
    T = len(keys)
    mean = float(d.mean())
    if T < 2:
        return {"n": n, "days": T, "mean": mean, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, T, size=(reps, T))
    means = sums[idx].sum(axis=1) / np.maximum(counts[idx].sum(axis=1), 1)
    tail = (1 - level) / 2
    lo, hi = np.quantile(means, [tail, 1 - tail])
    return {"n": n, "days": T, "mean": mean, "lo": float(lo), "hi": float(hi)}


CLUSTER_MIN_DAYS = 10  # below this many days a day-clustered SE is too noisy: the games' own SE is used
INTERVAL_MIN_DAYS = 3  # fewer days than this: no range at all


def t_interval(d: Any, days: Sequence[Any], level: float = 0.95, *, min_days: int = INTERVAL_MIN_DAYS
               ) -> Dict[str, Any]:
    """Mean per-game ``d`` and a small-sample range that holds its level with few days (a week's 2-7 game days,
    where a percentile bootstrap over days is far too narrow): mean +- t(level, T - 1) x the day-clustered standard
    error with the T / (T - 1) correction when there are at least CLUSTER_MIN_DAYS days, else mean +- t(level, n - 1)
    x the games' standard error; no range with fewer than ``min_days`` days. Same keys as ``block_bootstrap``."""
    d = np.asarray(d, dtype=float)
    n = int(d.size)
    if n == 0:
        return {"n": 0, "days": 0, "mean": None, "lo": None, "hi": None}
    keys, sums, counts = daily(d, days)
    T = len(keys)
    mean = float(d.mean())
    out = {"n": n, "days": T, "mean": mean, "lo": None, "hi": None}
    if T < max(min_days, 2) or n < 2:
        return out
    if T >= CLUSTER_MIN_DAYS:
        se = math.sqrt(T / (T - 1) * float(np.sum((sums - counts * mean) ** 2))) / n
        df = T - 1
    else:
        se = float(np.std(d, ddof=1)) / math.sqrt(n)
        df = n - 1
    half = t_ppf(1 - (1 - level) / 2, df) * se
    out.update({"lo": mean - half, "hi": mean + half})
    return out


def holm(pvalues: Sequence[Optional[float]], alpha: float = 0.05) -> Tuple[List[Optional[float]], List[bool]]:
    """Holm's step-down adjustment: (adjusted p-values, rejected) in the input order; None p-values stay None and are
    not counted in the family."""
    present = [(p, i) for i, p in enumerate(pvalues) if p is not None]
    m = len(present)
    adjusted: List[Optional[float]] = [None] * len(pvalues)
    rejected = [False] * len(pvalues)
    running = 0.0
    for rank, (p, i) in enumerate(sorted(present)):
        running = max(running, min(1.0, (m - rank) * float(p)))
        adjusted[i] = running
    for i, value in enumerate(adjusted):
        rejected[i] = value is not None and value <= alpha
    return adjusted, rejected


def cusum(z: Sequence[float], k: float = 0.25, h: float = 5.0) -> Dict[str, Any]:
    """One-sided upper CUSUM S_t = max(0, S_{t-1} + z_t - k) of standardised values ``z``; fires when S_t > h.
    Returns {"fired", "at" (index or None), "max", "last", "path"}."""
    s, top, at, path = 0.0, 0.0, None, []
    for t, value in enumerate(z):
        s = max(0.0, s + float(value) - k)
        path.append(s)
        top = max(top, s)
        if at is None and s > h:
            at = t
    return {"fired": at is not None, "at": at, "max": top, "last": s, "path": path}


def daily_z(d: Any, days: Sequence[Any], sd: Optional[float] = None) -> List[float]:
    """Each day's mean ``d`` standardised by sd / sqrt(games that day) (``sd`` default: the SD of all ``d``)."""
    d = np.asarray(d, dtype=float)
    if d.size < 2:
        return []
    scale = float(np.std(d, ddof=1)) if sd is None else float(sd)
    if not scale > 0:
        return []
    _, sums, counts = daily(d, days)
    return [float((s / c) / (scale / math.sqrt(c))) for s, c in zip(sums, counts)]


# --- calibration ---------------------------------------------------------------------------------------------------

def pav(y: Any, w: Optional[Any] = None) -> np.ndarray:
    """Isotonic (non-decreasing) least-squares fit of ``y`` in the given order: pool-adjacent-violators."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    values: List[float] = []
    weights: List[float] = []
    sizes: List[int] = []
    for yi, wi in zip(y, w):
        values.append(float(yi))
        weights.append(float(wi))
        sizes.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            wt = weights[-2] + weights[-1]
            v = (values[-2] * weights[-2] + values[-1] * weights[-1]) / wt
            n = sizes[-2] + sizes[-1]
            values[-2:], weights[-2:], sizes[-2:] = [v], [wt], [n]
    return np.repeat(values, sizes)


def reliability(p: Any, y: Any, points: int = 21, reps: int = 200, seed: int = 0) -> Dict[str, Any]:
    """CORP reliability (Dimitriadis, Gneiting & Jordan 2021): the isotonic recalibration of outcomes ``y`` on the
    forecasts ``p``, read at ``points`` evenly spaced quantiles of ``p``, with a 90% consistency band from ``reps``
    resamples of a perfectly calibrated forecaster (y* ~ Bernoulli(p)). Also the CORP Brier decomposition:
    miscalibration (MCB), discrimination (DSC) and uncertainty (UNC), mean Brier = MCB - DSC + UNC."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(p.size)
    if n < 10:
        return {"n": n, "x": [], "y": [], "lo": [], "hi": [], "mcb": None, "dsc": None, "unc": None}
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    cep = pav(ys)
    grid = np.quantile(ps, np.linspace(0, 1, points))
    at = np.clip(np.searchsorted(ps, grid, side="right") - 1, 0, n - 1)
    rng = np.random.default_rng(seed)
    sims = np.empty((reps, points))
    for r in range(reps):
        fake = (rng.random(n) < ps).astype(float)
        sims[r] = pav(fake)[at]
    lo, hi = np.quantile(sims, [0.05, 0.95], axis=0)
    clim = float(ys.mean())
    score = lambda q: float(np.mean((q - ys) ** 2))  # noqa: E731
    s_p, s_c, s_r = score(ps), score(cep), score(np.full(n, clim))
    return {"n": n, "x": grid.tolist(), "y": cep[at].tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
            "mcb": s_p - s_c, "dsc": s_r - s_c, "unc": s_r}


def _logistic_fit(x: np.ndarray, y: np.ndarray, offset: np.ndarray, iters: int = 50
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Newton's method for logit P(y) = offset + x @ beta; (beta, covariance)."""
    beta = np.zeros(x.shape[1])
    for _ in range(iters):
        eta = np.clip(offset + x @ beta, -35, 35)
        mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-12)
        grad = x.T @ (y - mu)
        hess = (x * w[:, None]).T @ x
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = np.clip(offset + x @ beta, -35, 35)
    mu = 1 / (1 + np.exp(-eta))
    hess = (x * np.maximum(mu * (1 - mu), 1e-12)[:, None]).T @ x
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.full((x.shape[1], x.shape[1]), np.nan)
    return beta, cov


def calibration(p: Any, y: Any) -> Dict[str, Optional[float]]:
    """Calibration slope b from logit P(y) = a + b logit(p) (b < 1: too confident, b > 1: too timid) and the
    calibration-in-the-large intercept (a with b fixed at 1: above 0 means events happen more often than said), each
    with a 95% range from the inverse Hessian. Nones with fewer than 30 games or no events of either kind."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    empty = {"n": int(p.size), "slope": None, "slope_lo": None, "slope_hi": None, "intercept": None,
             "intercept_lo": None, "intercept_hi": None}
    if p.size < 30 or y.min() == y.max():
        return empty
    lp = logit(p)
    beta, cov = _logistic_fit(np.column_stack([np.ones_like(lp), lp]), y, np.zeros_like(lp))
    b, se_b = float(beta[1]), math.sqrt(max(float(cov[1, 1]), 0.0))
    alpha, cov_a = _logistic_fit(np.ones((lp.size, 1)), y, lp)
    a, se_a = float(alpha[0]), math.sqrt(max(float(cov_a[0, 0]), 0.0))
    if not (math.isfinite(b) and math.isfinite(se_b) and math.isfinite(a) and math.isfinite(se_a)):
        return empty
    return {"n": int(p.size), "slope": b, "slope_lo": b - Z95 * se_b, "slope_hi": b + Z95 * se_b, "intercept": a,
            "intercept_lo": a - Z95 * se_a, "intercept_hi": a + Z95 * se_a}


# --- fitting -------------------------------------------------------------------------------------------------------

def numeric_hessian(f: Callable[[np.ndarray], float], x: np.ndarray, step: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    h = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = step
        h[i, i] = (f(x + e_i) - 2 * f0 + f(x - e_i)) / step ** 2
        for j in range(i + 1, n):
            e_j = np.zeros(n)
            e_j[j] = step
            value = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * step ** 2)
            h[i, j] = h[j, i] = value
    return h


def numeric_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, step: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.zeros(x.size)
    for i in range(x.size):
        e = np.zeros(x.size)
        e[i] = step
        g[i] = (f(x + e) - f(x - e)) / (2 * step)
    return g


def newton(f: Callable[[np.ndarray], float], x0: Any, grad: Optional[Callable[[np.ndarray], np.ndarray]] = None,
           iters: int = 30, tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Minimise a smooth ``f`` from ``x0``: damped Newton (Levenberg-style damping when the Hessian isn't positive
    definite or a step doesn't improve), numeric derivatives unless ``grad`` is given. Returns (x, Hessian at x,
    converged)."""
    x = np.asarray(x0, dtype=float).copy()
    fx = f(x)
    damping = 0.0
    converged = False
    for _ in range(iters):
        g = grad(x) if grad is not None else numeric_gradient(f, x)
        hess = numeric_hessian(f, x)
        improved = False
        step = np.zeros(x.size)
        for _ in range(30):
            try:
                step = np.linalg.solve(hess + damping * np.eye(x.size), -g)
            except np.linalg.LinAlgError:
                damping = max(damping * 10, 1e-6)
                continue
            candidate = x + step
            fc = f(candidate)
            if np.isfinite(fc) and fc <= fx + 1e-12:
                x, fx, improved = candidate, fc, True
                damping = damping / 10 if damping > 1e-9 else 0.0
                break
            damping = max(damping * 10, 1e-6)
        if not improved or float(np.max(np.abs(step))) < tol:
            converged = improved or float(np.max(np.abs(g))) < 1e-6
            break
    return x, numeric_hessian(f, x), converged


def standard_errors(hessian: np.ndarray) -> np.ndarray:
    """Standard errors from the Hessian of a negative log likelihood (inf where it isn't invertible)."""
    try:
        cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return np.full(hessian.shape[0], np.inf)
    diag = np.diag(cov)
    return np.where(diag > 0, np.sqrt(np.abs(diag)), np.inf)


def lcb_step(theta0: Any, theta_hat: Any, se: Any, z: float = SURE) -> np.ndarray:
    """theta0 + sign(theta_hat - theta0) x max(|theta_hat - theta0| - z x se, 0): only the 95%-sure part of a move."""
    theta0, theta_hat, se = (np.asarray(v, dtype=float) for v in (theta0, theta_hat, se))
    move = theta_hat - theta0
    sure = np.maximum(np.abs(move) - z * np.where(np.isfinite(se), se, np.inf), 0.0)
    return theta0 + np.sign(move) * np.where(np.isfinite(sure), sure, 0.0)


def golden(f: Callable[[float], float], low: float, high: float, tol: float = 1e-4, iters: int = 60) -> float:
    """The minimiser of a unimodal ``f`` on [low, high] (golden-section search)."""
    ratio = (math.sqrt(5) - 1) / 2
    a, b = low, high
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = f(d)
    return (a + b) / 2
