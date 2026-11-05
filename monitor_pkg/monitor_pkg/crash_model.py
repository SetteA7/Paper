"""Crash probability model (thesis sections 4.6 and 4.7). Pure math, no ROS.

Notation (times in ms):
  T       control period
  B       time budget = dt - tau_e2e  (dt = time the bot has before it must be stopped)
  A       number of failed transmissions (data packet + brake command)
  f(k)    P(A = k)          F(k)  P(A >= k)
  W       total wait = phi + psi, phi, psi ~ U(0, T) independent (triangular on [0, 2T])

A crash happens when  A*T + W > B.
"""
import math

EQ_TOL = 1e-6   # |p_tx - p_rx| below this: use the p_tx = p_rx closed forms


def F(k: int, p_tx: float, p_rx: float) -> float:
    """P(A >= k). F(k) = 1 for k <= 0."""
    if k <= 0:
        return 1.0
    if abs(p_tx - p_rx) < EQ_TOL:
        p = 0.5 * (p_tx + p_rx)
        return p ** k * (1.0 + k * (1.0 - p))
    return ((1.0 - p_tx) * p_rx ** (k + 1) - (1.0 - p_rx) * p_tx ** (k + 1)) / (p_rx - p_tx)


def f(k: int, p_tx: float, p_rx: float) -> float:
    """P(A = k). f(k) = 0 for k < 0."""
    if k < 0:
        return 0.0
    if abs(p_tx - p_rx) < EQ_TOL:
        p = 0.5 * (p_tx + p_rx)
        return (k + 1) * p ** k * (1.0 - p) ** 2
    return ((1.0 - p_tx) * (1.0 - p_rx)
            * (p_rx ** (k + 1) - p_tx ** (k + 1)) / (p_rx - p_tx))


def p_crash(B: float, T: float, p_tx: float, p_rx: float) -> float:
    """Exact crash probability for budget B (ms). B < 0 (dt < tau_e2e): crash is certain."""
    if B < 0:
        return 1.0
    n = math.floor(B / T)
    R = B - n * T
    p = (F(n + 1, p_tx, p_rx)
         + f(n, p_tx, p_rx) * (1.0 - R * R / (2 * T * T))
         + f(n - 1, p_tx, p_rx) * (T - R) ** 2 / (2 * T * T))
    return min(max(p, 0.0), 1.0)


def _P_n(n: int, p_tx: float, p_rx: float) -> float:
    """Crash probability at budget B = n*T."""
    return F(n, p_tx, p_rx) + 0.5 * f(n - 1, p_tx, p_rx)


def b_min(T: float, p_tx: float, p_rx: float, p_target: float, j_max: int) -> tuple[float, bool]:
    """Smallest budget B (ms) with p_crash(B) <= p_target.

    Returns (B, reached). If p_target cannot be met within j_max periods
    (link almost dead), returns (j_max * T, False).
    """
    # j = smallest integer >= 1 with P_j <= p_target. P_j decreases with j:
    # exponential bounding, then binary search.
    if _P_n(j_max, p_tx, p_rx) > p_target:
        return j_max * T, False
    hi = 1
    while hi < j_max and _P_n(hi, p_tx, p_rx) > p_target:
        hi *= 2
    hi = min(hi, j_max)
    lo = hi // 2 + 1 if hi > 1 else 1   # P_(hi//2) > target (or hi == 1)
    while lo < hi:
        mid = (lo + hi) // 2
        if _P_n(mid, p_tx, p_rx) <= p_target:
            hi = mid
        else:
            lo = mid + 1
    j = lo

    # Inside [nT, (n+1)T): P = P_n - f(n-1) x + gamma x^2 / 2, with x = R/T
    n = j - 1
    f_prev = f(n - 1, p_tx, p_rx)
    gamma = f_prev - f(n, p_tx, p_rx)
    D = _P_n(n, p_tx, p_rx) - p_target
    if D <= 0:                          # only if n == 0 and p_target >= 1
        return n * T, True
    disc = max(f_prev * f_prev - 2.0 * gamma * D, 0.0)
    den = f_prev + math.sqrt(disc)
    x = 2.0 * D / den if den > 0 else 1.0
    return n * T + min(max(x, 0.0), 1.0) * T, True


def threshold_from_budget(B: float, tau_e2e: float, vmax: float, fixed_cm: float) -> float:
    """S(B) = vmax/1000 * (B + tau_e2e) + L + m + d_brake  (fixed_cm = L + m + d_brake)."""
    return vmax / 1000.0 * (B + tau_e2e) + fixed_cm


def budget_from_threshold(S: float, tau_e2e: float, vmax: float, fixed_cm: float) -> float:
    """Inverse of threshold_from_budget."""
    return (S - fixed_cm) * 1000.0 / vmax - tau_e2e
