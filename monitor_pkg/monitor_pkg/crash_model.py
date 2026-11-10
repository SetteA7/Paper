"""Crash probability model (thesis sections 5.6 and 5.7). Pure math, no ROS.

Notation (times in ms):
  T       control period (= LiDAR period)
  psi     measured wait from the scan reaching the controller to the control tick
          that sends the command. Constant while the nodes run (see controller.py),
          so it is a known shift of the delay, not a random variable.
  B       time budget = dt - tau_e2e - psi  (dt = time the bot has before it must be stopped)
  A       number of failed transmissions (data packet + brake command)
  f(k)    P(A = k)          F(k)  P(A >= k)
  phi     wait from the threshold crossing to the next LiDAR sample, ~ U(0, T)

A crash happens when  A*T + phi > B.
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
    """Exact crash probability for budget B = dt - tau_e2e - psi (ms), given psi.

    P = F(n) - f(n) * R/T  with n = floor(B/T), R = B - n*T.
    B < 0 (dt < tau_e2e + psi): crash is certain.
    """
    if B < 0:
        return 1.0
    n = math.floor(B / T)
    R = B - n * T
    p = F(n, p_tx, p_rx) - f(n, p_tx, p_rx) * R / T
    return min(max(p, 0.0), 1.0)


def n_star(p_tx: float, p_rx: float, p_target: float, j_max: int) -> int | None:
    """First N >= 1 with F(N) <= p_target, or None if not reached within j_max."""
    if F(j_max, p_tx, p_rx) > p_target:
        return None
    # F decreases with N: exponential bounding, then binary search
    hi = 1
    while hi < j_max and F(hi, p_tx, p_rx) > p_target:
        hi *= 2
    hi = min(hi, j_max)
    lo = hi // 2 + 1 if hi > 1 else 1   # F(hi // 2) > p_target (or hi == 1)
    while lo < hi:
        mid = (lo + hi) // 2
        if F(mid, p_tx, p_rx) <= p_target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def b_min(T: float, p_tx: float, p_rx: float, p_target: float, j_max: int) -> tuple[float, bool]:
    """Smallest budget B (ms) with p_crash(B) <= p_target.

    P_crash is linear in B between whole periods, so with n = N* - 1:
        B_min = n*T + T * (F(n) - p_target) / f(n)
    Returns (B, reached). If p_target cannot be met within j_max periods
    (link almost dead), returns (j_max * T, False).
    """
    N = n_star(p_tx, p_rx, p_target, j_max)
    if N is None:
        return j_max * T, False
    n = N - 1
    fn = f(n, p_tx, p_rx)
    if fn <= 0:                         # p_tx = p_rx = 1: excluded by n_star, kept for safety
        return N * T, True
    R = T * (F(n, p_tx, p_rx) - p_target) / fn
    return n * T + min(max(R, 0.0), T), True


def threshold_from_budget(B: float, tau_e2e: float, psi: float, vmax: float,
                          fixed_cm: float) -> float:
    """S(B) = vmax/1000 * (B + tau_e2e + psi) + L + m + d_brake  (fixed_cm = L + m + d_brake)."""
    return vmax / 1000.0 * (B + tau_e2e + psi) + fixed_cm


def budget_from_threshold(S: float, tau_e2e: float, psi: float, vmax: float,
                          fixed_cm: float) -> float:
    """Inverse of threshold_from_budget: B = dt - tau_e2e - psi."""
    return (S - fixed_cm) * 1000.0 / vmax - tau_e2e - psi
