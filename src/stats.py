"""Small shared statistics helpers (Wilson CI, interval overlap).

Factored out of src/optimization.py so src/simulation.py can also use
them (for per-threshold failure-probability CIs) without a circular
import between the two modules.
"""

import math

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson_confidence_interval(p_hat, n, z=Z_95):
    """Return (low, high), the 95% Wilson score interval for a Bernoulli
    proportion p_hat estimated from n trials. More reliable than the
    normal (Wald) approximation when p_hat is close to 0 or 1, or n is
    small - both of which happen here."""
    if n <= 0:
        return (0.0, 1.0)
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    half_width = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def intervals_overlap(a, b):
    """Return True if closed intervals a=(low, high) and b=(low, high) overlap."""
    return a[0] <= b[1] and b[0] <= a[1]
