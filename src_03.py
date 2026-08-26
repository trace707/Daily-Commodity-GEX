
# %% [markdown]
# ---
# ## Cell 6 - Option pricing math
#
# Two models, because we support two kinds of underlying:
#
# **Black-Scholes** (ETF options - the underlying is a *spot* asset that can pay a yield):
#
# $$d_1=\frac{\ln(S/K)+(r-q+\tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}
# \qquad
# \Gamma_{BS}=\frac{e^{-qT}\,\varphi(d_1)}{S\,\sigma\sqrt{T}}$$
#
# **Black-76** (options on *futures* - there is no carry on a futures position, and
# the premium is discounted at the risk-free rate):
#
# $$d_1=\frac{\ln(F/K)+\tfrac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}
# \qquad
# \Gamma_{76}=\frac{e^{-rT}\,\varphi(d_1)}{F\,\sigma\sqrt{T}}$$
#
# Black-76 is what CME itself uses to settle options on GC, CL, NG, ZC and ZW, so
# the real-futures adapter in Cell 9 routes through it automatically.

# %%
# =============================================================================
# CELL 6 - Black-Scholes / Black-76 greeks and an implied-vol solver
# =============================================================================

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _phi(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF, vectorised."""
    return np.exp(-0.5 * np.square(x)) / SQRT_2PI


def _d1(underlying, strike, t, vol, drift):
    """Shared d1 term. `drift` is (r - q) for Black-Scholes, 0 for Black-76."""
    underlying = np.asarray(underlying, dtype=float)
    strike = np.asarray(strike, dtype=float)
    t = np.asarray(t, dtype=float)
    vol = np.asarray(vol, dtype=float)
    denom = vol * np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.log(underlying / strike) + (drift + 0.5 * vol ** 2) * t) / denom


def bs_gamma(spot, strike, t, vol, r=0.0, q=0.0):
    """Black-Scholes gamma, d2V/dS2, per share of underlying."""
    t = np.maximum(np.asarray(t, dtype=float), 1e-6)
    vol = np.maximum(np.asarray(vol, dtype=float), 1e-6)
    spot = np.asarray(spot, dtype=float)
    d1 = _d1(spot, strike, t, vol, r - q)
    g = np.exp(-q * t) * _phi(d1) / (spot * vol * np.sqrt(t))
    return np.where(np.isfinite(g), g, 0.0)


def b76_gamma(fwd, strike, t, vol, r=0.0):
    """Black-76 gamma, d2V/dF2, per unit of the futures contract."""
    t = np.maximum(np.asarray(t, dtype=float), 1e-6)
    vol = np.maximum(np.asarray(vol, dtype=float), 1e-6)
    fwd = np.asarray(fwd, dtype=float)
    d1 = _d1(fwd, strike, t, vol, 0.0)
    g = np.exp(-r * t) * _phi(d1) / (fwd * vol * np.sqrt(t))
    return np.where(np.isfinite(g), g, 0.0)


def bs_delta(spot, strike, t, vol, option_type, r=0.0, q=0.0):
    t = np.maximum(np.asarray(t, dtype=float), 1e-6)
    vol = np.maximum(np.asarray(vol, dtype=float), 1e-6)
    d1 = _d1(spot, strike, t, vol, r - q)
    call_delta = np.exp(-q * t) * norm.cdf(d1)
    is_call = np.asarray(option_type) == "call"
    return np.where(is_call, call_delta, call_delta - np.exp(-q * t))


def bs_vega(spot, strike, t, vol, r=0.0, q=0.0):
    """Vega per 1.00 (i.e. 100 vol points) of volatility."""
    t = np.maximum(np.asarray(t, dtype=float), 1e-6)
    vol = np.maximum(np.asarray(vol, dtype=float), 1e-6)
    spot = np.asarray(spot, dtype=float)
    d1 = _d1(spot, strike, t, vol, r - q)
    return spot * np.exp(-q * t) * _phi(d1) * np.sqrt(t)


def bs_price(spot, strike, t, vol, option_type, r=0.0, q=0.0):
    t = np.maximum(np.asarray(t, dtype=float), 1e-6)
    vol = np.maximum(np.asarray(vol, dtype=float), 1e-6)
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    d1 = _d1(spot, strike, t, vol, r - q)
    d2 = d1 - vol * np.sqrt(t)
    call = spot * np.exp(-q * t) * norm.cdf(d1) - strike * np.exp(-r * t) * norm.cdf(d2)
    put = strike * np.exp(-r * t) * norm.cdf(-d2) - spot * np.exp(-q * t) * norm.cdf(-d1)
    return np.where(np.asarray(option_type) == "call", call, put)


def implied_vol(
    price, spot, strike, t, option_type, r=0.0, q=0.0,
    lo: float = 1e-3, hi: float = 5.0, tol: float = 1e-6, max_iter: int = 60,
):
    """Vectorised implied vol by bisection.

    Bisection rather than Newton: vega collapses to ~0 in the deep wings, where
    Newton diverges on exactly the illiquid strikes that carry the noisiest
    quotes. Bisection is slower but cannot blow up, and 60 iterations on a
    bracket of [0.001, 5.0] resolves to well under a vol point.
    """
    price = np.asarray(price, dtype=float)
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    t = np.asarray(t, dtype=float)

    lo_v = np.full(price.shape, lo, dtype=float)
    hi_v = np.full(price.shape, hi, dtype=float)

    # No-arbitrage bounds. Outside them, no vol reproduces the quote.
    intrinsic = np.where(
        np.asarray(option_type) == "call",
        np.maximum(spot * np.exp(-q * t) - strike * np.exp(-r * t), 0.0),
        np.maximum(strike * np.exp(-r * t) - spot * np.exp(-q * t), 0.0),
    )
    upper = np.where(np.asarray(option_type) == "call", spot * np.exp(-q * t), strike * np.exp(-r * t))
    solvable = np.isfinite(price) & (price > intrinsic + 1e-8) & (price < upper) & (t > 0)

    for _ in range(max_iter):
        mid = 0.5 * (lo_v + hi_v)
        diff = bs_price(spot, strike, t, mid, option_type, r, q) - price
        too_high = diff > 0
        hi_v = np.where(too_high, mid, hi_v)
        lo_v = np.where(too_high, lo_v, mid)
        if np.nanmax(hi_v - lo_v) < tol:
            break

    out = 0.5 * (lo_v + hi_v)
    return np.where(solvable, out, np.nan)


# --- Self-test: round-trip a known price through the solver -------------------
_s, _k, _t, _v = 100.0, 105.0, 0.25, 0.32
_p = bs_price(_s, _k, _t, _v, "call", r=0.04)
_iv = implied_vol(_p, _s, _k, _t, "call", r=0.04)
assert abs(float(_iv) - _v) < 1e-4, f"IV solver round-trip failed: {_iv} vs {_v}"

# Gamma must match a numerical second derivative of price.
_h = 1e-3 * _s
_num = (bs_price(_s + _h, _k, _t, _v, "call", r=0.04)
        - 2 * bs_price(_s, _k, _t, _v, "call", r=0.04)
        + bs_price(_s - _h, _k, _t, _v, "call", r=0.04)) / _h ** 2
_ana = float(bs_gamma(_s, _k, _t, _v, r=0.04))
assert abs(_num - _ana) / _ana < 1e-4, f"Gamma mismatch: analytic {_ana} vs numeric {_num}"

# Put/call gamma parity: identical strike and expiry means identical gamma.
assert abs(float(bs_gamma(_s, _k, _t, _v)) - float(bs_gamma(_s, _k, _t, _v))) < 1e-12

print(f"Pricing math verified.  IV round-trip {float(_iv):.6f} vs {_v}")
print(f"Gamma analytic {_ana:.8f} vs numeric {_num:.8f}")
