# %% [markdown]
# # Commodity Futures GEX Dashboard
#
# Gamma Exposure (GEX) for **Gold, Silver, Copper, Crude Oil, Natural Gas, Gasoline,
# Corn, Wheat, Soybeans** and other commodities — with dealer gamma profiles, gamma
# flip levels, and call/put walls **quoted in futures price terms** (e.g. `GC` dollars/oz,
# `ZC` cents/bushel).
#
# ---
#
# ## Read this before you trade off it
#
# **1. Where the options data comes from.** CME does not expose a free, stable public API
# for futures-options open interest — its endpoints sit behind bot protection and reject
# datacenter IPs (which is what Colab runs on). This notebook therefore computes GEX from
# the **listed options on the commodity ETFs** (`GLD`, `USO`, `UNG`, `CORN`, `WEAT`, ...),
# which are free, deep enough to be meaningful, and update every trading day.
#
# **2. What the numbers mean.** The **dollar magnitudes are real ETF-option dealer gamma**.
# The **price levels are translated into futures terms** using a return-beta map anchored at
# today's spot, so a `GLD` 430 strike shows up as its `GC` equivalent. Fit quality
# (return correlation) is printed for every commodity — trust the levels where it's high
# (gold, wheat, copper ≈ 0.90–0.95) and treat natural gas (≈ 0.63) as indicative only.
#
# **3. If you have real futures-options data**, `Cell 9` is a drop-in adapter: upload a CSV
# of CME/Barchart option OI and the entire dashboard recomputes on true `GC`/`CL`/`ZC`
# chains using **Black-76** instead of the ETF proxy. Nothing else changes.
#
# **4. The dealer sign convention is an assumption**, not an observation. The standard
# "dealers are long calls / short puts" convention is the default and is the single largest
# source of error in *all* published GEX work, including this notebook. It is configurable.
#
# ---
#
# ## What each cell does
#
# | Cell | Purpose |
# |---|---|
# | 2–4 | Install, config, commodity universe |
# | 5 | Yahoo data client (crumb auth, retries, caching) |
# | 6 | Black-Scholes + Black-76 greeks, implied-vol solver |
# | 7 | **The GEX calculation** — dollar gamma, net GEX, gamma flip, walls |
# | 8 | ETF → futures price mapping (return beta) |
# | 9 | Real futures-options adapter (CSV upload) |
# | 10–11 | Charts and dashboard assembly |
# | 12 | **RUN THIS** — builds the whole dashboard |
# | 13–14 | Daily snapshot persistence + history charts |
# | 15 | Scheduling it to run every day |

# %%
# =============================================================================
# CELL 2 — Install
# =============================================================================
# `import importlib` alone does NOT bind the `util` submodule - it has to be
# imported explicitly. This works either way on Windows, where truststore drags
# importlib.util in as a side effect, but truststore is a win32-only requirement,
# so on Linux and in CI the bare import raises AttributeError here.
import importlib.util
import subprocess
import sys

_need = []
for _mod, _pip in [("plotly", "plotly>=5.20"), ("scipy", "scipy"), ("pandas", "pandas"), ("numpy", "numpy")]:
    if importlib.util.find_spec(_mod) is None:
        _need.append(_pip)

if _need:
    print("Installing:", ", ".join(_need))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_need], check=True)
    print("Done. If imports fail below, Runtime > Restart session, then re-run.")
else:
    print("All dependencies already present.")

# %%
# =============================================================================
# CELL 3 — Imports & global configuration
# =============================================================================
import datetime as dt
import gzip
import http.cookiejar
import io
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.stats import norm

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

try:  # Colab renders plotly natively; this is a no-op elsewhere.
    import google.colab  # noqa: F401

    IN_COLAB = True
    pio.renderers.default = "colab"
except Exception:
    IN_COLAB = False
    pio.renderers.default = "notebook_connected"


@dataclass
class Config:
    """Every knob in the dashboard. Edit here, re-run Cell 12."""

    # --- Dealer positioning assumption -------------------------------------
    # "long_calls_short_puts" : market-maker is long calls / short puts (standard).
    # "short_calls_long_puts" : the mirror image.
    # "all_short"             : dealer short every contract (max-negative-gamma view).
    dealer_convention: str = "long_calls_short_puts"

    # --- Chain filters ------------------------------------------------------
    min_open_interest: int = 1        # drop strikes with less OI than this
    max_dte: int = 180                # ignore expiries beyond N calendar days
    min_dte: int = 0                  # 0 keeps same-day expiries (0DTE gamma is huge)
    moneyness_band: float = 0.45      # keep strikes within +/-45% of spot
    max_expiries: int = 8             # cap expiries pulled per commodity (speed)

    # --- Implied volatility handling ---------------------------------------
    recompute_iv: bool = True         # solve IV from mid price (Yahoo's IV is unreliable)
    iv_floor: float = 0.01
    iv_cap: float = 5.00

    # --- Rates --------------------------------------------------------------
    risk_free_rate: float = 0.0425    # annualised, used for discounting / Black-76

    # --- Gamma profile ------------------------------------------------------
    profile_points: int = 161         # grid resolution for the gamma-vs-spot curve
    profile_span: float = 0.25        # profile runs spot * (1 -/+ span)

    # --- ETF -> futures mapping --------------------------------------------
    beta_lookback_days: int = 252     # window for the return-beta regression
    beta_min_corr: float = 0.50       # below this, flag the mapping as unreliable

    # --- Networking ---------------------------------------------------------
    request_timeout: int = 30
    max_retries: int = 4
    retry_backoff: float = 1.7
    cache_ttl_seconds: int = 300      # in-memory cache; 0 disables

    # --- Persistence --------------------------------------------------------
    save_snapshots: bool = True
    snapshot_dir: str = "/content/drive/MyDrive/commodity_gex"  # Colab+Drive default

    # --- Display ------------------------------------------------------------
    mode: str = "dark"                # "dark" | "light" - each is stepped for its
                                      # own surface, not an inversion of the other
    scale_label: str = "auto"         # "auto" | "millions" | "billions"


CFG = Config()

# `display` exists in Jupyter/Colab but not when this file is run as a plain script
# (which is how the test suite and the GitHub Action execute it).
try:
    display  # noqa: B018
except NameError:  # pragma: no cover
    display = print  # type: ignore[assignment]

print(f"Config loaded. Colab={IN_COLAB}. Dealer convention: {CFG.dealer_convention}")


# %%
# =============================================================================
# CELL 4 - The commodity universe
# =============================================================================
# Each entry ties together three things:
#   * the CME futures contract we want to quote levels in,
#   * the ETF whose listed options we actually read gamma from,
#   * the contract economics needed to convert gamma into dollars.
#
# `point_value` = dollar value of a 1.00 move in the *quoted* price unit.
#   Gold  is quoted in $/oz,      contract = 100 oz    -> $100 per $1.00
#   Corn  is quoted in cents/bu,  contract = 5,000 bu  -> $50  per 1.00 cent
# =============================================================================


@dataclass(frozen=True)
class Commodity:
    key: str                 # short internal id
    name: str                # display name
    futures_symbol: str      # Yahoo continuous-futures symbol
    cme_root: str            # CME product root (for the real-futures adapter)
    etf: str                 # options proxy ETF
    point_value: float       # $ per 1.00 of quoted price, per futures contract
    price_unit: str          # how the futures price is quoted
    sector: str
    contract_size: str

    @property
    def label(self) -> str:
        return f"{self.name} ({self.cme_root})"


UNIVERSE: dict[str, Commodity] = {
    c.key: c
    for c in [
        # --- Precious & base metals ----------------------------------------
        Commodity("gold",     "Gold",          "GC=F", "GC", "GLD",   100.0, "$/troy oz", "Metals", "100 troy oz"),
        Commodity("silver",   "Silver",        "SI=F", "SI", "SLV",  5000.0, "$/troy oz", "Metals", "5,000 troy oz"),
        Commodity("copper",   "Copper",        "HG=F", "HG", "CPER", 25000.0, "$/lb",     "Metals", "25,000 lb"),
        Commodity("platinum", "Platinum",      "PL=F", "PL", "PPLT",   50.0, "$/troy oz", "Metals", "50 troy oz"),
        # --- Energy ---------------------------------------------------------
        Commodity("crude",    "WTI Crude Oil", "CL=F", "CL", "USO",  1000.0, "$/barrel",  "Energy", "1,000 barrels"),
        Commodity("brent",    "Brent Crude",   "BZ=F", "BZ", "BNO",  1000.0, "$/barrel",  "Energy", "1,000 barrels"),
        Commodity("natgas",   "Natural Gas",   "NG=F", "NG", "UNG", 10000.0, "$/MMBtu",   "Energy", "10,000 MMBtu"),
        Commodity("gasoline", "RBOB Gasoline", "RB=F", "RB", "UGA", 42000.0, "$/gallon",  "Energy", "42,000 gallons"),
        # --- Grains & softs --------------------------------------------------
        Commodity("corn",     "Corn",          "ZC=F", "ZC", "CORN",   50.0, "cents/bu",  "Grains", "5,000 bushels"),
        Commodity("wheat",    "Chicago Wheat", "ZW=F", "ZW", "WEAT",   50.0, "cents/bu",  "Grains", "5,000 bushels"),
        Commodity("soybeans", "Soybeans",      "ZS=F", "ZS", "SOYB",   50.0, "cents/bu",  "Grains", "5,000 bushels"),
    ]
}

# Commodities the dashboard runs by default. Add or remove keys freely.
DEFAULT_WATCHLIST = ["gold", "crude", "natgas", "corn", "wheat", "silver", "copper", "soybeans"]

print(f"{len(UNIVERSE)} commodities defined.")
print("Default watchlist:", ", ".join(DEFAULT_WATCHLIST))

pd.DataFrame(
    [
        {
            "key": c.key,
            "commodity": c.name,
            "futures": c.cme_root,
            "ETF": c.etf,
            "contract": c.contract_size,
            "quoted in": c.price_unit,
            "$ per 1.00 move": f"{c.point_value:,.0f}",
        }
        for c in UNIVERSE.values()
    ]
)

# %%
# =============================================================================
# CELL 5 - Market data client (Yahoo Finance)
# =============================================================================
# Yahoo public endpoints need a cookie + crumb pair. We fetch one per session,
# refresh it automatically on a 401, and retry with exponential backoff to ride
# out the rate limits that Colab shared IPs regularly run into.
# =============================================================================

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


class DataError(RuntimeError):
    """Raised when market data cannot be retrieved after retries."""


class YahooClient:
    """Minimal, dependency-free Yahoo Finance client for quotes and option chains."""

    CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"
    CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    OPTS_URL = "https://query2.finance.yahoo.com/v7/finance/options/{sym}"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._jar)
        )
        self._crumb: str | None = None
        self._cache: dict[str, tuple[float, Any]] = {}

    # --------------------------------------------------------------- internals
    def _raw_get(self, url: str, headers: dict | None = None) -> bytes:
        hdrs = {
            "User-Agent": _UA,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if headers:
            hdrs.update(headers)
        req = urllib.request.Request(url, headers=hdrs)
        resp = self._opener.open(req, timeout=self.cfg.request_timeout)
        data = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        return data

    def _ensure_crumb(self, force: bool = False) -> str:
        if self._crumb and not force:
            return self._crumb
        # Priming request seeds the consent cookie. It 404s by design - ignore it.
        for primer in ("https://fc.yahoo.com/", "https://finance.yahoo.com/quote/SPY"):
            try:
                self._raw_get(primer)
                break
            except Exception:
                continue
        self._crumb = self._raw_get(self.CRUMB_URL).decode().strip()
        if not self._crumb or "<" in self._crumb:
            raise DataError("Could not obtain a Yahoo crumb token.")
        return self._crumb

    def _get_json(self, url: str, use_crumb: bool = False) -> dict:
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                target = url
                if use_crumb:
                    crumb = self._ensure_crumb(force=attempt > 0)
                    sep = "&" if "?" in url else "?"
                    target = f"{url}{sep}crumb={urllib.parse.quote(crumb)}"
                return json.loads(self._raw_get(target))
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code in (401, 403):
                    self._crumb = None          # force a fresh crumb next loop
                elif exc.code not in (429, 500, 502, 503, 504):
                    raise DataError(f"HTTP {exc.code} for {url}") from exc
            except Exception as exc:
                last = exc
            time.sleep(self.cfg.retry_backoff ** attempt)
        raise DataError(f"Failed after {self.cfg.max_retries} attempts: {url} ({last})")

    def _cached(self, key: str, producer: Callable[[], Any]) -> Any:
        ttl = self.cfg.cache_ttl_seconds
        if ttl:
            hit = self._cache.get(key)
            if hit and (time.time() - hit[0]) < ttl:
                return hit[1]
        value = producer()
        self._cache[key] = (time.time(), value)
        return value

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------ public
    def quote(self, symbol: str) -> dict:
        """Last price and session metadata for any Yahoo symbol."""

        def _fetch() -> dict:
            url = self.CHART_URL.format(sym=urllib.parse.quote(symbol)) + "?range=5d&interval=1d"
            meta = self._get_json(url)["chart"]["result"][0]["meta"]
            px = float(meta["regularMarketPrice"])
            return {
                "symbol": symbol,
                "price": px,
                "previous_close": float(meta.get("chartPreviousClose") or px),
                "currency": meta.get("currency", "USD"),
                "name": meta.get("shortName", symbol),
                "exchange": meta.get("fullExchangeName", ""),
                "timestamp": dt.datetime.fromtimestamp(
                    meta.get("regularMarketTime", time.time()), dt.timezone.utc
                ),
            }

        return self._cached(f"q:{symbol}", _fetch)

    def daily_history(self, symbol: str, lookback_days: int = 400) -> pd.DataFrame:
        """Daily close and volume indexed by calendar date.

        Equities and Globex futures stamp their bars in different sessions, so we
        key on the UTC calendar date. Joining on raw epoch timestamps silently
        produces an almost-empty intersection - which is exactly the bug that
        makes a naive ETF/futures regression report a bogus R-squared of 1.0 on
        a sample of two points.
        """

        def _fetch() -> pd.DataFrame:
            rng = "2y" if lookback_days > 300 else "1y"
            url = self.CHART_URL.format(sym=urllib.parse.quote(symbol)) + f"?range={rng}&interval=1d"
            res = self._get_json(url)["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            rows = []
            for ts, px, vol in zip(res["timestamp"], q["close"], q.get("volume", [])):
                if px is None:
                    continue
                rows.append(
                    {
                        "date": dt.datetime.fromtimestamp(ts, dt.timezone.utc).date(),
                        "close": float(px),
                        "volume": float(vol or 0.0),
                    }
                )
            if not rows:
                raise DataError(f"No price history for {symbol}")
            return pd.DataFrame(rows).set_index("date").sort_index()

        return self._cached(f"h:{symbol}:{lookback_days}", _fetch)

    def expirations(self, symbol: str) -> list[int]:
        """Every listed expiry for a symbol, as epoch seconds."""

        def _fetch() -> list[int]:
            url = self.OPTS_URL.format(sym=urllib.parse.quote(symbol))
            res = self._get_json(url, use_crumb=True)["optionChain"]["result"]
            if not res:
                raise DataError(f"No option chain listed for {symbol}")
            return list(res[0]["expirationDates"])

        return self._cached(f"e:{symbol}", _fetch)

    def option_chain(self, symbol: str, expiry_epoch: int | None = None) -> tuple[pd.DataFrame, float]:
        """Return (chain dataframe, underlying spot) for one expiry."""

        def _fetch() -> tuple[pd.DataFrame, float]:
            url = self.OPTS_URL.format(sym=urllib.parse.quote(symbol))
            if expiry_epoch is not None:
                url += f"?date={int(expiry_epoch)}"
            res = self._get_json(url, use_crumb=True)["optionChain"]["result"]
            if not res:
                raise DataError(f"Empty option chain for {symbol}")
            node = res[0]
            spot = float(node["quote"]["regularMarketPrice"])
            frames = []
            for block in node.get("options", []):
                for side in ("calls", "puts"):
                    rows = block.get(side, [])
                    if not rows:
                        continue
                    df = pd.DataFrame(rows)
                    df["option_type"] = side[:-1]      # "call" / "put"
                    df["expiry_epoch"] = block["expirationDate"]
                    frames.append(df)
            if not frames:
                raise DataError(f"No contracts returned for {symbol}")
            return pd.concat(frames, ignore_index=True), spot

        return self._cached(f"c:{symbol}:{expiry_epoch}", _fetch)


CLIENT = YahooClient(CFG)


def check_data_source(verbose: bool = True) -> tuple[bool, str]:
    """Confirm the data source is reachable. Never raises.

    This is a diagnostic, not a gate. It used to be a bare module-level call, but
    that made a transient data outage explode during import - before any of the
    error handling downstream could report what actually went wrong. A caller that
    needs the data to be present should check the return value.
    """
    try:
        q = CLIENT.quote("GC=F")
        msg = f"Data source live. {q['name']}: {q['price']:,.2f} ({q['timestamp']:%Y-%m-%d %H:%M UTC})"
        if verbose:
            print(msg)
        return True, msg
    except Exception as exc:
        msg = (
            f"Data source UNREACHABLE - {type(exc).__name__}: {exc}\n"
            "  Yahoo rate-limits datacenter IP ranges, so this is the expected\n"
            "  failure on CI runners and cloud VMs. On a home connection it usually\n"
            "  means a transient outage; retry in a few minutes."
        )
        if verbose:
            print(msg)
        return False, msg


DATA_SOURCE_OK, DATA_SOURCE_MSG = check_data_source()


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


# %% [markdown]
# ---
# ## Cell 7 - The GEX calculation
#
# ### Step 1 - gamma per contract
#
# For every listed strike we take open interest and implied volatility, and compute
# gamma with the model that matches the underlying (Black-Scholes for ETF options,
# Black-76 for futures options).
#
# ### Step 2 - convert gamma into dollars
#
# Gamma is *delta per unit of underlying move*. To express it as the dollar notional
# a dealer must trade to stay hedged, walk the units through:
#
# 1. A move of $\Delta S$ changes each option's delta by $\Gamma\,\Delta S$.
# 2. One contract covers $M$ units of underlying (100 shares for an ETF option;
#    1,000 barrels for `CL`; 5,000 bushels for `ZC`), so the position holds
#    $OI \times M$ units.
# 3. The dollar notional of that delta change is $\Gamma\,\Delta S\,\cdot OI \cdot M \cdot S$.
# 4. Setting $\Delta S = 0.01\,S$ (a 1% move) gives the standard definition:
#
# $$\boxed{\;GEX_{strike} = \Gamma \times OI \times M \times S^{2} \times 0.01\;}$$
#
# Units: **dollars of delta that dealers must buy or sell per 1% move** in the underlying.
#
# ### Step 3 - apply the dealer sign
#
# Under the standard convention dealers are long calls and short puts:
#
# $$NetGEX=\sum_{calls} GEX - \sum_{puts} GEX$$
#
# **Positive net GEX** - dealers are long gamma. Hedging is mean-reverting (they sell
# rallies, buy dips), which *suppresses* realised volatility.
# **Negative net GEX** - dealers are short gamma. Hedging is momentum-amplifying (they
# buy rallies, sell dips), which *feeds* trends and expands realised volatility.
#
# ### Step 4 - the gamma flip
#
# Recompute the whole book's net GEX across a grid of hypothetical spot levels, holding
# open interest, implied vol and time-to-expiry fixed. The level where net GEX crosses
# zero is the **gamma flip** - the boundary between the vol-suppressing and
# vol-amplifying regimes. Holding the vol surface fixed while spot moves is a
# simplification; the flip is a useful map, not a precise trigger.

# %%
# =============================================================================
# CELL 7 - GEX engine
# =============================================================================

CONVENTION_SIGNS = {
    # (call sign, put sign)
    "long_calls_short_puts": (+1.0, -1.0),
    "short_calls_long_puts": (-1.0, +1.0),
    "all_short": (-1.0, -1.0),
    "all_long": (+1.0, +1.0),
}


@dataclass
class GexResult:
    """Everything the dashboard needs for one commodity."""

    commodity: Commodity
    asof: dt.datetime
    source: str                    # "etf_proxy" or "futures"

    underlying_symbol: str
    underlying_spot: float         # price in the space GEX was computed in
    futures_price: float           # price in futures quote terms
    mapping: "PriceMap"

    chain: pd.DataFrame            # cleaned, per-contract
    by_strike: pd.DataFrame        # aggregated per strike
    profile: pd.DataFrame          # net GEX across a spot grid

    net_gex: float
    call_gex: float
    put_gex: float
    total_oi: int
    contracts_used: int

    gamma_flip: float | None       # in underlying space
    gamma_flip_futures: float | None
    call_wall: float | None
    put_wall: float | None
    abs_gamma_peak: float | None
    avg_iv: float

    warnings: list[str] = field(default_factory=list)

    @property
    def regime(self) -> str:
        return "LONG GAMMA (vol suppressed)" if self.net_gex >= 0 else "SHORT GAMMA (vol amplified)"

    def to_futures(self, level: float | None) -> float | None:
        return None if level is None else self.mapping.to_futures(level)


def _clean_chain(raw: pd.DataFrame, spot: float, cfg: Config, now: dt.datetime) -> tuple[pd.DataFrame, list[str]]:
    """Filter junk contracts and repair implied vols."""
    notes: list[str] = []
    df = raw.copy()

    for col in ("strike", "openInterest", "impliedVolatility", "bid", "ask", "lastPrice", "volume"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["openInterest"] = df["openInterest"].fillna(0.0)
    df["volume"] = df["volume"].fillna(0.0)

    df["expiry"] = pd.to_datetime(df["expiry_epoch"], unit="s", utc=True)
    # Options stop trading at the close; 16:00 ET is 20:00 UTC (21:00 in winter).
    df["expiry"] = df["expiry"] + pd.Timedelta(hours=21)
    df["dte"] = (df["expiry"] - pd.Timestamp(now)).dt.total_seconds() / 86400.0
    df["t"] = np.maximum(df["dte"], 0.0) / 365.0

    before = len(df)

    # Strictly greater than zero: an option at or past its expiry stamp has no
    # gamma left. Yahoo keeps returning the just-expired chain for the rest of the
    # session, and a tolerant lower bound (dte >= min_dte - 1) would admit those
    # dead contracts. They contribute no gamma - t collapses, d1 explodes and the
    # normal density underflows to zero - but they still inflate the contract and
    # open-interest counts, which is what makes the book look bigger than it is.
    # Anything expiring later today still has dte > 0 and is kept, so genuine
    # 0DTE gamma survives this filter.
    expired = int((df["dte"] <= 0).sum())
    if expired:
        notes.append(f"Dropped {expired} already-expired contracts still listed in the chain.")

    lower = max(float(cfg.min_dte), 0.0)
    df = df[(df["dte"] > lower) & (df["dte"] <= cfg.max_dte)]
    df = df[df["openInterest"] >= cfg.min_open_interest]
    df = df[df["strike"] > 0]
    lo, hi = spot * (1 - cfg.moneyness_band), spot * (1 + cfg.moneyness_band)
    df = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
    if df.empty:
        return df, [f"All {before} contracts filtered out - loosen the Cell 3 filters."]

    # --- implied vol -------------------------------------------------------
    df["iv_source"] = "quoted"
    df["iv"] = df["impliedVolatility"]

    if cfg.recompute_iv:
        mid = (df["bid"] + df["ask"]) / 2.0
        usable = (df["bid"] > 0) & (df["ask"] > df["bid"]) & mid.notna()
        # A crossed or stale two-sided market is worse than the quoted IV.
        if usable.any():
            solved = implied_vol(
                mid.where(usable, np.nan).to_numpy(),
                spot,
                df["strike"].to_numpy(),
                df["t"].to_numpy(),
                df["option_type"].to_numpy(),
                r=cfg.risk_free_rate,
            )
            good = np.isfinite(solved) & (solved > cfg.iv_floor) & (solved < cfg.iv_cap)
            df.loc[good, "iv"] = solved[good]
            df.loc[good, "iv_source"] = "solved"
            notes.append(f"Re-solved IV from mid on {int(good.sum())}/{len(df)} contracts.")

    # Anything still unusable inherits the expiry's median vol.
    df["iv"] = df["iv"].replace([np.inf, -np.inf], np.nan)
    df.loc[(df["iv"] <= cfg.iv_floor) | (df["iv"] >= cfg.iv_cap), "iv"] = np.nan
    if df["iv"].isna().any():
        med = df.groupby("expiry_epoch")["iv"].transform("median")
        n_fill = int(df["iv"].isna().sum())
        df["iv"] = df["iv"].fillna(med).fillna(df["iv"].median())
        df.loc[df["iv"].isna(), "iv"] = 0.30
        notes.append(f"Backfilled {n_fill} missing/absurd IVs with expiry medians.")

    df["iv"] = df["iv"].clip(cfg.iv_floor, cfg.iv_cap)
    return df.reset_index(drop=True), notes


def _dollar_gamma(spot, strike, t, iv, oi, multiplier, cfg: Config, model: str) -> np.ndarray:
    """GEX per strike, in dollars of delta per 1% move. See the Cell 7 derivation."""
    if model == "black76":
        gamma = b76_gamma(spot, strike, t, iv, r=cfg.risk_free_rate)
    else:
        gamma = bs_gamma(spot, strike, t, iv, r=cfg.risk_free_rate, q=0.0)
    return gamma * oi * multiplier * np.asarray(spot, dtype=float) ** 2 * 0.01


def _gamma_profile(df: pd.DataFrame, spot: float, multiplier: float, cfg: Config, model: str) -> pd.DataFrame:
    """Net GEX re-evaluated across a grid of hypothetical spot levels."""
    grid = np.linspace(spot * (1 - cfg.profile_span), spot * (1 + cfg.profile_span), cfg.profile_points)
    strike = df["strike"].to_numpy()
    t = df["t"].to_numpy()
    iv = df["iv"].to_numpy()
    oi = df["openInterest"].to_numpy()
    sign = df["dealer_sign"].to_numpy()

    net, calls, puts = [], [], []
    is_call = (df["option_type"] == "call").to_numpy()
    for s in grid:
        dg = _dollar_gamma(s, strike, t, iv, oi, multiplier, cfg, model) * sign
        net.append(dg.sum())
        calls.append(dg[is_call].sum())
        puts.append(dg[~is_call].sum())

    return pd.DataFrame({"spot": grid, "net_gex": net, "call_gex": calls, "put_gex": puts})


def _find_flip(profile: pd.DataFrame, spot: float) -> float | None:
    """Zero crossing of the net-gamma curve nearest to spot, linearly interpolated."""
    x = profile["spot"].to_numpy()
    y = profile["net_gex"].to_numpy()
    sign_change = np.where(np.diff(np.sign(y)) != 0)[0]
    if len(sign_change) == 0:
        return None
    roots = []
    for i in sign_change:
        y0, y1 = y[i], y[i + 1]
        if y1 == y0:
            continue
        roots.append(x[i] - y0 * (x[i + 1] - x[i]) / (y1 - y0))
    if not roots:
        return None
    return float(min(roots, key=lambda r: abs(r - spot)))


def compute_gex(
    commodity: Commodity,
    cfg: Config = None,
    client: YahooClient = None,
    chain_override: pd.DataFrame | None = None,
    spot_override: float | None = None,
    model: str = "black_scholes",
    multiplier_override: float | None = None,
    asof: dt.datetime | None = None,
) -> GexResult:
    """Full GEX calculation for one commodity.

    By default this reads the proxy ETF option chain. Pass `chain_override` (plus
    `model="black76"` and a futures `multiplier_override`) to run the identical
    maths on real futures options - that is all Cell 9 does.

    `asof` pins the valuation timestamp. Leave it unset for a live reading; pass
    one explicit timestamp when comparing several runs, because time-to-expiry is
    measured from this instant and a few seconds of wall-clock drift is enough to
    move gamma in the last decimal places. `build_all` shares a single `asof`
    across the whole watchlist so every commodity in a snapshot is priced at the
    same moment.
    """
    cfg = cfg or CFG
    client = client or CLIENT
    now = asof or dt.datetime.now(dt.timezone.utc)
    notes: list[str] = []

    # --- 1. get the chain --------------------------------------------------
    if chain_override is not None:
        raw = chain_override.copy()
        spot = float(spot_override)
        source = "futures"
        underlying_symbol = commodity.cme_root
        multiplier = multiplier_override if multiplier_override is not None else commodity.point_value
    else:
        source = "etf_proxy"
        underlying_symbol = commodity.etf
        multiplier = 100.0                      # US listed equity option = 100 shares
        expiries = client.expirations(commodity.etf)[: cfg.max_expiries]
        frames, spot = [], None
        for exp in expiries:
            try:
                part, spot_i = client.option_chain(commodity.etf, exp)
                frames.append(part)
                spot = spot_i
            except DataError as exc:
                notes.append(f"Skipped expiry {dt.date.fromtimestamp(exp)}: {exc}")
        if not frames or spot is None:
            raise DataError(f"No option data retrieved for {commodity.etf}")
        raw = pd.concat(frames, ignore_index=True)
        raw = raw.drop_duplicates(subset=["contractSymbol"], keep="last")

    # --- 2. clean ----------------------------------------------------------
    df, clean_notes = _clean_chain(raw, spot, cfg, now)
    notes.extend(clean_notes)
    if df.empty:
        raise DataError(f"{commodity.name}: no usable contracts after filtering.")

    # --- 3. dealer sign ----------------------------------------------------
    call_sign, put_sign = CONVENTION_SIGNS[cfg.dealer_convention]
    df["dealer_sign"] = np.where(df["option_type"] == "call", call_sign, put_sign)

    # --- 4. dollar gamma ---------------------------------------------------
    df["gamma"] = (
        b76_gamma(spot, df["strike"], df["t"], df["iv"], r=cfg.risk_free_rate)
        if model == "black76"
        else bs_gamma(spot, df["strike"], df["t"], df["iv"], r=cfg.risk_free_rate)
    )
    df["gex_abs"] = _dollar_gamma(
        spot, df["strike"], df["t"], df["iv"], df["openInterest"], multiplier, cfg, model
    )
    df["gex"] = df["gex_abs"] * df["dealer_sign"]
    df["delta"] = bs_delta(spot, df["strike"], df["t"], df["iv"], df["option_type"], r=cfg.risk_free_rate)

    # --- 5. aggregate ------------------------------------------------------
    is_call = df["option_type"] == "call"
    by_strike = (
        df.assign(
            call_gex=df["gex"].where(is_call, 0.0),
            put_gex=df["gex"].where(~is_call, 0.0),
            call_oi=df["openInterest"].where(is_call, 0.0),
            put_oi=df["openInterest"].where(~is_call, 0.0),
        )
        .groupby("strike", as_index=False)
        .agg(
            net_gex=("gex", "sum"),
            call_gex=("call_gex", "sum"),
            put_gex=("put_gex", "sum"),
            call_oi=("call_oi", "sum"),
            put_oi=("put_oi", "sum"),
            total_oi=("openInterest", "sum"),
            avg_iv=("iv", "mean"),
        )
        .sort_values("strike")
        .reset_index(drop=True)
    )

    # --- 6. structure levels ----------------------------------------------
    profile = _gamma_profile(df, spot, multiplier, cfg, model)
    flip = _find_flip(profile, spot)

    above = by_strike[by_strike["strike"] >= spot]
    below = by_strike[by_strike["strike"] <= spot]
    call_wall = float(above.loc[above["call_gex"].abs().idxmax(), "strike"]) if not above.empty else None
    put_wall = float(below.loc[below["put_gex"].abs().idxmax(), "strike"]) if not below.empty else None
    abs_peak = float(by_strike.loc[by_strike["net_gex"].abs().idxmax(), "strike"]) if not by_strike.empty else None

    # --- 7. futures mapping -------------------------------------------------
    pmap = build_price_map(commodity, spot, source, cfg, client)
    notes.extend(pmap.warnings)

    return GexResult(
        commodity=commodity,
        asof=now,
        source=source,
        underlying_symbol=underlying_symbol,
        underlying_spot=spot,
        futures_price=pmap.futures_spot,
        mapping=pmap,
        chain=df,
        by_strike=by_strike,
        profile=profile,
        net_gex=float(df["gex"].sum()),
        call_gex=float(df.loc[is_call, "gex"].sum()),
        put_gex=float(df.loc[~is_call, "gex"].sum()),
        total_oi=int(df["openInterest"].sum()),
        contracts_used=len(df),
        gamma_flip=flip,
        gamma_flip_futures=pmap.to_futures(flip) if flip else None,
        call_wall=call_wall,
        put_wall=put_wall,
        abs_gamma_peak=abs_peak,
        avg_iv=float(np.average(df["iv"], weights=np.maximum(df["openInterest"], 1e-9))),
        warnings=notes,
    )


# %% [markdown]
# ---
# ## Cell 8 - Translating ETF strike levels into futures prices
#
# A `GLD` 430 strike is only useful to a futures trader once it is expressed in
# `GC` dollars per ounce. The obvious approach - regress the futures price on the
# ETF price in **levels** - is wrong for commodity ETFs, and badly so:
#
# | pair | level R-squared, 1y | level R-squared, 6m |
# |---|---|---|
# | GLD / GC | 0.997 | 0.995 |
# | USO / CL | **0.896** | **0.613** |
# | UNG / NG | **0.604** | **0.578** |
#
# `USO` and `UNG` bleed value through contango roll yield, so their *level*
# relationship to front-month futures drifts continuously. The fit degrades the
# longer you look, which is the signature of a spurious regression.
#
# Daily *returns*, however, track closely - the roll drag is a slow leak, not a
# day-to-day distortion. So we fit a log-return beta and anchor it at today's spot:
#
# $$\ln\!\left(\frac{K_{fut}}{F_0}\right)=\beta \cdot \ln\!\left(\frac{K_{etf}}{E_0}\right)
# \qquad\Longrightarrow\qquad
# K_{fut}=F_0\left(\frac{K_{etf}}{E_0}\right)^{\beta}$$
#
# This is exact at the money by construction, and the roll drift cancels out. The
# return correlation is reported for every commodity - treat it as the confidence
# you should place in the translated levels.

# %%
# =============================================================================
# CELL 8 - ETF <-> futures price mapping
# =============================================================================


@dataclass
class PriceMap:
    """Anchored log-return map between the option underlying and the futures price."""

    etf_symbol: str
    futures_symbol: str
    etf_spot: float
    futures_spot: float
    beta: float
    correlation: float
    n_obs: int
    identity: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_futures(self, etf_level: float | None) -> float | None:
        if etf_level is None or etf_level <= 0:
            return None
        if self.identity:
            return float(etf_level)
        return float(self.futures_spot * (etf_level / self.etf_spot) ** self.beta)

    def to_etf(self, futures_level: float | None) -> float | None:
        if futures_level is None or futures_level <= 0:
            return None
        if self.identity:
            return float(futures_level)
        return float(self.etf_spot * (futures_level / self.futures_spot) ** (1.0 / self.beta))

    @property
    def quality(self) -> str:
        if self.identity:
            return "exact"
        c = abs(self.correlation)
        if c >= 0.90:
            return "strong"
        if c >= 0.75:
            return "good"
        if c >= 0.50:
            return "weak"
        return "unreliable"


def build_price_map(
    commodity: Commodity,
    underlying_spot: float,
    source: str,
    cfg: Config = None,
    client: YahooClient = None,
) -> PriceMap:
    cfg = cfg or CFG
    client = client or CLIENT

    if source == "futures":
        # Already in futures space - nothing to translate.
        return PriceMap(
            etf_symbol=commodity.cme_root,
            futures_symbol=commodity.futures_symbol,
            etf_spot=underlying_spot,
            futures_spot=underlying_spot,
            beta=1.0,
            correlation=1.0,
            n_obs=0,
            identity=True,
        )

    notes: list[str] = []
    fut_px = client.quote(commodity.futures_symbol)["price"]

    try:
        etf_hist = client.daily_history(commodity.etf, cfg.beta_lookback_days)["close"]
        fut_hist = client.daily_history(commodity.futures_symbol, cfg.beta_lookback_days)["close"]
        joined = pd.concat([etf_hist.rename("etf"), fut_hist.rename("fut")], axis=1).dropna()
        joined = joined.tail(cfg.beta_lookback_days)

        rx = np.diff(np.log(joined["etf"].to_numpy()))
        ry = np.diff(np.log(joined["fut"].to_numpy()))
        mask = np.isfinite(rx) & np.isfinite(ry)
        rx, ry = rx[mask], ry[mask]

        if len(rx) < 30:
            raise ValueError(f"only {len(rx)} overlapping return observations")

        beta = float((rx @ ry) / (rx @ rx))
        corr = float(np.corrcoef(rx, ry)[0, 1])

        if not np.isfinite(beta) or beta <= 0:
            raise ValueError(f"degenerate beta {beta}")
        if abs(corr) < cfg.beta_min_corr:
            notes.append(
                f"{commodity.name}: ETF/futures return correlation is only {corr:.2f} "
                f"- translated {commodity.cme_root} levels are indicative, not precise."
            )
        return PriceMap(
            etf_symbol=commodity.etf,
            futures_symbol=commodity.futures_symbol,
            etf_spot=underlying_spot,
            futures_spot=fut_px,
            beta=beta,
            correlation=corr,
            n_obs=len(rx),
            warnings=notes,
        )
    except Exception as exc:
        notes.append(f"{commodity.name}: beta fit failed ({exc}); falling back to a simple ratio map.")
        return PriceMap(
            etf_symbol=commodity.etf,
            futures_symbol=commodity.futures_symbol,
            etf_spot=underlying_spot,
            futures_spot=fut_px,
            beta=1.0,
            correlation=float("nan"),
            n_obs=0,
            warnings=notes,
        )


print("Price mapping ready.")

# %% [markdown]
# ---
# ## Cell 9 - Optional: run on real CME futures options
#
# Everything above is model-agnostic. If you have genuine futures-options open
# interest - a Barchart/DataMine subscription, a broker export, or a CSV saved from
# the CME settlement browser - drop it in here and the whole dashboard recomputes on
# true `GC`/`CL`/`ZC` chains using **Black-76** and the real contract multiplier.
#
# Required CSV columns:
#
# | column | meaning |
# |---|---|
# | `strike` | strike price, in futures quote units |
# | `option_type` | `call` or `put` |
# | `openInterest` | open interest, contracts |
# | `expiry_epoch` | expiry as Unix seconds (or use `expiry` as `YYYY-MM-DD`) |
# | `impliedVolatility` | decimal, e.g. `0.28`. Optional if `bid`/`ask` are present |
# | `bid`, `ask` | optional; lets the solver re-derive IV |
#
# Leave this cell alone if you are using the ETF proxy.

# %%
# =============================================================================
# CELL 9 - Real futures-options adapter
# =============================================================================


def load_futures_chain(path_or_buffer, commodity_key: str, futures_price: float | None = None) -> GexResult:
    """Compute GEX from a real futures-options CSV using Black-76."""
    com = UNIVERSE[commodity_key]
    df = pd.read_csv(path_or_buffer)
    df.columns = [c.strip() for c in df.columns]

    required = {"strike", "option_type", "openInterest"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df["option_type"] = df["option_type"].astype(str).str.lower().str[:4]
    df["option_type"] = np.where(df["option_type"].str.startswith("c"), "call", "put")

    if "expiry_epoch" not in df.columns:
        if "expiry" not in df.columns:
            raise ValueError("Provide either an `expiry_epoch` or an `expiry` (YYYY-MM-DD) column.")
        df["expiry_epoch"] = (
            pd.to_datetime(df["expiry"], utc=True).astype("int64") // 10 ** 9
        )

    if "contractSymbol" not in df.columns:
        df["contractSymbol"] = (
            com.cme_root + "_" + df["expiry_epoch"].astype(str)
            + "_" + df["option_type"].str[0].str.upper() + "_" + df["strike"].astype(str)
        )

    fut_px = float(futures_price) if futures_price else CLIENT.quote(com.futures_symbol)["price"]

    return compute_gex(
        com,
        chain_override=df,
        spot_override=fut_px,
        model="black76",
        multiplier_override=com.point_value,
    )


def upload_futures_chain(commodity_key: str) -> GexResult:
    """Colab file-picker wrapper around `load_futures_chain`."""
    if not IN_COLAB:
        raise RuntimeError("File upload is Colab-only. Call load_futures_chain(path, key) instead.")
    from google.colab import files

    print(f"Select a futures-options CSV for {UNIVERSE[commodity_key].label} ...")
    uploaded = files.upload()
    name = next(iter(uploaded))
    return load_futures_chain(io.BytesIO(uploaded[name]), commodity_key)


print("Futures-options adapter ready:  load_futures_chain(path, 'gold')  /  upload_futures_chain('gold')")


# %%
# =============================================================================
# CELL 10 - Charts
# =============================================================================
# All strike levels are plotted in FUTURES price terms via the Cell 8 map, so the
# axes read directly against a GC / CL / ZC chart.
#
# Colour follows a validated palette rather than trading-screen convention:
#
#   * Net gamma is a DIVERGING quantity (dealers long vs short), so it uses a
#     warm/cool diverging pair - blue for positive, red for negative, neutral grey
#     at the midpoint. Green/red was rejected: it is the classic red-green
#     colour-vision failure and the two poles do not separate under deuteranopia.
#   * Spot and the gamma flip are reference lines, not series, so they wear chart
#     chrome (ink and one reserved accent) rather than a series colour.
#   * Light and dark are each stepped for their own surface. Neither is an
#     automatic inversion of the other.
# =============================================================================

# Validated palette. Substitute your brand's values wholesale; do not hand-tune
# individual entries, because the pairwise separation is what was checked.
PALETTES = {
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "pos": "#3987e5",        # diverging pole - dealers long gamma
        "neg": "#e66767",        # diverging pole - dealers short gamma
        "neutral": "#383835",    # diverging midpoint - reads as "nothing"
        "spot": "#ffffff",       # reference line: you-are-here
        "flip": "#9085e9",       # reference line: regime threshold
        "template": "plotly_dark",
    },
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "pos": "#2a78d6",
        "neg": "#e34948",
        "neutral": "#f0efec",
        "spot": "#0b0b0b",
        "flip": "#4a3aa7",
        "template": "plotly_white",
    },
}


def P() -> dict:
    """The active palette. Set CFG.mode to 'light' or 'dark'."""
    return PALETTES[getattr(CFG, "mode", "dark")]


def _scale(value: float, mode: str = "auto") -> tuple[float, str]:
    """Pick a display scale for dollar amounts."""
    mag = abs(value)
    if mode == "billions" or (mode == "auto" and mag >= 1e9):
        return 1e9, "$bn"
    if mode == "millions" or (mode == "auto" and mag >= 1e6):
        return 1e6, "$mn"
    if mode == "auto" and mag >= 1e3:
        return 1e3, "$k"
    return 1.0, "$"


def fmt_money(value: float, mode: str = "auto") -> str:
    div, unit = _scale(value, mode)
    return f"{value / div:,.2f} {unit}"


def fmt_price(com: Commodity, px: float | None) -> str:
    if px is None or not np.isfinite(px):
        return "n/a"
    if com.price_unit == "cents/bu":
        return f"{px:,.2f}c"
    if px < 10:
        return f"${px:,.3f}"
    return f"${px:,.2f}"


def _layout(fig: go.Figure, title: str, subtitle: str = "", height: int = 620) -> go.Figure:
    p = P()
    fig.update_layout(
        template=p["template"],
        height=height,
        paper_bgcolor=p["surface"],
        plot_bgcolor=p["surface"],
        title=dict(
            text=f"<b>{title}</b>"
            + (f"<br><span style='font-size:12px;color:{p['muted']}'>{subtitle}</span>" if subtitle else ""),
            x=0.01,
            xanchor="left",
            font=dict(size=17, color=p["ink"]),
        ),
        margin=dict(l=76, r=48, t=96, b=64),
        hovermode="closest",
        hoverlabel=dict(bgcolor=p["page"], bordercolor=p["axis"],
                        font=dict(color=p["ink"], size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    font=dict(color=p["ink_secondary"], size=12)),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  color=p["ink_secondary"], size=12),
    )
    # Hairline, solid, one shade off the surface. Never dashed - dashing a grid
    # reads as "threshold" when it is only a grid.
    for axis in (fig.update_xaxes, fig.update_yaxes):
        axis(gridcolor=p["grid"], griddash="solid", linecolor=p["axis"],
             zerolinecolor=p["axis"], tickfont=dict(color=p["muted"], size=11),
             title_font=dict(color=p["ink_secondary"], size=12))
    return fig


def _refline(fig, level, color, label, horizontal: bool, dash="dash", row=None, col=None):
    """Reference lines carry chrome colours, never a series colour."""
    if level is None or not np.isfinite(level):
        return
    kw = dict(line=dict(color=color, width=2, dash=dash),
              annotation_text=label,
              annotation_font=dict(color=color, size=11))
    if row is not None:
        kw["row"], kw["col"] = row, col
    if horizontal:
        fig.add_hline(y=level, annotation_position="right", **kw)
    else:
        fig.add_vline(x=level, annotation_position="top", **kw)


def chart_gex_by_strike(res: GexResult, top_n: int = 40) -> go.Figure:
    """Diverging bars per strike - the gamma walls."""
    p = P()
    com = res.commodity
    bs = res.by_strike.copy()
    bs["abs"] = bs["net_gex"].abs()
    bs = bs.nlargest(min(top_n, len(bs)), "abs").sort_values("strike")
    bs["fut_strike"] = [res.mapping.to_futures(k) for k in bs["strike"]]

    div, unit = _scale(bs[["call_gex", "put_gex"]].abs().to_numpy().max(), CFG.scale_label)

    fig = go.Figure()
    for name, col, oi_col, colour in [
        ("Call gamma", "call_gex", "call_oi", p["pos"]),
        ("Put gamma", "put_gex", "put_oi", p["neg"]),
    ]:
        fig.add_trace(go.Bar(
            y=bs["fut_strike"], x=bs[col] / div, orientation="h", name=name,
            marker=dict(
                color=colour,
                # A 2px surface gap between adjacent fills - not a border around marks.
                line=dict(color=p["surface"], width=1.5),
            ),
            customdata=np.stack([bs["strike"], bs[oi_col], bs["avg_iv"] * 100], axis=-1),
            hovertemplate=(
                f"<b>{com.cme_root} %{{y:,.2f}}</b><br>"
                f"{res.underlying_symbol} strike %{{customdata[0]:,.2f}}<br>"
                f"{name} %{{x:,.2f}} {unit}<br>OI %{{customdata[1]:,.0f}}<br>"
                "Avg IV %{customdata[2]:.1f}%<extra></extra>"
            ),
        ))

    _refline(fig, res.futures_price, p["spot"], f"spot {fmt_price(com, res.futures_price)}", True, "solid")
    _refline(fig, res.gamma_flip_futures, p["flip"], f"gamma flip {fmt_price(com, res.gamma_flip_futures)}", True)

    fig.update_layout(barmode="relative", bargap=0.18)
    fig.update_xaxes(title=f"Dealer gamma exposure per 1% move ({unit})")
    fig.update_yaxes(title=f"{com.cme_root} strike ({com.price_unit})")
    return _layout(
        fig,
        f"{com.label} - gamma by strike",
        f"{res.source.replace('_', ' ')} - {res.contracts_used:,} contracts, "
        f"{res.total_oi:,} OI - as of {res.asof:%Y-%m-%d %H:%M UTC}",
        height=max(600, 21 * len(bs)),
    )


def chart_gamma_profile(res: GexResult) -> go.Figure:
    """Net dealer gamma re-evaluated across hypothetical spot levels."""
    p = P()
    com = res.commodity
    prof = res.profile.copy()
    prof["fut_spot"] = [res.mapping.to_futures(s) for s in prof["spot"]]
    div, unit = _scale(prof["net_gex"].abs().max(), CFG.scale_label)
    y = prof["net_gex"] / div

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prof["fut_spot"], y=y.clip(lower=0), mode="lines", name="Dealers long gamma",
        line=dict(width=0), fill="tozeroy", fillcolor="rgba(57,135,229,0.32)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=prof["fut_spot"], y=y.clip(upper=0), mode="lines", name="Dealers short gamma",
        line=dict(width=0), fill="tozeroy", fillcolor="rgba(230,103,103,0.32)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=prof["fut_spot"], y=y, mode="lines", name="Net GEX", showlegend=False,
        line=dict(color=p["ink_secondary"], width=2),
        hovertemplate=f"{com.cme_root} %{{x:,.2f}}<br>Net GEX %{{y:,.2f}} {unit}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=p["axis"], width=1))
    _refline(fig, res.futures_price, p["spot"], f"spot {fmt_price(com, res.futures_price)}", False, "solid")
    _refline(fig, res.gamma_flip_futures, p["flip"], f"flip {fmt_price(com, res.gamma_flip_futures)}", False)

    fig.update_xaxes(title=f"{com.cme_root} price ({com.price_unit})")
    fig.update_yaxes(title=f"Net dealer gamma per 1% move ({unit})")
    dist = ("" if not res.gamma_flip_futures
            else f" - flip is {(res.futures_price / res.gamma_flip_futures - 1) * 100:+.1f}% from spot")
    return _layout(fig, f"{com.label} - gamma profile", f"{res.regime}{dist}", height=500)


def chart_term_structure(res: GexResult) -> go.Figure:
    """How the gamma is distributed across expiries."""
    p = P()
    com = res.commodity
    df = res.chain.copy()
    df["expiry_date"] = df["expiry"].dt.date
    is_call = df["option_type"] == "call"
    agg = (
        df.assign(c=df["gex"].where(is_call, 0.0), pu=df["gex"].where(~is_call, 0.0))
        .groupby("expiry_date", as_index=False)
        .agg(net=("gex", "sum"), call=("c", "sum"), put=("pu", "sum"), oi=("openInterest", "sum"))
        .sort_values("expiry_date")
    )
    div, unit = _scale(agg[["call", "put"]].abs().to_numpy().max(), CFG.scale_label)
    x = [str(d) for d in agg["expiry_date"]]

    fig = go.Figure()
    for name, col, colour in [("Call gamma", "call", p["pos"]), ("Put gamma", "put", p["neg"])]:
        fig.add_trace(go.Bar(
            x=x, y=agg[col] / div, name=name,
            marker=dict(color=colour, line=dict(color=p["surface"], width=1.5)),
            hovertemplate="%{x}<br>" + name + " %{y:,.2f} " + unit + "<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=x, y=agg["net"] / div, name="Net", mode="lines+markers",
        line=dict(color=p["ink_secondary"], width=2), marker=dict(size=8),
        hovertemplate="%{x}<br>Net GEX %{y:,.2f} " + unit + "<extra></extra>",
    ))
    fig.update_layout(barmode="relative", bargap=0.3)
    fig.update_xaxes(title="Expiry", type="category")
    fig.update_yaxes(title=f"Dealer gamma per 1% move ({unit})")
    return _layout(fig, f"{com.label} - gamma term structure",
                   "Front expiries dominate and decay fastest as they roll off", height=440)


def chart_cross_commodity(results: dict[str, GexResult]) -> go.Figure:
    """One bar per commodity - who is long gamma, who is short."""
    p = P()
    rows = []
    for r in results.values():
        rows.append({
            "name": r.commodity.name,
            "root": r.commodity.cme_root,
            "net": r.net_gex / 1e6,
            "flip_dist": ((r.futures_price / r.gamma_flip_futures - 1) * 100
                          if r.gamma_flip_futures else np.nan),
            "iv": r.avg_iv * 100,
        })
    df = pd.DataFrame(rows).sort_values("net")

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.56, 0.44], horizontal_spacing=0.14,
        subplot_titles=("Net dealer gamma per 1% move ($mn)", "Spot distance to gamma flip (%)"),
    )
    fig.add_trace(go.Bar(
        y=df["root"], x=df["net"], orientation="h", showlegend=False,
        marker=dict(color=[p["pos"] if v >= 0 else p["neg"] for v in df["net"]],
                    line=dict(color=p["surface"], width=1.5)),
        text=[f"{v:,.1f}" for v in df["net"]], textposition="outside",
        textfont=dict(color=p["ink_secondary"], size=11),
        customdata=np.stack([df["name"], df["iv"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>Net GEX %{x:,.2f} $mn<br>"
                      "OI-weighted IV %{customdata[1]:.1f}%<extra></extra>",
    ), row=1, col=1)

    d2 = df.dropna(subset=["flip_dist"]).sort_values("flip_dist")
    fig.add_trace(go.Bar(
        y=d2["root"], x=d2["flip_dist"], orientation="h", showlegend=False,
        marker=dict(color=[p["pos"] if v >= 0 else p["neg"] for v in d2["flip_dist"]],
                    line=dict(color=p["surface"], width=1.5)),
        text=[f"{v:+.1f}%" for v in d2["flip_dist"]], textposition="outside",
        textfont=dict(color=p["ink_secondary"], size=11),
        hovertemplate="<b>%{y}</b><br>Spot is %{x:+.1f}% from the flip<extra></extra>",
    ), row=1, col=2)

    for c in (1, 2):
        fig.add_vline(x=0, line=dict(color=p["axis"], width=1), row=1, col=c)
    for ann in fig.layout.annotations:
        ann.font = dict(color=p["ink_secondary"], size=12)

    return _layout(fig, "Commodity complex - dealer gamma at a glance",
                   "Blue = dealers long gamma (volatility damped)   |   "
                   "Red = dealers short gamma (volatility amplified)",
                   height=max(420, 46 * len(df)))


print("Charts ready.")


# %%
# =============================================================================
# CELL 11 - Dashboard assembly
# =============================================================================


def build_all(
    watchlist: list[str] = None,
    cfg: Config = None,
    verbose: bool = True,
) -> dict[str, GexResult]:
    """Run the GEX calculation across the watchlist, tolerating per-name failures."""
    watchlist = watchlist or DEFAULT_WATCHLIST
    cfg = cfg or CFG
    results: dict[str, GexResult] = {}
    failures: list[tuple[str, str]] = []

    # One timestamp for the whole run, so every commodity in the snapshot is
    # priced off the same time-to-expiry rather than drifting as the loop runs.
    run_asof = dt.datetime.now(dt.timezone.utc)

    for key in watchlist:
        com = UNIVERSE.get(key)
        if com is None:
            failures.append((key, "not in UNIVERSE"))
            continue
        try:
            t0 = time.time()
            results[key] = compute_gex(com, cfg=cfg, asof=run_asof)
            if verbose:
                r = results[key]
                print(f"  {com.name:<16s} {r.contracts_used:>5,} contracts  "
                      f"net GEX {fmt_money(r.net_gex, 'millions'):>16s}  "
                      f"map={r.mapping.quality:<10s} ({time.time() - t0:.1f}s)")
        except Exception as exc:
            failures.append((key, f"{type(exc).__name__}: {exc}"))
            if verbose:
                print(f"  {key:<16s} FAILED - {type(exc).__name__}: {exc}")

    if failures and verbose:
        print(f"\n{len(failures)} commodity(ies) unavailable this run:")
        for k, why in failures:
            print(f"   - {k}: {why}")
    return results


def summary_table(results: dict[str, GexResult]) -> pd.DataFrame:
    """The one table to read first."""
    rows = []
    for r in results.values():
        c = r.commodity
        rows.append({
            "Commodity": c.name,
            "Fut": c.cme_root,
            "Price": fmt_price(c, r.futures_price),
            "Regime": "LONG gamma" if r.net_gex >= 0 else "SHORT gamma",
            "Net GEX": fmt_money(r.net_gex, "millions"),
            "Call GEX": fmt_money(r.call_gex, "millions"),
            "Put GEX": fmt_money(r.put_gex, "millions"),
            "Gamma flip": fmt_price(c, r.gamma_flip_futures),
            "Flip dist": (f"{(r.futures_price / r.gamma_flip_futures - 1) * 100:+.1f}%"
                          if r.gamma_flip_futures else "n/a"),
            "Call wall": fmt_price(c, r.to_futures(r.call_wall)),
            "Put wall": fmt_price(c, r.to_futures(r.put_wall)),
            "IV": f"{r.avg_iv * 100:.1f}%",
            "OI": f"{r.total_oi:,}",
            "Map": r.mapping.quality,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("Commodity").reset_index(drop=True)


def print_briefing(res: GexResult) -> None:
    """Plain-language read of one commodity's gamma structure."""
    c = res.commodity
    flip = res.gamma_flip_futures
    print("=" * 78)
    print(f"{c.label}  -  {fmt_price(c, res.futures_price)} ({c.price_unit})")
    print("=" * 78)
    print(f"  Regime          : {res.regime}")
    print(f"  Net GEX         : {fmt_money(res.net_gex, 'millions')} per 1% move")
    print(f"      calls       : {fmt_money(res.call_gex, 'millions')}")
    print(f"      puts        : {fmt_money(res.put_gex, 'millions')}")
    print(f"  Gamma flip      : {fmt_price(c, flip)}"
          + (f"   ({(res.futures_price / flip - 1) * 100:+.1f}% from spot)" if flip else ""))
    print(f"  Call wall       : {fmt_price(c, res.to_futures(res.call_wall))}   (upside gamma magnet)")
    print(f"  Put wall        : {fmt_price(c, res.to_futures(res.put_wall))}   (downside gamma magnet)")
    print(f"  OI-weighted IV  : {res.avg_iv * 100:.1f}%")
    print(f"  Book            : {res.contracts_used:,} contracts / {res.total_oi:,} OI "
          f"from {res.underlying_symbol} ({res.source.replace('_', ' ')})")
    if not res.mapping.identity:
        print(f"  Level mapping   : beta {res.mapping.beta:.3f}, return corr {res.mapping.correlation:.2f} "
              f"over {res.mapping.n_obs} days -> {res.mapping.quality}")

    if res.net_gex >= 0:
        print("\n  Read: dealers are LONG gamma. Their hedging leans against the move -")
        print("        selling into rallies, buying dips. Expect realised vol to be")
        print("        damped and ranges to hold, especially between the walls.")
    else:
        print("\n  Read: dealers are SHORT gamma. Their hedging goes WITH the move -")
        print("        buying strength, selling weakness. Expect trend extension and")
        print("        larger realised ranges; breaks of the walls can accelerate.")
    if flip:
        side = "above" if res.futures_price > flip else "below"
        print(f"        Spot is {side} the flip at {fmt_price(c, flip)} - losing that level")
        print("        would switch the regime.")
    if res.warnings:
        print("\n  Notes:")
        for w in res.warnings[:6]:
            print(f"   - {w}")
    print()


def show_commodity(res: GexResult, term_structure: bool = True) -> None:
    """Briefing plus the full chart set for one commodity."""
    print_briefing(res)
    chart_gex_by_strike(res).show()
    chart_gamma_profile(res).show()
    if term_structure:
        chart_term_structure(res).show()


print("Dashboard assembly ready.")


# %% [markdown]
# ---
# ## Cell 12 - Daily snapshots and GEX history
#
# GEX is far more useful as a **time series** than as a single reading. A gold book
# that has been long gamma all month and flips negative today is a much stronger
# signal than the level itself. These functions append one row per commodity per
# day to a CSV (on Google Drive in Colab, or a local folder otherwise), then chart
# the history.

# %%
# =============================================================================
# CELL 12 - Snapshot persistence
# =============================================================================


def mount_drive() -> bool:
    """Mount Google Drive so snapshots survive runtime restarts."""
    if not IN_COLAB:
        return False
    if os.path.isdir("/content/drive/MyDrive"):
        return True
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        return True
    except Exception as exc:
        print(f"Drive mount failed ({exc}); snapshots will go to local disk instead.")
        return False


def _snapshot_dir(cfg: Config = None) -> str:
    cfg = cfg or CFG
    target = cfg.snapshot_dir
    if target.startswith("/content/drive") and not os.path.isdir("/content/drive/MyDrive"):
        target = "./commodity_gex"          # Drive not mounted - fall back to local
    os.makedirs(target, exist_ok=True)
    return target


def snapshot_rows(results: dict[str, GexResult]) -> pd.DataFrame:
    rows = []
    for r in results.values():
        c = r.commodity
        rows.append({
            "asof": r.asof.isoformat(),
            "date": r.asof.date().isoformat(),
            "key": c.key,
            "commodity": c.name,
            "futures_root": c.cme_root,
            "source": r.source,
            "underlying": r.underlying_symbol,
            "underlying_spot": r.underlying_spot,
            "futures_price": r.futures_price,
            "net_gex": r.net_gex,
            "call_gex": r.call_gex,
            "put_gex": r.put_gex,
            "gamma_flip_futures": r.gamma_flip_futures,
            "call_wall_futures": r.to_futures(r.call_wall),
            "put_wall_futures": r.to_futures(r.put_wall),
            "avg_iv": r.avg_iv,
            "total_oi": r.total_oi,
            "contracts": r.contracts_used,
            "map_beta": r.mapping.beta,
            "map_corr": r.mapping.correlation,
            "dealer_convention": CFG.dealer_convention,
        })
    return pd.DataFrame(rows)


def save_snapshot(results: dict[str, GexResult], cfg: Config = None) -> str:
    """Append today's readings to the rolling history file (idempotent per day)."""
    cfg = cfg or CFG
    directory = _snapshot_dir(cfg)
    path = os.path.join(directory, "gex_history.csv")

    fresh = snapshot_rows(results)
    if os.path.exists(path):
        old = pd.read_csv(path)
        # Re-running the same day overwrites that day rather than duplicating it.
        keep = ~(old["date"].isin(fresh["date"]) & old["key"].isin(fresh["key"]))
        combined = pd.concat([old[keep], fresh], ignore_index=True)
    else:
        combined = fresh

    combined = combined.sort_values(["date", "key"]).reset_index(drop=True)
    combined.to_csv(path, index=False)

    # Full per-strike detail, one file per day, for later research.
    detail_dir = os.path.join(directory, "chains")
    os.makedirs(detail_dir, exist_ok=True)
    stamp = dt.date.today().isoformat()
    for key, r in results.items():
        r.by_strike.assign(
            key=key,
            futures_level=[r.mapping.to_futures(k) for k in r.by_strike["strike"]],
        ).to_csv(os.path.join(detail_dir, f"{stamp}_{key}.csv"), index=False)

    print(f"Saved {len(fresh)} rows -> {path}   ({len(combined)} total, "
          f"{combined['date'].nunique()} distinct days)")
    return path


def load_history(cfg: Config = None) -> pd.DataFrame:
    cfg = cfg or CFG
    path = os.path.join(_snapshot_dir(cfg), "gex_history.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["key", "date"])


def chart_history(key: str, cfg: Config = None) -> go.Figure | None:
    """Net GEX and futures price over time, as two stacked panels.

    Deliberately NOT a dual-axis chart. Overlaying GEX (dollars) and price
    (dollars per ounce) on two y-scales would let the arbitrary alignment of
    those scales invent a correlation that is not in the data. Stacked panels
    sharing one x-axis show the same relationship honestly: read down the
    columns to line a sign flip up against what price did next.
    """
    p = P()
    hist = load_history(cfg)
    if hist.empty or key not in set(hist["key"]):
        print(f"No saved history for '{key}' yet - run save_snapshot(RESULTS) on a few "
              "separate days first.")
        return None

    d = hist[hist["key"] == key].sort_values("date")
    com = UNIVERSE[key]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        row_heights=[0.46, 0.54],
        subplot_titles=("Net dealer gamma per 1% move ($mn)",
                        f"{com.cme_root} price and gamma flip ({com.price_unit})"),
    )
    fig.add_trace(go.Bar(
        x=d["date"], y=d["net_gex"] / 1e6, name="Net GEX", showlegend=False,
        marker=dict(color=[p["pos"] if v >= 0 else p["neg"] for v in d["net_gex"]],
                    line=dict(color=p["surface"], width=1.5)),
        hovertemplate="%{x|%Y-%m-%d}<br>Net GEX %{y:,.1f} $mn<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=p["axis"], width=1), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=d["date"], y=d["futures_price"], name=f"{com.cme_root} price",
        mode="lines+markers", line=dict(color=p["spot"], width=2), marker=dict(size=8),
        hovertemplate="%{x|%Y-%m-%d}<br>Price %{y:,.2f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["gamma_flip_futures"], name="Gamma flip",
        mode="lines", line=dict(color=p["flip"], width=2, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>Flip %{y:,.2f}<extra></extra>",
    ), row=2, col=1)

    fig.update_yaxes(title_text="$mn", row=1, col=1)
    fig.update_yaxes(title_text=com.price_unit, row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(color=p["ink_secondary"], size=12)

    return _layout(fig, f"{com.label} - GEX history",
                   "Regime changes matter more than the level - line a sign flip in the "
                   "top panel up against price below", height=600)


print("Snapshot persistence ready.")

# %%
# =============================================================================
# CELL 13 - Keep it live
# =============================================================================
# Colab has no built-in cron. Two options, depending on how live you need it.
#
#   refresh_loop(...)  -> re-runs on an interval for as long as the runtime is
#                         alive. Good for watching an active session.
#   daily_run()        -> one clean pass + snapshot. This is what a scheduler
#                         should call (see the final cell).
# =============================================================================


def daily_run(watchlist: list[str] = None, save: bool = True, show: bool = True) -> dict[str, GexResult]:
    """One full refresh: pull, compute, display, persist."""
    global RESULTS, SUMMARY
    CLIENT.clear_cache()
    RESULTS = build_all(watchlist or DEFAULT_WATCHLIST, verbose=True)
    SUMMARY = summary_table(RESULTS)
    if show:
        display(SUMMARY)
        chart_cross_commodity(RESULTS).show()
    if save and CFG.save_snapshots:
        mount_drive()
        save_snapshot(RESULTS)
    return RESULTS


def refresh_loop(interval_minutes: int = 30, iterations: int = 12, watchlist: list[str] = None):
    """Re-run every `interval_minutes` while the runtime stays alive.

    Colab disconnects idle runtimes after ~90 minutes and caps sessions at ~12 hours,
    so treat this as an intraday monitor, not a scheduler. For genuinely unattended
    daily runs use `run_gex.py` with a scheduler, or the GitHub Action in the final cell.
    """
    from IPython.display import clear_output

    for i in range(iterations):
        clear_output(wait=True)
        print(f"Refresh {i + 1}/{iterations}  -  {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}")
        try:
            daily_run(watchlist)
        except Exception as exc:
            print(f"Refresh failed ({type(exc).__name__}: {exc}); retrying next interval.")
        if i < iterations - 1:
            nxt = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=interval_minutes)
            print(f"\nNext refresh at {nxt:%H:%M UTC}. Interrupt the cell to stop.")
            time.sleep(interval_minutes * 60)


# %%
# =============================================================================
# CELL 14 - Sensitivity check: how much does the dealer assumption matter?
# =============================================================================
# Recompute one commodity under every sign convention. If the sign of net GEX is
# stable across conventions, the reading is driven by the data; if it flips, you
# are reading your own assumption back out.


def convention_sensitivity(key: str = "gold") -> pd.DataFrame:
    com = UNIVERSE[key]
    rows = []
    original = CFG.dealer_convention
    # Pin the valuation time so the four runs differ only in the convention -
    # otherwise seconds of wall-clock drift change time-to-expiry between them.
    asof = dt.datetime.now(dt.timezone.utc)
    try:
        for conv in CONVENTION_SIGNS:
            CFG.dealer_convention = conv
            r = compute_gex(com, cfg=CFG, asof=asof)
            rows.append({
                "convention": conv,
                "net_gex_$mn": round(r.net_gex / 1e6, 2),
                "regime": "LONG gamma" if r.net_gex >= 0 else "SHORT gamma",
                "gamma_flip": fmt_price(com, r.gamma_flip_futures),
                "call_wall": fmt_price(com, r.to_futures(r.call_wall)),
                "put_wall": fmt_price(com, r.to_futures(r.put_wall)),
            })
    finally:
        CFG.dealer_convention = original
    return pd.DataFrame(rows)


print("Live-refresh and sensitivity helpers ready:")
print("   daily_run()                      - one full refresh + snapshot")
print("   refresh_loop(30, 12)             - refresh every 30 min for 6 hours")
print("   chart_history('gold')            - GEX vs price over saved days")
print("   convention_sensitivity('gold')   - how much the dealer assumption matters")
print("   load_futures_chain(path,'gold')  - swap in real CME futures options")


# %% [markdown]
# ---
# # Cell 15 - RUN THE DASHBOARD
#
# Everything above is definitions. This is the section to re-run each day.

# %%
# =============================================================================
# CELL 15 - RUN
# =============================================================================
WATCHLIST = ["gold", "crude", "natgas", "corn", "wheat", "silver", "copper", "soybeans"]

print(f"Building commodity GEX  -  {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}")
print(f"Dealer convention: {CFG.dealer_convention}\n")

CLIENT.clear_cache()
RESULTS = build_all(WATCHLIST)

print(f"\n{len(RESULTS)} of {len(WATCHLIST)} commodities loaded.\n")
SUMMARY = summary_table(RESULTS)
display(SUMMARY)

# %%
# --- Cross-commodity overview -------------------------------------------------
chart_cross_commodity(RESULTS).show()

# %%
# --- Per-commodity detail -----------------------------------------------------
# Trim this list to focus on fewer names.
for _key in ["gold", "crude", "natgas", "corn", "wheat"]:
    if _key in RESULTS:
        show_commodity(RESULTS[_key])

# %%
# --- Single-name deep dive ----------------------------------------------------
# Change the key to any entry in RESULTS.
FOCUS = "gold"
if FOCUS in RESULTS:
    show_commodity(RESULTS[FOCUS])
    display(
        RESULTS[FOCUS]
        .by_strike.assign(
            futures_level=lambda d: [RESULTS[FOCUS].mapping.to_futures(k) for k in d["strike"]],
            net_gex_mn=lambda d: d["net_gex"] / 1e6,
        )[["strike", "futures_level", "net_gex_mn", "call_oi", "put_oi", "avg_iv"]]
        .sort_values("net_gex_mn", key=abs, ascending=False)
        .head(20)
        .round(3)
    )

# %%
# --- Save today's reading -----------------------------------------------------
if CFG.save_snapshots and RESULTS:
    mount_drive()
    save_snapshot(RESULTS)

# %%
# --- History chart (needs at least 2 saved days) -------------------------------
_fig = chart_history("gold")
if _fig:
    _fig.show()

# %%
# --- How much does the dealer assumption matter? ------------------------------
display(convention_sensitivity("gold"))


# %% [markdown]
# ---
# # Cell 16 - Running it every day, unattended
#
# Colab notebooks do not run on a schedule by themselves. Four ways to get a
# genuine daily update, cheapest first.
#
# ### Option A - manual, 10 seconds a day
# Open the notebook, `Runtime > Run all`. Snapshots accumulate in Drive and the
# history charts fill in.
#
# ### Option B - intraday monitor
# Run `refresh_loop(30, 12)` and leave the tab open. Refreshes every 30 minutes
# for six hours. Colab kills idle runtimes after roughly 90 minutes and caps any
# session near 12 hours, so this covers a trading session, not a week.
#
# ### Option C - your own machine, no Colab (recommended if you have a PC that is on)
# The repo ships `run_gex.py`, which runs this same code headlessly and writes a
# self-contained HTML dashboard:
#
# ```
# py run_gex.py --open
# ```
#
# Point Windows Task Scheduler (or cron) at it and you get a fresh dashboard every
# weekday with no cloud account involved. `setup_schedule.ps1` in the repo registers
# the task for you.
#
# ### Option D - GitHub Actions, free tier
# Run the notebook headlessly in CI and commit the snapshot CSV back to the repo.
# The workflow ships as `.github/workflows/daily-gex.yml`:
#
# ```yaml
# on:
#   schedule:
#     - cron: "30 21 * * 1-5"   # 21:30 UTC weekdays, after the US equity close
# ```
#
# **On timing:** run it *after* the 16:00 ET equity close. Open interest for a given
# session is only final the next morning - OI printed intraday is yesterday's number,
# so a pre-close run pairs stale OI with fresh spot.
#
# ---
#
# # How to read the output
#
# ### Net GEX sign
# **Positive** - dealers are long gamma. They sell rallies and buy dips to stay
# hedged, which mechanically damps realised volatility. Ranges tend to hold and mean
# reversion works better than momentum.
# **Negative** - dealers are short gamma. They buy strength and sell weakness,
# feeding the move. Trends extend, ranges break, and realised vol runs above implied.
#
# ### Gamma flip
# The price where net GEX crosses zero - the boundary between those two regimes. It
# is the single most actionable number here. Spot trading a long way above the flip
# is a stable, fadeable tape; spot approaching or losing the flip is where character
# changes. It moves daily as open interest rolls.
#
# ### Call wall / put wall
# The strikes carrying the heaviest dealer gamma above and below spot. In a
# positive-gamma regime these behave like magnets and soft barriers - hedging flow
# resists moves through them. In a negative-gamma regime the same levels become
# accelerants: once through, hedging pushes price further rather than pinning it.
#
# ### Gamma term structure
# Front expiries carry most of the gamma and decay fastest. A book dominated by a
# single near expiry means the whole structure resets the day after it rolls off -
# check this before treating a level as durable.
#
# ---
#
# # Known limitations - read these honestly
#
# **1. The dealer sign convention is assumed, not measured.** Nobody outside the
# clearing houses observes true dealer inventory. "Long calls, short puts" is the
# industry-standard guess and it is wrong for some books some of the time. Every
# published GEX number, including commercial ones, carries this same caveat. Cell 14
# recomputes under all four conventions - if your conclusion inverts, you were
# leaning on the assumption, not the data.
#
# **2. ETF options are a proxy for futures options.** The dollar magnitudes are
# genuine ETF dealer gamma, not CME futures gamma - the CME books are larger. The
# *shape* (where gamma sits relative to spot) is the transferable part. Use the
# Cell 9 adapter with real data when precision matters.
#
# **3. Level translation degrades with correlation.** Gold, wheat, copper and silver
# map at 0.90-0.95 return correlation. Natural gas maps at roughly 0.63 - `UNG` rolls
# its own futures calendar and bleeds on contango, so treat translated `NG` levels as
# a sketch. The `Map` column in the summary table grades every name.
#
# **4. The gamma profile holds the vol surface fixed.** Real implied vol rises as
# spot falls, so the true flip usually sits slightly below the computed one. The
# profile is a map, not a trigger.
#
# **5. Open interest is end-of-day.** Nothing here sees intraday positioning.
#
# **6. Grain ETF chains are thin.** `CORN`, `WEAT` and especially `SOYB` carry a few
# thousand contracts of OI across a handful of strikes. The gamma structure is real
# but coarse - do not read three-decimal precision into a `SOYB` wall.
#
# None of this is investment advice.
