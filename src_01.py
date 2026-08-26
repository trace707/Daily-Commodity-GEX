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
import importlib, subprocess, sys

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
