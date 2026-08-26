
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
