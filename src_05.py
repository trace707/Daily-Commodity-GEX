
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
