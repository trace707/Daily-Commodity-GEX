
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
