
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
