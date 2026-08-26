"""End-to-end test of the notebook against live market data.

Executes commodity_gex_dashboard.py in-process with plotly rendering stubbed out,
then asserts invariants on the results. Run:  py test_dashboard.py
"""
from __future__ import annotations

import os
import sys
import types

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

import numpy as np
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "commodity_gex_dashboard.py")

# Stub the renderer: we care that figures build, not that a browser opens.
_shown: list[str] = []
go.Figure.show = lambda self, *a, **k: _shown.append(  # type: ignore[method-assign]
    (self.layout.title.text or "untitled").split("<br>")[0]
)

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    src = open(SCRIPT, encoding="utf-8").read()

    # Trim the watchlist so the test finishes quickly.
    src = src.replace(
        'WATCHLIST = ["gold", "crude", "natgas", "corn", "wheat", "silver", "copper", "soybeans"]',
        'WATCHLIST = ["gold", "crude", "natgas", "corn", "wheat"]',
    )
    src = src.replace('for _key in ["gold", "crude", "natgas", "corn", "wheat"]:', 'for _key in ["gold"]:')
    # Keep the test off the user's Drive folder.
    src = src.replace(
        'snapshot_dir: str = "/content/drive/MyDrive/commodity_gex"',
        'snapshot_dir: str = "./_test_snapshots"',
    )

    mod = types.ModuleType("gexnb")
    mod.__file__ = SCRIPT
    mod.__dict__["display"] = lambda *a, **k: None
    # `dataclasses` resolves cls.__module__ through sys.modules, so the synthetic
    # module has to be registered before any @dataclass in the script is evaluated.
    sys.modules["gexnb"] = mod

    print("=" * 74)
    print("EXECUTING NOTEBOOK")
    print("=" * 74)
    exec(compile(src, SCRIPT, "exec"), mod.__dict__)

    g = mod.__dict__
    RESULTS = g["RESULTS"]

    print("\n" + "=" * 74)
    print("ASSERTIONS")
    print("=" * 74)

    check("at least 4 commodities computed", len(RESULTS) >= 4, f"got {len(RESULTS)}")
    check("figures rendered", len(_shown) >= 3, f"{len(_shown)} figures")

    for key, r in RESULTS.items():
        pre = f"[{key}]"
        check(f"{pre} contracts survived filtering", r.contracts_used > 0, f"{r.contracts_used}")
        check(f"{pre} net GEX is finite", np.isfinite(r.net_gex), f"{r.net_gex:,.0f}")
        check(f"{pre} futures price positive", r.futures_price > 0, f"{r.futures_price:,.2f}")

        # Net must equal calls + puts exactly.
        check(f"{pre} net == call + put", abs(r.net_gex - (r.call_gex + r.put_gex)) < 1e-6)

        # Under the default convention call gamma is >= 0 and put gamma <= 0.
        if g["CFG"].dealer_convention == "long_calls_short_puts":
            check(f"{pre} call GEX >= 0", r.call_gex >= 0, f"{r.call_gex:,.0f}")
            check(f"{pre} put GEX <= 0", r.put_gex <= 0, f"{r.put_gex:,.0f}")

        # Per-strike aggregation must reconcile with the contract-level total.
        agg = r.by_strike["net_gex"].sum()
        check(f"{pre} by_strike reconciles", abs(agg - r.net_gex) < max(1.0, abs(r.net_gex) * 1e-9),
              f"{agg:,.0f} vs {r.net_gex:,.0f}")

        # The profile evaluated at spot must reproduce the reported net GEX.
        prof = r.profile
        at_spot = np.interp(r.underlying_spot, prof["spot"], prof["net_gex"])
        rel = abs(at_spot - r.net_gex) / max(abs(r.net_gex), 1.0)
        check(f"{pre} profile agrees with net GEX at spot", rel < 0.05, f"{rel * 100:.2f}% apart")

        # The mapping must be exact at the money and monotone.
        mapped = r.mapping.to_futures(r.underlying_spot)
        check(f"{pre} mapping exact at spot", abs(mapped - r.futures_price) < 1e-6,
              f"{mapped:,.4f} vs {r.futures_price:,.4f}")
        lo = r.mapping.to_futures(r.underlying_spot * 0.9)
        hi = r.mapping.to_futures(r.underlying_spot * 1.1)
        check(f"{pre} mapping monotone increasing", lo < r.futures_price < hi,
              f"{lo:,.2f} < {r.futures_price:,.2f} < {hi:,.2f}")

        # Round-tripping a level through the map must return it.
        rt = r.mapping.to_etf(r.mapping.to_futures(r.underlying_spot * 1.07))
        check(f"{pre} mapping round-trips", abs(rt - r.underlying_spot * 1.07) < 1e-4)

        # Walls must straddle spot.
        if r.call_wall is not None:
            check(f"{pre} call wall at or above spot", r.call_wall >= r.underlying_spot * 0.999)
        if r.put_wall is not None:
            check(f"{pre} put wall at or below spot", r.put_wall <= r.underlying_spot * 1.001)

        # IV must be sane after cleaning.
        iv = r.chain["iv"]
        check(f"{pre} IVs within bounds", bool(((iv > 0.005) & (iv < 5.01)).all()),
              f"min {iv.min():.3f} max {iv.max():.3f}")
        check(f"{pre} no NaN gamma", bool(np.isfinite(r.chain['gex']).all()))

        # Gamma is never negative and never NaN. It may legitimately underflow to
        # exactly zero in the deep wings of a near-dated expiry (|d1| > ~38 puts
        # the normal density below float64 resolution), so the invariant is
        # non-negative and finite - not strictly positive.
        gam = r.chain["gamma"]
        check(f"{pre} gamma non-negative and finite",
              bool((gam >= 0).all() and np.isfinite(gam).all()),
              f"min {gam.min():.3e} max {gam.max():.3e}")

        # No expired contracts. Yahoo keeps serving the just-expired chain for the
        # rest of the session; those have no gamma but would inflate OI counts.
        check(f"{pre} no expired contracts in the book",
              bool((r.chain["dte"] > 0).all()),
              f"min dte {r.chain['dte'].min():.4f}")

        # At least the near-the-money strikes must carry real gamma.
        atm = r.chain[(r.chain["strike"] / r.underlying_spot).between(0.95, 1.05)]
        if len(atm):
            check(f"{pre} ATM gamma strictly positive",
                  bool((atm["gamma"] > 0).all()), f"{len(atm)} ATM contracts")

    # --- Flip point must actually be a zero crossing --------------------------
    for key, r in RESULTS.items():
        if r.gamma_flip is None:
            continue
        val = np.interp(r.gamma_flip, r.profile["spot"], r.profile["net_gex"])
        scale = r.profile["net_gex"].abs().max()
        check(f"[{key}] net GEX ~ 0 at the flip", abs(val) < 0.02 * scale,
              f"{val:,.0f} vs peak {scale:,.0f}")

    # --- Convention sensitivity must invert the sign --------------------------
    com = g["UNIVERSE"]["gold"]
    base = RESULTS["gold"]
    # Identical config apart from the convention, so the result must be an exact
    # negation: net = C - P becomes net = -C + P.
    #
    # Freeze the cache and recompute the baseline from it. RESULTS["gold"] was
    # built minutes earlier and Cell 16's sensitivity sweep has since refreshed
    # the cached chain, so comparing against it would be comparing two different
    # snapshots of the market, not two conventions.
    import dataclasses as _dc

    g["CLIENT"].cfg.cache_ttl_seconds = 10 ** 6
    import datetime as _dtm

    pinned = _dtm.datetime.now(_dtm.timezone.utc)
    base_now = g["compute_gex"](com, cfg=g["CFG"], asof=pinned)

    cfg = _dc.replace(g["CFG"], dealer_convention="short_calls_long_puts")
    inv = g["compute_gex"](com, cfg=cfg, asof=pinned)
    check("mirroring the dealer convention negates net GEX exactly",
          abs(inv.net_gex + base_now.net_gex) < max(1.0, abs(base_now.net_gex) * 1e-9),
          f"{base_now.net_gex / 1e6:,.1f}mn -> {inv.net_gex / 1e6:,.1f}mn")
    check("mirroring swaps call and put gamma",
          abs(inv.call_gex + base_now.call_gex) < max(1.0, abs(base_now.call_gex) * 1e-9)
          and abs(inv.put_gex + base_now.put_gex) < max(1.0, abs(base_now.put_gex) * 1e-9))
    check("re-running the same config is deterministic",
          abs(base_now.net_gex - base.net_gex) / max(abs(base.net_gex), 1.0) < 0.05,
          f"{base.net_gex / 1e6:,.1f}mn vs {base_now.net_gex / 1e6:,.1f}mn")

    # "all_short" must be strictly negative - a dealer short every contract cannot
    # be long gamma anywhere.
    cfg_short = _dc.replace(g["CFG"], dealer_convention="all_short")
    allshort = g["compute_gex"](com, cfg=cfg_short, asof=pinned)
    check("all_short == -(calls + puts) of the base run",
          abs(allshort.net_gex + (base_now.call_gex - base_now.put_gex)) < max(1.0, abs(allshort.net_gex) * 1e-9))
    check("all_short convention is unambiguously negative gamma",
          allshort.net_gex < 0 and allshort.profile["net_gex"].max() <= 0,
          f"{allshort.net_gex / 1e6:,.1f}mn")
    check("all_short has no gamma flip", allshort.gamma_flip is None)

    # --- Black-76 path must run and be close to Black-Scholes ATM -------------
    import pandas as pd

    chain = base.chain[["strike", "option_type", "openInterest", "expiry_epoch"]].copy()
    chain["impliedVolatility"] = base.chain["iv"]
    chain["bid"] = np.nan
    chain["ask"] = np.nan
    csv = os.path.join(HERE, "_test_futures_chain.csv")
    chain.to_csv(csv, index=False)
    f76 = g["load_futures_chain"](csv, "gold", futures_price=base.underlying_spot)
    check("Black-76 adapter runs", f76.contracts_used > 0, f"{f76.contracts_used} contracts")
    check("Black-76 uses futures multiplier", f76.source == "futures")
    # Same inputs, same sign, and gamma within the discounting difference.
    check("Black-76 sign matches Black-Scholes", np.sign(f76.net_gex) == np.sign(base.net_gex))
    ratio = abs(f76.net_gex) / abs(base.net_gex * com.point_value / 100.0)
    check("Black-76 magnitude consistent after multiplier", 0.9 < ratio < 1.1, f"ratio {ratio:.4f}")
    os.remove(csv)

    # --- Snapshot persistence --------------------------------------------------
    path = g["save_snapshot"](RESULTS)
    hist = g["load_history"]()
    check("snapshot written", os.path.exists(path))
    check("history reloads all rows", len(hist) == len(RESULTS), f"{len(hist)} rows")
    # Saving twice on the same day must not duplicate.
    g["save_snapshot"](RESULTS)
    check("re-saving same day is idempotent", len(g["load_history"]()) == len(RESULTS))

    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("\nFAILURES:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 74)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
