"""Headless runner - builds the commodity GEX dashboard without Jupyter or Colab.

Produces a single self-contained HTML file (charts embedded, works offline) plus
the rolling snapshot CSVs. This is what a scheduler should call.

    py run_gex.py                        # default watchlist -> output/
    py run_gex.py --open                 # ...and open it in the browser
    py run_gex.py --watchlist gold,crude
    py run_gex.py --mode light
    py run_gex.py --outdir "D:/gex"
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import os
import sys
import traceback
import types
import webbrowser

# Windows Python cannot verify HTTPS against the OS certificate store on its own.
# Colab and Linux CI do not need this, so a missing truststore is not an error.
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "commodity_gex_dashboard.py")
SENTINEL = "# CELL 15 - RUN"


def load_notebook_module(mode: str, outdir: str) -> types.ModuleType:
    """Exec the notebook's definition cells, stopping before the run cells."""
    if not os.path.exists(SCRIPT):
        raise SystemExit(
            f"{SCRIPT} not found. Run `py build_notebook.py` first."
        )
    src = open(SCRIPT, encoding="utf-8").read()
    if SENTINEL not in src:
        raise SystemExit(f"Sentinel {SENTINEL!r} missing from {SCRIPT} - cannot split definitions.")
    src = src.split(SENTINEL)[0]

    src = src.replace(
        'snapshot_dir: str = "/content/drive/MyDrive/commodity_gex"',
        f'snapshot_dir: str = {os.path.join(outdir, "data")!r}',
    )
    src = src.replace('mode: str = "dark"', f'mode: str = {mode!r}')

    mod = types.ModuleType("gex_headless")
    mod.__file__ = SCRIPT
    mod.__dict__["display"] = lambda *a, **k: None
    # dataclasses resolves cls.__module__ through sys.modules.
    sys.modules["gex_headless"] = mod
    exec(compile(src, SCRIPT, "exec"), mod.__dict__)
    return mod


# --------------------------------------------------------------------------- HTML


def _tile(label: str, value: str, sub: str, accent: str, palette: dict) -> str:
    return f"""
    <div class="tile">
      <div class="tile-label">{html.escape(label)}</div>
      <div class="tile-value" style="color:{accent}">{html.escape(value)}</div>
      <div class="tile-sub">{html.escape(sub)}</div>
    </div>"""


def build_html(mod, results: dict, mode: str, asof: dt.datetime) -> str:
    g = mod.__dict__
    p = g["PALETTES"][mode]
    fmt_money, fmt_price = g["fmt_money"], g["fmt_price"]

    # --- headline tiles -------------------------------------------------------
    total_net = sum(r.net_gex for r in results.values())
    n_long = sum(1 for r in results.values() if r.net_gex >= 0)
    n_short = len(results) - n_long
    tiles = [
        _tile("Complex net gamma", fmt_money(total_net, "millions"), "per 1% move",
              p["pos"] if total_net >= 0 else p["neg"], p),
        _tile("Dealers long gamma", str(n_long), "volatility damped", p["pos"], p),
        _tile("Dealers short gamma", str(n_short), "volatility amplified", p["neg"], p),
        _tile("Commodities", str(len(results)), "loaded this run", p["ink"], p),
    ]

    # --- summary table ---------------------------------------------------------
    summary = g["summary_table"](results)
    thead = "".join(f"<th>{html.escape(c)}</th>" for c in summary.columns)
    rows = []
    for _, row in summary.iterrows():
        cells = []
        for col in summary.columns:
            val = str(row[col])
            style = ""
            if col == "Regime":
                style = f' style="color:{p["pos"] if "LONG" in val else p["neg"]};font-weight:600"'
            cells.append(f"<td{style}>{html.escape(val)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    table = f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

    # --- figures ---------------------------------------------------------------
    blocks: list[str] = []
    first = True

    def add(fig) -> None:
        nonlocal first
        blocks.append(fig.to_html(
            include_plotlyjs="inline" if first else False,
            full_html=False,
            config={"displayModeBar": False, "responsive": True},
        ))
        first = False

    add(g["chart_cross_commodity"](results))

    sections = []
    for key, r in results.items():
        com = r.commodity
        flip_txt = (f"{(r.futures_price / r.gamma_flip_futures - 1) * 100:+.1f}% away"
                    if r.gamma_flip_futures else "no flip in range")
        start = len(blocks)
        add(g["chart_gex_by_strike"](r))
        add(g["chart_gamma_profile"](r))
        add(g["chart_term_structure"](r))
        sections.append({
            "key": key,
            "title": com.label,
            "regime": "LONG gamma" if r.net_gex >= 0 else "SHORT gamma",
            "colour": p["pos"] if r.net_gex >= 0 else p["neg"],
            "price": fmt_price(com, r.futures_price),
            "net": fmt_money(r.net_gex, "millions"),
            "flip": fmt_price(com, r.gamma_flip_futures),
            "flip_txt": flip_txt,
            "map": r.mapping.quality,
            "figs": blocks[start:],
        })

    section_html = []
    for s in sections:
        section_html.append(f"""
    <section class="commodity">
      <div class="c-head">
        <h2>{html.escape(s['title'])}</h2>
        <div class="c-meta">
          <span class="pill" style="background:{s['colour']}22;color:{s['colour']};border-color:{s['colour']}55">
            {html.escape(s['regime'])}
          </span>
          <span>spot <b>{html.escape(s['price'])}</b></span>
          <span>net GEX <b>{html.escape(s['net'])}</b></span>
          <span>flip <b>{html.escape(s['flip'])}</b> ({html.escape(s['flip_txt'])})</span>
          <span>level map <b>{html.escape(s['map'])}</b></span>
        </div>
      </div>
      {''.join(f'<div class="fig">{f}</div>' for f in s['figs'])}
    </section>""")

    warn_rows = []
    for r in results.values():
        for w in r.warnings:
            if "correlation" in w or "failed" in w:
                warn_rows.append(f"<li>{html.escape(w)}</li>")
    warn_block = (f'<div class="callout"><b>Data notes</b><ul>{"".join(warn_rows)}</ul></div>'
                  if warn_rows else "")

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Commodity GEX - {asof:%Y-%m-%d}</title>
<style>
  :root {{
    color-scheme: {mode};
    --page: {p['page']}; --surface: {p['surface']};
    --ink: {p['ink']}; --ink2: {p['ink_secondary']}; --muted: {p['muted']};
    --grid: {p['grid']}; --axis: {p['axis']};
    --pos: {p['pos']}; --neg: {p['neg']};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--page); color: var(--ink2);
    font: 14px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 1280px; margin: 0 auto; padding: 32px 24px 72px; }}
  header h1 {{ color: var(--ink); font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }}
  header .sub {{ color: var(--muted); font-size: 13px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 14px; margin: 26px 0 30px; }}
  .tile {{ background: var(--surface); border: 1px solid var(--grid);
           border-radius: 10px; padding: 16px 18px; }}
  .tile-label {{ color: var(--muted); font-size: 11.5px; text-transform: uppercase;
                 letter-spacing: 0.06em; }}
  .tile-value {{ font-size: 30px; font-weight: 600; margin: 6px 0 2px; letter-spacing: -0.02em; }}
  .tile-sub {{ color: var(--muted); font-size: 12px; }}
  h2 {{ color: var(--ink); font-size: 19px; margin: 0; }}
  h3 {{ color: var(--ink); font-size: 15px; margin: 34px 0 12px;
        text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}
  .table-wrap {{ overflow-x: auto; background: var(--surface);
                 border: 1px solid var(--grid); border-radius: 10px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12.5px;
           font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 9px 13px; text-align: right; white-space: nowrap;
            border-bottom: 1px solid var(--grid); }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 11px;
        text-transform: uppercase; letter-spacing: 0.05em; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  .fig {{ background: var(--surface); border: 1px solid var(--grid);
          border-radius: 10px; margin: 14px 0; overflow: hidden; }}
  .commodity {{ margin-top: 40px; padding-top: 26px; border-top: 1px solid var(--grid); }}
  .c-head {{ display: flex; flex-wrap: wrap; align-items: baseline;
             justify-content: space-between; gap: 12px; }}
  .c-meta {{ display: flex; flex-wrap: wrap; gap: 16px; color: var(--muted); font-size: 12.5px; }}
  .c-meta b {{ color: var(--ink2); font-weight: 600; }}
  .pill {{ border: 1px solid; border-radius: 999px; padding: 2px 10px;
           font-size: 11.5px; font-weight: 600; }}
  .callout {{ background: var(--surface); border: 1px solid var(--grid);
              border-left: 3px solid var(--neg); border-radius: 8px;
              padding: 14px 18px; margin: 22px 0; font-size: 12.5px; }}
  .callout ul {{ margin: 8px 0 0; padding-left: 20px; }}
  footer {{ margin-top: 46px; padding-top: 20px; border-top: 1px solid var(--grid);
            color: var(--muted); font-size: 12px; }}
  footer b {{ color: var(--ink2); }}
</style>
</head><body><div class="wrap">
<header>
  <h1>Commodity Futures GEX</h1>
  <div class="sub">Dealer gamma exposure &middot; {asof:%A %d %B %Y, %H:%M UTC} &middot;
    convention: {html.escape(g['CFG'].dealer_convention.replace('_', ' '))}</div>
</header>

<div class="tiles">{''.join(tiles)}</div>

<h3>Summary</h3>
<div class="table-wrap">{table}</div>
{warn_block}

<h3>Complex overview</h3>
<div class="fig">{blocks[0]}</div>

{''.join(section_html)}

<footer>
  <p><b>How to read it.</b> Positive net GEX means dealers are long gamma - their
  hedging sells rallies and buys dips, damping realised volatility. Negative means
  they are short gamma and hedging amplifies the move. The <b>gamma flip</b> is the
  price where that regime changes.</p>
  <p><b>Caveats.</b> Gamma is computed from commodity-ETF option chains; dollar
  magnitudes are genuine ETF dealer gamma, and price levels are translated into
  futures terms by an anchored return beta (the <i>Map</i> column grades each
  translation). The dealer sign convention is an assumption, not an observation.
  Open interest is end-of-day. Not investment advice.</p>
</footer>
</div></body></html>"""


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the commodity GEX dashboard headlessly.")
    ap.add_argument("--watchlist", default="", help="comma-separated keys, e.g. gold,crude,corn")
    ap.add_argument("--outdir", default=os.path.join(HERE, "output"))
    ap.add_argument("--mode", default="dark", choices=["dark", "light"])
    ap.add_argument("--no-save", action="store_true", help="skip appending to the history CSV")
    ap.add_argument("--open", action="store_true", help="open the report when done")
    ap.add_argument(
        "--min-success", type=int, default=0, metavar="N",
        help="exit non-zero unless at least N commodities returned data. Use this in "
             "CI: Yahoo rate-limits datacenter IPs, and a run that quietly yields two "
             "of eight commodities should fail the job rather than publish a report "
             "that looks complete.",
    )
    ap.add_argument("--no-archive", action="store_true",
                    help="skip the dated copy (CI does not need it)")
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    started = dt.datetime.now(dt.timezone.utc)
    print(f"[{started:%Y-%m-%d %H:%M:%S UTC}] commodity GEX run starting")

    mod = load_notebook_module(args.mode, outdir)
    g = mod.__dict__

    watchlist = ([w.strip() for w in args.watchlist.split(",") if w.strip()]
                 or g["DEFAULT_WATCHLIST"])
    unknown = [w for w in watchlist if w not in g["UNIVERSE"]]
    if unknown:
        raise SystemExit(f"Unknown commodity key(s): {unknown}. "
                         f"Valid: {sorted(g['UNIVERSE'])}")

    results = g["build_all"](watchlist, verbose=True)
    if not results:
        print("No commodities returned data. Nothing written.", file=sys.stderr)
        return 2

    if args.min_success and len(results) < args.min_success:
        print(
            f"\nOnly {len(results)}/{len(watchlist)} commodities returned data, "
            f"below --min-success={args.min_success}. Refusing to publish a partial "
            "dashboard. This usually means the data source rate-limited this IP.",
            file=sys.stderr,
        )
        return 3

    if not args.no_save:
        g["save_snapshot"](results)

    report = os.path.join(outdir, "gex_dashboard.html")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write(build_html(mod, results, args.mode, started))

    # Keep a dated copy so the history is browsable, not just the CSV. Each report
    # is ~4.5 MB, so this is deliberately opt-out for CI, which republishes rather
    # than accumulating.
    dated = None
    if not args.no_archive:
        dated = os.path.join(outdir, "archive", f"gex_{started:%Y-%m-%d}.html")
        os.makedirs(os.path.dirname(dated), exist_ok=True)
        with open(dated, "w", encoding="utf-8") as fh:
            fh.write(open(report, encoding="utf-8").read())

    size_mb = os.path.getsize(report) / 1e6
    elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    print(f"\nReport : {report}  ({size_mb:.1f} MB)")
    if dated:
        print(f"Archive: {dated}")
    print(f"Done in {elapsed:.0f}s - {len(results)}/{len(watchlist)} commodities.")

    if args.open:
        webbrowser.open("file:///" + report.replace("\\", "/"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
