# Commodity Futures GEX Dashboard

Dealer gamma exposure for Gold, Crude, Natural Gas, Corn, Wheat, Silver, Copper,
Soybeans, Platinum, Brent and RBOB — with gamma flip levels and call/put walls
quoted in **futures price terms** (GC $/oz, ZC cents/bu, and so on).

Two ways to run it. No API keys, no paid data, no accounts required for the local path.

---

## Quick start — on this PC, no Colab

```bash
py run_gex.py --open
```

Builds the whole dashboard and opens `output/gex_dashboard.html` — a single
self-contained file (charts embedded, works offline, ~4.7 MB). Takes about 30
seconds for eight commodities.

To have it rebuild itself every weekday afternoon:

```bash
powershell -ExecutionPolicy Bypass -File setup_schedule.ps1
```

Preview what that would register without touching anything:

```bash
powershell -ExecutionPolicy Bypass -File setup_schedule.ps1 -WhatIfOnly
```

Remove it later with `-Remove`.

## Quick start — Google Colab

Upload `commodity_gex_dashboard.ipynb`, then `Runtime > Run all`. Snapshots persist
to Google Drive. Everything before Cell 15 is definitions; Cell 15 is the run.

## Quick start — GitHub Actions (dashboard on the web)

The repo is committed and the workflow is ready. Create an **empty private repo**
on GitHub (no README, no .gitignore), then:

```bash
git remote add origin https://github.com/<you>/commodity-gex.git
git push -u origin main
```

Then, once, in the repo's **Settings → Pages**, set **Source = GitHub Actions**.

Trigger the first run from the **Actions** tab → *Daily commodity GEX* → *Run
workflow*. After it succeeds the dashboard is live at
`https://<you>.github.io/commodity-gex/` and rebuilds every weekday at 21:30 UTC.

**A caveat worth knowing up front:** Yahoo rate-limits datacenter IPs, and GitHub
runners are datacenter IPs. This may work fine, or it may 429 intermittently — it
cannot be tested from a local machine. The workflow passes `--min-success 5`, so a
run that only retrieves a few commodities **fails loudly instead of publishing a
report that looks complete**. Your Windows scheduled task is the reliable path;
treat CI as the convenience layer for reading it off your phone.

---

## Files

| file | what it is |
|---|---|
| `commodity_gex_dashboard.ipynb` | The Colab notebook. |
| `run_gex.py` | **Headless runner** → self-contained HTML dashboard. No Jupyter needed. |
| `setup_schedule.ps1` | Registers/removes the weekday Windows Scheduled Task. |
| `.github/workflows/daily-gex.yml` | Free-tier CI that runs it daily and publishes to GitHub Pages. |
| `.gitignore` / `.gitattributes` | Keeps the 4.5 MB reports out of history; pins LF for the Linux runner. |
| `requirements.txt` | Dependencies. |
| `commodity_gex_dashboard.py` | The notebook as a flat script (what the runner and CI execute). |
| `src_01.py` … `src_10.py` | Cell sources. **Edit these, not the `.ipynb`.** |
| `build_notebook.py` | Rebuilds the `.ipynb` and the flat script from `src_*.py`. |
| `test_dashboard.py` | 108-assertion end-to-end test against live data. |
| `output/` | Generated: HTML report, dated archive, snapshot CSVs. |

Workflow after editing any `src_*.py`:

```bash
py build_notebook.py && py test_dashboard.py
```

---

## The GEX calculation

Gamma comes from **Black-Scholes** for ETF options and **Black-76** for real
futures options (the model CME itself settles GC/CL/NG/ZC/ZW options with):

```
Black-Scholes    Γ = e^(-qT)·φ(d₁) / (S·σ·√T)
Black-76         Γ = e^(-rT)·φ(d₁) / (F·σ·√T)
```

Converting gamma into dollars of hedging flow:

1. A move of `ΔS` shifts each option's delta by `Γ·ΔS`.
2. One contract covers `M` units (100 shares for an ETF option, 1,000 barrels for CL,
   5,000 bushels for ZC), so the position holds `OI × M` units.
3. Dollar notional of that delta change: `Γ·ΔS·OI·M·S`.
4. Set `ΔS = 0.01·S`:

```
GEX_strike = Γ × OI × M × S² × 0.01

NetGEX = Σ_calls − Σ_puts        (default dealer convention)
```

Read as **dollars of delta dealers must trade per 1% move**. Gamma is verified
against a numerical second derivative of price; the IV solver round-trips to 1e-4.

The **gamma flip** is found by re-evaluating the whole book across a grid of
hypothetical spot levels and interpolating the zero crossing.

---

## Data sources, and an honest caveat

CME publishes no free, stable API for futures-options open interest — its endpoints
sit behind bot protection and reject datacenter IPs, which is exactly what Colab and
GitHub Actions run on. Verified: every CME settlement and quote endpoint returns HTTP 403.

So GEX is computed from **listed options on the commodity ETFs** (GLD, USO, UNG,
CORN, WEAT, …), which are free and update daily. The **dollar magnitudes are genuine
ETF dealer gamma**; the **price levels are translated into futures terms**.

If you have real futures-options data, the Cell 9 adapter takes a CSV and recomputes
everything on true chains via Black-76:

```python
res = load_futures_chain("my_gc_options.csv", "gold")
```

Required columns: `strike`, `option_type`, `openInterest`, `expiry_epoch` (or
`expiry` as `YYYY-MM-DD`), and either `impliedVolatility` or `bid`/`ask`.

### Why levels are mapped by return beta, not by regressing levels

Level regression is spurious for commodity ETFs — USO and UNG bleed on contango, so
their level relationship to front-month futures drifts, and the fit gets *worse* the
longer you look:

| pair | level R², 1y | level R², 6m | **return corr** |
|---|---|---|---|
| GLD / GC | 0.997 | 0.995 | 0.92 |
| WEAT / ZW | 0.966 | 0.966 | 0.95 |
| CPER / HG | 0.996 | 0.987 | 0.91 |
| USO / CL | 0.896 | **0.613** | 0.90 |
| UNG / NG | 0.604 | **0.578** | **0.63** |

Daily returns track closely even where levels do not, so strikes map through an
anchored log-return beta:

```
K_fut = F₀ · (K_etf / E₀) ^ β
```

Exact at the money by construction, and the roll drag cancels. The `Map` column in
the summary table grades each name — natural gas is the weak one.

---

## Chart conventions

Net gamma is a **diverging** quantity, so it uses a warm/cool diverging pair —
**blue for dealers long gamma, red for short**, with a neutral midpoint. Green/red
was rejected: it is the classic red-green colour-vision failure and the two poles do
not separate under deuteranopia. Spot and the gamma flip are reference lines, so they
wear chart chrome rather than a series colour.

The GEX-history chart deliberately uses **two stacked panels sharing one x-axis
rather than a dual y-axis**. Overlaying GEX (dollars) on price (dollars per ounce)
with two independently-scaled axes invents a correlation that isn't in the data.

---

## Timing

Run **after** the 16:00 ET close. Open interest is an end-of-day figure; OI printed
intraday is yesterday's number, so a pre-close run pairs stale OI with fresh spot.
The scheduled task defaults to 17:15 local and the GitHub Action to 21:30 UTC.

Each run appends one row per commodity to `output/data/gex_history.csv` and dumps
per-strike detail to `output/data/chains/`. Once you have two or more saved days,
`chart_history('gold')` plots net GEX against price. **Regime changes are the signal,
not the level.**

---

## Limitations

1. **The dealer sign convention is assumed, not measured.** Nobody outside the
   clearing houses observes true dealer inventory. `convention_sensitivity('gold')`
   recomputes under all four conventions — if your conclusion inverts, you were
   reading the assumption, not the data. This caveat applies to every published GEX
   number, commercial ones included.
2. **ETF options proxy futures options.** The shape (where gamma sits relative to
   spot) transfers; the absolute size does not — CME books are larger.
3. **Level translation degrades with correlation.** Natural gas maps at ~0.63.
4. **The gamma profile holds the vol surface fixed.** Real IV rises as spot falls, so
   the true flip usually sits slightly below the computed one.
5. **Grain ETF chains are thin.** CORN and WEAT carry a few thousand contracts across
   ~20 strikes; SOYB is thinner still. Structure is real but coarse.
6. **Yahoo is the single point of failure.** If it rate-limits, the run retries with
   backoff and then skips that commodity rather than failing the whole report.

Not investment advice.
