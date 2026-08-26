
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
