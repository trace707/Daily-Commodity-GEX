
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
