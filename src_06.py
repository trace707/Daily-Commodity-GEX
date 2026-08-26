
# %%
# =============================================================================
# CELL 10 - Charts
# =============================================================================
# All strike levels are plotted in FUTURES price terms via the Cell 8 map, so the
# axes read directly against a GC / CL / ZC chart.
#
# Colour follows a validated palette rather than trading-screen convention:
#
#   * Net gamma is a DIVERGING quantity (dealers long vs short), so it uses a
#     warm/cool diverging pair - blue for positive, red for negative, neutral grey
#     at the midpoint. Green/red was rejected: it is the classic red-green
#     colour-vision failure and the two poles do not separate under deuteranopia.
#   * Spot and the gamma flip are reference lines, not series, so they wear chart
#     chrome (ink and one reserved accent) rather than a series colour.
#   * Light and dark are each stepped for their own surface. Neither is an
#     automatic inversion of the other.
# =============================================================================

# Validated palette. Substitute your brand's values wholesale; do not hand-tune
# individual entries, because the pairwise separation is what was checked.
PALETTES = {
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "pos": "#3987e5",        # diverging pole - dealers long gamma
        "neg": "#e66767",        # diverging pole - dealers short gamma
        "neutral": "#383835",    # diverging midpoint - reads as "nothing"
        "spot": "#ffffff",       # reference line: you-are-here
        "flip": "#9085e9",       # reference line: regime threshold
        "template": "plotly_dark",
    },
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "pos": "#2a78d6",
        "neg": "#e34948",
        "neutral": "#f0efec",
        "spot": "#0b0b0b",
        "flip": "#4a3aa7",
        "template": "plotly_white",
    },
}


def P() -> dict:
    """The active palette. Set CFG.mode to 'light' or 'dark'."""
    return PALETTES[getattr(CFG, "mode", "dark")]


def _scale(value: float, mode: str = "auto") -> tuple[float, str]:
    """Pick a display scale for dollar amounts."""
    mag = abs(value)
    if mode == "billions" or (mode == "auto" and mag >= 1e9):
        return 1e9, "$bn"
    if mode == "millions" or (mode == "auto" and mag >= 1e6):
        return 1e6, "$mn"
    if mode == "auto" and mag >= 1e3:
        return 1e3, "$k"
    return 1.0, "$"


def fmt_money(value: float, mode: str = "auto") -> str:
    div, unit = _scale(value, mode)
    return f"{value / div:,.2f} {unit}"


def fmt_price(com: Commodity, px: float | None) -> str:
    if px is None or not np.isfinite(px):
        return "n/a"
    if com.price_unit == "cents/bu":
        return f"{px:,.2f}c"
    if px < 10:
        return f"${px:,.3f}"
    return f"${px:,.2f}"


def _layout(fig: go.Figure, title: str, subtitle: str = "", height: int = 620) -> go.Figure:
    p = P()
    fig.update_layout(
        template=p["template"],
        height=height,
        paper_bgcolor=p["surface"],
        plot_bgcolor=p["surface"],
        title=dict(
            text=f"<b>{title}</b>"
            + (f"<br><span style='font-size:12px;color:{p['muted']}'>{subtitle}</span>" if subtitle else ""),
            x=0.01,
            xanchor="left",
            font=dict(size=17, color=p["ink"]),
        ),
        margin=dict(l=76, r=48, t=96, b=64),
        hovermode="closest",
        hoverlabel=dict(bgcolor=p["page"], bordercolor=p["axis"],
                        font=dict(color=p["ink"], size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    font=dict(color=p["ink_secondary"], size=12)),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  color=p["ink_secondary"], size=12),
    )
    # Hairline, solid, one shade off the surface. Never dashed - dashing a grid
    # reads as "threshold" when it is only a grid.
    for axis in (fig.update_xaxes, fig.update_yaxes):
        axis(gridcolor=p["grid"], griddash="solid", linecolor=p["axis"],
             zerolinecolor=p["axis"], tickfont=dict(color=p["muted"], size=11),
             title_font=dict(color=p["ink_secondary"], size=12))
    return fig


def _refline(fig, level, color, label, horizontal: bool, dash="dash", row=None, col=None):
    """Reference lines carry chrome colours, never a series colour."""
    if level is None or not np.isfinite(level):
        return
    kw = dict(line=dict(color=color, width=2, dash=dash),
              annotation_text=label,
              annotation_font=dict(color=color, size=11))
    if row is not None:
        kw["row"], kw["col"] = row, col
    if horizontal:
        fig.add_hline(y=level, annotation_position="right", **kw)
    else:
        fig.add_vline(x=level, annotation_position="top", **kw)


def chart_gex_by_strike(res: GexResult, top_n: int = 40) -> go.Figure:
    """Diverging bars per strike - the gamma walls."""
    p = P()
    com = res.commodity
    bs = res.by_strike.copy()
    bs["abs"] = bs["net_gex"].abs()
    bs = bs.nlargest(min(top_n, len(bs)), "abs").sort_values("strike")
    bs["fut_strike"] = [res.mapping.to_futures(k) for k in bs["strike"]]

    div, unit = _scale(bs[["call_gex", "put_gex"]].abs().to_numpy().max(), CFG.scale_label)

    fig = go.Figure()
    for name, col, oi_col, colour in [
        ("Call gamma", "call_gex", "call_oi", p["pos"]),
        ("Put gamma", "put_gex", "put_oi", p["neg"]),
    ]:
        fig.add_trace(go.Bar(
            y=bs["fut_strike"], x=bs[col] / div, orientation="h", name=name,
            marker=dict(
                color=colour,
                # A 2px surface gap between adjacent fills - not a border around marks.
                line=dict(color=p["surface"], width=1.5),
            ),
            customdata=np.stack([bs["strike"], bs[oi_col], bs["avg_iv"] * 100], axis=-1),
            hovertemplate=(
                f"<b>{com.cme_root} %{{y:,.2f}}</b><br>"
                f"{res.underlying_symbol} strike %{{customdata[0]:,.2f}}<br>"
                f"{name} %{{x:,.2f}} {unit}<br>OI %{{customdata[1]:,.0f}}<br>"
                "Avg IV %{customdata[2]:.1f}%<extra></extra>"
            ),
        ))

    _refline(fig, res.futures_price, p["spot"], f"spot {fmt_price(com, res.futures_price)}", True, "solid")
    _refline(fig, res.gamma_flip_futures, p["flip"], f"gamma flip {fmt_price(com, res.gamma_flip_futures)}", True)

    fig.update_layout(barmode="relative", bargap=0.18)
    fig.update_xaxes(title=f"Dealer gamma exposure per 1% move ({unit})")
    fig.update_yaxes(title=f"{com.cme_root} strike ({com.price_unit})")
    return _layout(
        fig,
        f"{com.label} - gamma by strike",
        f"{res.source.replace('_', ' ')} - {res.contracts_used:,} contracts, "
        f"{res.total_oi:,} OI - as of {res.asof:%Y-%m-%d %H:%M UTC}",
        height=max(600, 21 * len(bs)),
    )


def chart_gamma_profile(res: GexResult) -> go.Figure:
    """Net dealer gamma re-evaluated across hypothetical spot levels."""
    p = P()
    com = res.commodity
    prof = res.profile.copy()
    prof["fut_spot"] = [res.mapping.to_futures(s) for s in prof["spot"]]
    div, unit = _scale(prof["net_gex"].abs().max(), CFG.scale_label)
    y = prof["net_gex"] / div

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prof["fut_spot"], y=y.clip(lower=0), mode="lines", name="Dealers long gamma",
        line=dict(width=0), fill="tozeroy", fillcolor="rgba(57,135,229,0.32)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=prof["fut_spot"], y=y.clip(upper=0), mode="lines", name="Dealers short gamma",
        line=dict(width=0), fill="tozeroy", fillcolor="rgba(230,103,103,0.32)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=prof["fut_spot"], y=y, mode="lines", name="Net GEX", showlegend=False,
        line=dict(color=p["ink_secondary"], width=2),
        hovertemplate=f"{com.cme_root} %{{x:,.2f}}<br>Net GEX %{{y:,.2f}} {unit}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=p["axis"], width=1))
    _refline(fig, res.futures_price, p["spot"], f"spot {fmt_price(com, res.futures_price)}", False, "solid")
    _refline(fig, res.gamma_flip_futures, p["flip"], f"flip {fmt_price(com, res.gamma_flip_futures)}", False)

    fig.update_xaxes(title=f"{com.cme_root} price ({com.price_unit})")
    fig.update_yaxes(title=f"Net dealer gamma per 1% move ({unit})")
    dist = ("" if not res.gamma_flip_futures
            else f" - flip is {(res.futures_price / res.gamma_flip_futures - 1) * 100:+.1f}% from spot")
    return _layout(fig, f"{com.label} - gamma profile", f"{res.regime}{dist}", height=500)


def chart_term_structure(res: GexResult) -> go.Figure:
    """How the gamma is distributed across expiries."""
    p = P()
    com = res.commodity
    df = res.chain.copy()
    df["expiry_date"] = df["expiry"].dt.date
    is_call = df["option_type"] == "call"
    agg = (
        df.assign(c=df["gex"].where(is_call, 0.0), pu=df["gex"].where(~is_call, 0.0))
        .groupby("expiry_date", as_index=False)
        .agg(net=("gex", "sum"), call=("c", "sum"), put=("pu", "sum"), oi=("openInterest", "sum"))
        .sort_values("expiry_date")
    )
    div, unit = _scale(agg[["call", "put"]].abs().to_numpy().max(), CFG.scale_label)
    x = [str(d) for d in agg["expiry_date"]]

    fig = go.Figure()
    for name, col, colour in [("Call gamma", "call", p["pos"]), ("Put gamma", "put", p["neg"])]:
        fig.add_trace(go.Bar(
            x=x, y=agg[col] / div, name=name,
            marker=dict(color=colour, line=dict(color=p["surface"], width=1.5)),
            hovertemplate="%{x}<br>" + name + " %{y:,.2f} " + unit + "<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=x, y=agg["net"] / div, name="Net", mode="lines+markers",
        line=dict(color=p["ink_secondary"], width=2), marker=dict(size=8),
        hovertemplate="%{x}<br>Net GEX %{y:,.2f} " + unit + "<extra></extra>",
    ))
    fig.update_layout(barmode="relative", bargap=0.3)
    fig.update_xaxes(title="Expiry", type="category")
    fig.update_yaxes(title=f"Dealer gamma per 1% move ({unit})")
    return _layout(fig, f"{com.label} - gamma term structure",
                   "Front expiries dominate and decay fastest as they roll off", height=440)


def chart_cross_commodity(results: dict[str, GexResult]) -> go.Figure:
    """One bar per commodity - who is long gamma, who is short."""
    p = P()
    rows = []
    for r in results.values():
        rows.append({
            "name": r.commodity.name,
            "root": r.commodity.cme_root,
            "net": r.net_gex / 1e6,
            "flip_dist": ((r.futures_price / r.gamma_flip_futures - 1) * 100
                          if r.gamma_flip_futures else np.nan),
            "iv": r.avg_iv * 100,
        })
    df = pd.DataFrame(rows).sort_values("net")

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.56, 0.44], horizontal_spacing=0.14,
        subplot_titles=("Net dealer gamma per 1% move ($mn)", "Spot distance to gamma flip (%)"),
    )
    fig.add_trace(go.Bar(
        y=df["root"], x=df["net"], orientation="h", showlegend=False,
        marker=dict(color=[p["pos"] if v >= 0 else p["neg"] for v in df["net"]],
                    line=dict(color=p["surface"], width=1.5)),
        text=[f"{v:,.1f}" for v in df["net"]], textposition="outside",
        textfont=dict(color=p["ink_secondary"], size=11),
        customdata=np.stack([df["name"], df["iv"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>Net GEX %{x:,.2f} $mn<br>"
                      "OI-weighted IV %{customdata[1]:.1f}%<extra></extra>",
    ), row=1, col=1)

    d2 = df.dropna(subset=["flip_dist"]).sort_values("flip_dist")
    fig.add_trace(go.Bar(
        y=d2["root"], x=d2["flip_dist"], orientation="h", showlegend=False,
        marker=dict(color=[p["pos"] if v >= 0 else p["neg"] for v in d2["flip_dist"]],
                    line=dict(color=p["surface"], width=1.5)),
        text=[f"{v:+.1f}%" for v in d2["flip_dist"]], textposition="outside",
        textfont=dict(color=p["ink_secondary"], size=11),
        hovertemplate="<b>%{y}</b><br>Spot is %{x:+.1f}% from the flip<extra></extra>",
    ), row=1, col=2)

    for c in (1, 2):
        fig.add_vline(x=0, line=dict(color=p["axis"], width=1), row=1, col=c)
    for ann in fig.layout.annotations:
        ann.font = dict(color=p["ink_secondary"], size=12)

    return _layout(fig, "Commodity complex - dealer gamma at a glance",
                   "Blue = dealers long gamma (volatility damped)   |   "
                   "Red = dealers short gamma (volatility amplified)",
                   height=max(420, 46 * len(df)))


print("Charts ready.")
