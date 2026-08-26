
# %%
# =============================================================================
# CELL 4 - The commodity universe
# =============================================================================
# Each entry ties together three things:
#   * the CME futures contract we want to quote levels in,
#   * the ETF whose listed options we actually read gamma from,
#   * the contract economics needed to convert gamma into dollars.
#
# `point_value` = dollar value of a 1.00 move in the *quoted* price unit.
#   Gold  is quoted in $/oz,      contract = 100 oz    -> $100 per $1.00
#   Corn  is quoted in cents/bu,  contract = 5,000 bu  -> $50  per 1.00 cent
# =============================================================================


@dataclass(frozen=True)
class Commodity:
    key: str                 # short internal id
    name: str                # display name
    futures_symbol: str      # Yahoo continuous-futures symbol
    cme_root: str            # CME product root (for the real-futures adapter)
    etf: str                 # options proxy ETF
    point_value: float       # $ per 1.00 of quoted price, per futures contract
    price_unit: str          # how the futures price is quoted
    sector: str
    contract_size: str

    @property
    def label(self) -> str:
        return f"{self.name} ({self.cme_root})"


UNIVERSE: dict[str, Commodity] = {
    c.key: c
    for c in [
        # --- Precious & base metals ----------------------------------------
        Commodity("gold",     "Gold",          "GC=F", "GC", "GLD",   100.0, "$/troy oz", "Metals", "100 troy oz"),
        Commodity("silver",   "Silver",        "SI=F", "SI", "SLV",  5000.0, "$/troy oz", "Metals", "5,000 troy oz"),
        Commodity("copper",   "Copper",        "HG=F", "HG", "CPER", 25000.0, "$/lb",     "Metals", "25,000 lb"),
        Commodity("platinum", "Platinum",      "PL=F", "PL", "PPLT",   50.0, "$/troy oz", "Metals", "50 troy oz"),
        # --- Energy ---------------------------------------------------------
        Commodity("crude",    "WTI Crude Oil", "CL=F", "CL", "USO",  1000.0, "$/barrel",  "Energy", "1,000 barrels"),
        Commodity("brent",    "Brent Crude",   "BZ=F", "BZ", "BNO",  1000.0, "$/barrel",  "Energy", "1,000 barrels"),
        Commodity("natgas",   "Natural Gas",   "NG=F", "NG", "UNG", 10000.0, "$/MMBtu",   "Energy", "10,000 MMBtu"),
        Commodity("gasoline", "RBOB Gasoline", "RB=F", "RB", "UGA", 42000.0, "$/gallon",  "Energy", "42,000 gallons"),
        # --- Grains & softs --------------------------------------------------
        Commodity("corn",     "Corn",          "ZC=F", "ZC", "CORN",   50.0, "cents/bu",  "Grains", "5,000 bushels"),
        Commodity("wheat",    "Chicago Wheat", "ZW=F", "ZW", "WEAT",   50.0, "cents/bu",  "Grains", "5,000 bushels"),
        Commodity("soybeans", "Soybeans",      "ZS=F", "ZS", "SOYB",   50.0, "cents/bu",  "Grains", "5,000 bushels"),
    ]
}

# Commodities the dashboard runs by default. Add or remove keys freely.
DEFAULT_WATCHLIST = ["gold", "crude", "natgas", "corn", "wheat", "silver", "copper", "soybeans"]

print(f"{len(UNIVERSE)} commodities defined.")
print("Default watchlist:", ", ".join(DEFAULT_WATCHLIST))

pd.DataFrame(
    [
        {
            "key": c.key,
            "commodity": c.name,
            "futures": c.cme_root,
            "ETF": c.etf,
            "contract": c.contract_size,
            "quoted in": c.price_unit,
            "$ per 1.00 move": f"{c.point_value:,.0f}",
        }
        for c in UNIVERSE.values()
    ]
)

# %%
# =============================================================================
# CELL 5 - Market data client (Yahoo Finance)
# =============================================================================
# Yahoo public endpoints need a cookie + crumb pair. We fetch one per session,
# refresh it automatically on a 401, and retry with exponential backoff to ride
# out the rate limits that Colab shared IPs regularly run into.
# =============================================================================

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


class DataError(RuntimeError):
    """Raised when market data cannot be retrieved after retries."""


class YahooClient:
    """Minimal, dependency-free Yahoo Finance client for quotes and option chains."""

    CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"
    CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    OPTS_URL = "https://query2.finance.yahoo.com/v7/finance/options/{sym}"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._jar)
        )
        self._crumb: str | None = None
        self._cache: dict[str, tuple[float, Any]] = {}

    # --------------------------------------------------------------- internals
    def _raw_get(self, url: str, headers: dict | None = None) -> bytes:
        hdrs = {
            "User-Agent": _UA,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if headers:
            hdrs.update(headers)
        req = urllib.request.Request(url, headers=hdrs)
        resp = self._opener.open(req, timeout=self.cfg.request_timeout)
        data = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        return data

    def _ensure_crumb(self, force: bool = False) -> str:
        if self._crumb and not force:
            return self._crumb
        # Priming request seeds the consent cookie. It 404s by design - ignore it.
        for primer in ("https://fc.yahoo.com/", "https://finance.yahoo.com/quote/SPY"):
            try:
                self._raw_get(primer)
                break
            except Exception:
                continue
        self._crumb = self._raw_get(self.CRUMB_URL).decode().strip()
        if not self._crumb or "<" in self._crumb:
            raise DataError("Could not obtain a Yahoo crumb token.")
        return self._crumb

    def _get_json(self, url: str, use_crumb: bool = False) -> dict:
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                target = url
                if use_crumb:
                    crumb = self._ensure_crumb(force=attempt > 0)
                    sep = "&" if "?" in url else "?"
                    target = f"{url}{sep}crumb={urllib.parse.quote(crumb)}"
                return json.loads(self._raw_get(target))
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code in (401, 403):
                    self._crumb = None          # force a fresh crumb next loop
                elif exc.code not in (429, 500, 502, 503, 504):
                    raise DataError(f"HTTP {exc.code} for {url}") from exc
            except Exception as exc:
                last = exc
            time.sleep(self.cfg.retry_backoff ** attempt)
        raise DataError(f"Failed after {self.cfg.max_retries} attempts: {url} ({last})")

    def _cached(self, key: str, producer: Callable[[], Any]) -> Any:
        ttl = self.cfg.cache_ttl_seconds
        if ttl:
            hit = self._cache.get(key)
            if hit and (time.time() - hit[0]) < ttl:
                return hit[1]
        value = producer()
        self._cache[key] = (time.time(), value)
        return value

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------ public
    def quote(self, symbol: str) -> dict:
        """Last price and session metadata for any Yahoo symbol."""

        def _fetch() -> dict:
            url = self.CHART_URL.format(sym=urllib.parse.quote(symbol)) + "?range=5d&interval=1d"
            meta = self._get_json(url)["chart"]["result"][0]["meta"]
            px = float(meta["regularMarketPrice"])
            return {
                "symbol": symbol,
                "price": px,
                "previous_close": float(meta.get("chartPreviousClose") or px),
                "currency": meta.get("currency", "USD"),
                "name": meta.get("shortName", symbol),
                "exchange": meta.get("fullExchangeName", ""),
                "timestamp": dt.datetime.fromtimestamp(
                    meta.get("regularMarketTime", time.time()), dt.timezone.utc
                ),
            }

        return self._cached(f"q:{symbol}", _fetch)

    def daily_history(self, symbol: str, lookback_days: int = 400) -> pd.DataFrame:
        """Daily close and volume indexed by calendar date.

        Equities and Globex futures stamp their bars in different sessions, so we
        key on the UTC calendar date. Joining on raw epoch timestamps silently
        produces an almost-empty intersection - which is exactly the bug that
        makes a naive ETF/futures regression report a bogus R-squared of 1.0 on
        a sample of two points.
        """

        def _fetch() -> pd.DataFrame:
            rng = "2y" if lookback_days > 300 else "1y"
            url = self.CHART_URL.format(sym=urllib.parse.quote(symbol)) + f"?range={rng}&interval=1d"
            res = self._get_json(url)["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            rows = []
            for ts, px, vol in zip(res["timestamp"], q["close"], q.get("volume", [])):
                if px is None:
                    continue
                rows.append(
                    {
                        "date": dt.datetime.fromtimestamp(ts, dt.timezone.utc).date(),
                        "close": float(px),
                        "volume": float(vol or 0.0),
                    }
                )
            if not rows:
                raise DataError(f"No price history for {symbol}")
            return pd.DataFrame(rows).set_index("date").sort_index()

        return self._cached(f"h:{symbol}:{lookback_days}", _fetch)

    def expirations(self, symbol: str) -> list[int]:
        """Every listed expiry for a symbol, as epoch seconds."""

        def _fetch() -> list[int]:
            url = self.OPTS_URL.format(sym=urllib.parse.quote(symbol))
            res = self._get_json(url, use_crumb=True)["optionChain"]["result"]
            if not res:
                raise DataError(f"No option chain listed for {symbol}")
            return list(res[0]["expirationDates"])

        return self._cached(f"e:{symbol}", _fetch)

    def option_chain(self, symbol: str, expiry_epoch: int | None = None) -> tuple[pd.DataFrame, float]:
        """Return (chain dataframe, underlying spot) for one expiry."""

        def _fetch() -> tuple[pd.DataFrame, float]:
            url = self.OPTS_URL.format(sym=urllib.parse.quote(symbol))
            if expiry_epoch is not None:
                url += f"?date={int(expiry_epoch)}"
            res = self._get_json(url, use_crumb=True)["optionChain"]["result"]
            if not res:
                raise DataError(f"Empty option chain for {symbol}")
            node = res[0]
            spot = float(node["quote"]["regularMarketPrice"])
            frames = []
            for block in node.get("options", []):
                for side in ("calls", "puts"):
                    rows = block.get(side, [])
                    if not rows:
                        continue
                    df = pd.DataFrame(rows)
                    df["option_type"] = side[:-1]      # "call" / "put"
                    df["expiry_epoch"] = block["expirationDate"]
                    frames.append(df)
            if not frames:
                raise DataError(f"No contracts returned for {symbol}")
            return pd.concat(frames, ignore_index=True), spot

        return self._cached(f"c:{symbol}:{expiry_epoch}", _fetch)


CLIENT = YahooClient(CFG)

# Smoke test - proves auth works before anything expensive runs.
_t = CLIENT.quote("GC=F")
print(f"Data client live. {_t['name']}: {_t['price']:,.2f}  ({_t['timestamp']:%Y-%m-%d %H:%M UTC})")
