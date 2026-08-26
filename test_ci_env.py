"""Guard against Windows-only assumptions that break the Linux CI runner.

The notebook is developed on Windows and executed on Ubuntu by GitHub Actions.
Two things differ there and both have already caused a red build:

  * `truststore` is a win32-only requirement, so it is absent on CI - and it
    happens to import `importlib.util` as a side effect, which masked a missing
    explicit import in Cell 2 for as long as it was only ever run on Windows.
  * Nothing pre-imports `importlib.util`, so any module relying on `import
    importlib` alone raises AttributeError.

This test loads the notebook's definition cells with truststore deliberately
blocked, which is the cheapest available approximation of the CI interpreter. It
makes no network calls of its own.

Run:  py test_ci_env.py
"""
import builtins
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "commodity_gex_dashboard.py")
SENTINEL = "# CELL 15 - RUN"

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    print("=" * 70)
    print("CI ENVIRONMENT SIMULATION (truststore blocked, as on Linux)")
    print("=" * 70)

    src = open(SCRIPT, encoding="utf-8").read()
    check("run-cell sentinel present", SENTINEL in src, SENTINEL)
    defs = src.split(SENTINEL)[0]

    # Cell 2 must import importlib.util explicitly rather than leaning on another
    # package having already loaded it.
    check("Cell 2 imports importlib.util explicitly",
          "import importlib.util" in defs)

    # Make `import truststore` fail, exactly as it does on the CI runner.
    real_import = builtins.__import__
    blocked = []

    def fake_import(name, *a, **k):
        if name == "truststore" or name.startswith("truststore."):
            blocked.append(name)
            raise ImportError("blocked by test_ci_env (simulating Linux)")
        return real_import(name, *a, **k)

    # Drop anything already cached so the block actually bites.
    for mod_name in [m for m in sys.modules if m.startswith("truststore")]:
        del sys.modules[mod_name]

    mod = types.ModuleType("gex_ci")
    mod.__file__ = SCRIPT
    mod.__dict__["display"] = lambda *a, **k: None
    sys.modules["gex_ci"] = mod

    builtins.__import__ = fake_import
    try:
        exec(compile(defs, SCRIPT, "exec"), mod.__dict__)
        loaded, err = True, ""
    except BaseException as exc:
        loaded, err = False, f"{type(exc).__name__}: {exc}"
    finally:
        builtins.__import__ = real_import

    check("definition cells execute without truststore", loaded, err)
    if not loaded:
        import traceback
        traceback.print_exc()
        print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
        return 1

    g = mod.__dict__

    # The data-source probe must report, never raise - that was the bug that
    # turned a network condition into an opaque exit 1.
    check("check_data_source is defined", callable(g.get("check_data_source")))
    check("DATA_SOURCE_OK flag exported", "DATA_SOURCE_OK" in g,
          str(g.get("DATA_SOURCE_OK")))
    check("DATA_SOURCE_MSG explains the state", bool(g.get("DATA_SOURCE_MSG")))

    # Everything the runner calls must exist.
    for fn in ("build_all", "summary_table", "save_snapshot", "load_history",
               "chart_cross_commodity", "chart_gex_by_strike", "chart_gamma_profile",
               "chart_term_structure", "fmt_money", "fmt_price", "PALETTES",
               "UNIVERSE", "DEFAULT_WATCHLIST", "CFG"):
        check(f"run_gex.py dependency '{fn}' is defined", fn in g)

    # Both palettes must carry every key the HTML template interpolates.
    needed = {"page", "surface", "ink", "ink_secondary", "muted", "grid",
              "axis", "pos", "neg"}
    for mode in ("dark", "light"):
        missing = needed - set(g["PALETTES"][mode])
        check(f"'{mode}' palette has all template keys", not missing, str(missing))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
