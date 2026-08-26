"""Concatenate the src_*.py cell files into commodity_gex_dashboard.ipynb.

Cells are delimited by `# %%` (code) and `# %% [markdown]` (markdown). Markdown
cells have their leading `# ` comment prefix stripped.
"""
from __future__ import annotations

import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "commodity_gex_dashboard.ipynb")


def split_cells(text: str):
    lines = text.splitlines()
    cells, cur, kind = [], [], "code"
    for line in lines:
        if re.match(r"^# %%(\s|$)", line) or re.match(r"^# %% \[markdown\]\s*$", line):
            if cur:
                cells.append((kind, cur))
            kind = "markdown" if "[markdown]" in line else "code"
            cur = []
        else:
            cur.append(line)
    if cur:
        cells.append((kind, cur))
    return cells


def strip_md(lines):
    out = []
    for ln in lines:
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln.strip() == "#":
            out.append("")
        else:
            out.append(ln)
    return out


def trim(lines):
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def to_source(lines):
    """nbformat wants one string per line, each ending in \n except the last."""
    if not lines:
        return []
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def main() -> None:
    parts = sorted(glob.glob(os.path.join(HERE, "src_*.py")))
    if not parts:
        raise SystemExit("No src_*.py files found.")

    combined = "\n".join(open(p, encoding="utf-8").read() for p in parts)
    nb_cells = []
    for idx, (kind, lines) in enumerate(split_cells(combined)):
        body = trim(strip_md(lines) if kind == "markdown" else lines)
        if not body:
            continue
        # nbformat >= 4.5 requires a stable per-cell id.
        cell = {
            "cell_type": kind,
            "id": f"gex-{kind[:2]}-{idx:03d}",
            "metadata": {},
            "source": to_source(body),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)

    nb = {
        "cells": nb_cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True, "name": "commodity_gex_dashboard.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)

    md = sum(1 for c in nb_cells if c["cell_type"] == "markdown")
    print(f"Wrote {OUT}")
    print(f"  {len(nb_cells)} cells ({md} markdown, {len(nb_cells) - md} code) "
          f"from {len(parts)} source files")

    # Also emit a flat script - this is what the local test suite executes.
    flat = os.path.join(HERE, "commodity_gex_dashboard.py")
    with open(flat, "w", encoding="utf-8") as fh:
        fh.write(combined)
    print(f"Wrote {flat}")


if __name__ == "__main__":
    main()
