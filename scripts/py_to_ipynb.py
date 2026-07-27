"""Thin converter: a self-verifying .py bridge/gallery script -> .ipynb.

Each `.py` script in this project ends with `LEVEL N COMPLETE` or similar
and is structured as a sequence of function calls (`part1_...`, `part2_...`,
`main()`), each followed by print-and-assert blocks. We convert each such
"part" call into a code cell and its preceding docstring + section header
into a markdown cell.

Usage:
  python scripts/py_to_ipynb.py notebooks/math_bridge/level0_foundations.py
  -> writes notebooks/math_bridge/level0_foundations.ipynb

  python scripts/py_to_ipynb.py --all
  -> converts all math bridge + gallery scripts
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def convert_one(py_path: Path, out_path: Path | None = None) -> Path:
    if out_path is None:
        out_path = py_path.with_suffix(".ipynb")
    source = py_path.read_text(encoding="utf-8")
    lines = source.split("\n")

    cells = []
    # Split on "def part" and "def main" boundaries for code cells,
    # adding markdown cells for section headers + docstring content
    current_md = []
    current_code_lines = []

    def flush_md():
        nonlocal current_md
        if current_md:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": ["\n".join(current_md)],
            })
            current_md = []

    def flush_code():
        nonlocal current_code_lines
        if current_code_lines:
            cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": ["\n".join(current_code_lines)],
                "outputs": [],
                "execution_count": None,
            })
            current_code_lines = []

    in_docstring = False
    for line in lines:
        if re.match(r'^def (part\d+_|main)\b', line):
            flush_code()
            flush_md()
            # Start a new code cell
            current_code_lines.append(line)
            in_this_func = True
        elif re.match(r'^if __name__', line):
            flush_code()
            current_code_lines.append(line)
        elif (line.strip().startswith('print("=') and line.strip().endswith('")')):
            # section header — start a markdown cell
            if len(current_code_lines) < 3:  # still in the function header
                continue
            flush_code()
            flush_md()
            current_md.append(f"### {line.strip()[8:-2]}")
        elif line.strip().startswith('print("takeaway'):
            # takeaway message -> markdown
            flush_code()
            flush_md()
            content = line.strip()[8:-2].replace('\\n', '\n')
            current_md.append(f"> **Takeaway:** {content}")
        elif current_code_lines:
            current_code_lines.append(line)
        else:
            current_code_lines.append(line)

    flush_code()
    flush_md()

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "causality-nd",
                "language": "python",
                "name": "causality-nd",
            },
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "cells": cells,
    }
    out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"  wrote {out_path} ({len(cells)} cells)")
    return out_path


def main() -> None:
    targets = []
    if "--all" in sys.argv:
        targets = sorted(Path("notebooks/math_bridge").glob("level*.py"))
        targets += sorted(Path("notebooks/tier1_gallery").glob("gallery_*.py"))
    elif len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        print("usage: py_to_ipynb.py <file.py> | --all")
        sys.exit(1)

    for t in targets:
        if t.is_file():
            convert_one(t)
    print(f"Converted {len(targets)} files.")


if __name__ == "__main__":
    main()
