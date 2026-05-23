#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT="$ROOT" uv run --frozen --project "$ROOT/.." python - <<'PY'
import os
from pathlib import Path

import matplotlib
import nbformat
from nbclient import NotebookClient

matplotlib.use("Agg")

root = Path(os.environ["ROOT"])
os.chdir(root)
output_dir = root / "_executed"
output_dir.mkdir(exist_ok=True)


def execute_notebook(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
        allow_errors=False,
    )
    client.execute()

    out_path = output_dir / path.name
    with out_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)


notebooks = sorted(root.glob("[0-9][0-9][0-9][0-9]-*.ipynb"))
total = len(notebooks)

for idx, nb_path in enumerate(notebooks, start=1):
    percent = (idx / total) * 100 if total else 100.0
    print(f"[{idx}/{total} | {percent:.1f}%] Running {nb_path.name}")
    execute_notebook(nb_path)
    print(f"[{idx}/{total} | {percent:.1f}%] Saved {nb_path.name} -> _executed/")

print("All notebooks completed")
PY
