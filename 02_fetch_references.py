#!/usr/bin/env python3
"""Paso 2: descarga de RCSB los cristales holo (mmCIF) que aparecen en results/index.csv.

Los archivos ya presentes no se vuelven a bajar. Registra el resultado de cada
descarga en results/references.csv (pdb_id, status, path, bytes).
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from silc_boltz.config import ROOT, load_config  # noqa: E402


def fetch(pdb_id, url_tmpl, dest_dir, retries=3):
    dest = dest_dir / f"{pdb_id}.cif"
    if dest.exists() and dest.stat().st_size > 0:
        return dict(pdb_id=pdb_id, status="cached", path=str(dest.relative_to(ROOT)), bytes=dest.stat().st_size)
    url = url_tmpl.format(pdb_id=pdb_id)
    err = ""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and r.content:
                tmp = dest.with_suffix(".cif.part")
                tmp.write_bytes(r.content)
                tmp.rename(dest)
                return dict(pdb_id=pdb_id, status="downloaded", path=str(dest.relative_to(ROOT)), bytes=len(r.content))
            err = f"http_{r.status_code}"
            if r.status_code == 404:
                break
        except requests.RequestException as e:
            err = type(e).__name__
        time.sleep(2 * (attempt + 1))
    return dict(pdb_id=pdb_id, status=f"error:{err}", path="", bytes=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)

    index = pd.read_csv(cfg["results_dir"] / "index.csv")
    ids = sorted(index.holo.str.upper().unique())
    dest_dir = cfg["references_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(args.threads) as ex:
        rows = list(ex.map(lambda i: fetch(i, cfg["rcsb_url"], dest_dir), ids))

    df = pd.DataFrame(rows).sort_values("pdb_id")
    df.to_csv(cfg["results_dir"] / "references.csv", index=False)
    print(df.status.value_counts().to_string())
    bad = df[df.status.str.startswith("error")]
    if len(bad):
        print("Fallaron:", " ".join(bad.pdb_id))


if __name__ == "__main__":
    main()
