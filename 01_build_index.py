#!/usr/bin/env python3
"""Paso 1: indice de todas las predicciones Boltz.

Recorre prediction_roots/*/predictions/<run>/, parsea el nombre de cada corrida
y lee confidence_*.json. Si la carpeta del set termina en _x<n> (Set_Replicas), las
copias de sonda salen de ahi y no del nombre de la corrida. Una fila por (run, modelo). Solo se indexan los estados de
secuencia listados en seq_states (config.yaml).

Salidas:
  results/index.csv              predicciones con metadatos y confianza
  results/index_unparsed.txt     carpetas cuyo nombre no coincide con el formato
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from silc_boltz.config import ROOT, load_config  # noqa: E402
from silc_boltz.naming import conc_from_root, parse_run_name  # noqa: E402

CONF_KEYS = ["confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
             "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    rows, unparsed, n_skipped = [], [], 0
    for root in cfg["prediction_roots"]:
        root_conc = conc_from_root(root.name)
        for run_dir in sorted(root.glob("*/predictions/*")):
            if not run_dir.is_dir():
                continue
            meta = parse_run_name(run_dir.name, conc=root_conc)
            if meta is None:
                unparsed.append(str(run_dir))
                continue
            if meta["seq_state"] not in cfg["seq_states"]:
                n_skipped += 1
                continue
            for pdb in sorted(run_dir.glob(f"{run_dir.name}_model_*.pdb")):
                model_idx = int(pdb.stem.rsplit("_model_", 1)[1])
                row = dict(meta, model=model_idx, source=root.name,
                           pdb_path=str(pdb.relative_to(ROOT)))
                conf = run_dir / f"confidence_{run_dir.name}_model_{model_idx}.json"
                if conf.exists():
                    with open(conf) as fh:
                        c = json.load(fh)
                    row.update({k: c.get(k) for k in CONF_KEYS})
                rows.append(row)

    df = pd.DataFrame(rows)
    dup = df.duplicated(["run", "model"], keep=False)
    if dup.any():
        print(f"[!] {dup.sum()} filas con run/model repetido entre carpetas; se conserva la primera")
        df = df.drop_duplicates(["run", "model"], keep="first")

    cols = ["run", "model", "case_id", "cluster", "apo", "apo_chain", "holo", "holo_chain",
            "lig_resname", "lig_resseq", "seq_state", "probe", "conc", "conc_name", "tag", "source", "run_dir", "pdb_path"] + CONF_KEYS
    df = df[[c for c in cols if c in df.columns]].sort_values(["case_id", "seq_state", "probe", "conc", "model"])

    out = cfg["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "index.csv", index=False)
    (out / "index_unparsed.txt").write_text("\n".join(unparsed) + ("\n" if unparsed else ""))

    print(f"{len(df)} predicciones, {df.case_id.nunique()} casos, {df.holo.nunique()} cristales holo")
    print(df.groupby(["seq_state", "conc"]).size().unstack(fill_value=0).to_string())
    print(f"sin parsear: {len(unparsed)}  excluidas por seq_state: {n_skipped}")
    print(f"-> {out / 'index.csv'}")


if __name__ == "__main__":
    main()
