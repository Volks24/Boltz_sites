#!/usr/bin/env python3
"""Paso 6: tablas para el paper (CSV + PNG, en ingles).

Usa el mismo conjunto de casos que el paso 4 (predicciones que pasan el QC y con tipo de
bolsillo), para que todas las tablas describan los mismos complejos.

Salidas (results/tables/):
  interactions_per_complex.csv/.png   interacciones y residuos por complejo, por tipo de bolsillo
  interaction_types_pct.csv/.png      reparto de tipos de interaccion (% del total de cada bolsillo)
  probe_groups.csv/.png               grupos funcionales de cada sonda y tipo que representan
"""
import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from silc_boltz.config import load_config  # noqa: E402
from silc_boltz.probes import PROBE_SITES, type_label  # noqa: E402
from silc_boltz.tables import draw_table  # noqa: E402

# Reutiliza la carga + QC del paso 4.
_spec = importlib.util.spec_from_file_location("step4", HERE / "04_confidence_plots.py")
step4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step4)


TYPE_LABELS = {
    "n_acceptor": "H-bond, ligand acceptor",
    "n_donor": "H-bond, ligand donor",
    "n_hydrophobic": "Hydrophobic",
    "n_aromatic": "Aromatic (π–π / T-shaped)",
    "n_salt_bridge": "Salt bridge",
    "n_pi_cation": "π–cation",
}
POCKET_LABELS = {"rigid": "Rigid", "flexible": "Flexible", "cryptic": "Cryptic", "all": "All"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = cfg["results_dir"] / "tables"
    out.mkdir(parents=True, exist_ok=True)

    with contextlib.redirect_stdout(io.StringIO()):
        runs, _ = step4.load(cfg)
    cases = runs.drop_duplicates("case_id")[["case_id", "pocket"]]
    summ = pd.read_csv(cfg["results_dir"] / "interactions" / "crystal_summary.csv")
    s = cases.merge(summ[summ.status == "ok"], on="case_id", how="inner")
    order = [p for p in cfg["pocket_classes"] if p in set(s.pocket)] + ["all"]
    groups = {p: (s if p == "all" else s[s.pocket == p]) for p in order}
    cols = [POCKET_LABELS.get(p, p) for p in order]
    print(f"complejos: {len(s)}  " + "  ".join(f"{p}={len(g)}" for p, g in groups.items()))

    # Tabla A: interacciones por complejo
    a = pd.DataFrame({
        "Complexes (n)": [len(g) for g in groups.values()],
        "Interactions per complex (mean)": [g.n_total.mean() for g in groups.values()],
        "Interactions per complex (median)": [g.n_total.median() for g in groups.values()],
        "Residues involved per complex (mean)": [g.n_residues.mean() for g in groups.values()],
        "Total interactions (n)": [g.n_total.sum() for g in groups.values()],
    }, index=cols).T
    a.index.name = "Pocket type"
    a.round(2).to_csv(out / "interactions_per_complex.csv")
    fmt = a.copy().astype(object)
    for r in a.index:
        fmt.loc[r] = [f"{v:,.0f}" if "(n)" in r or "median" in r else f"{v:.1f}" for v in a.loc[r]]
    draw_table(fmt, "Protein–ligand interactions per complex, by pocket type",
               out / "interactions_per_complex.png",
               note="Validated interactions (distance + angle) between the crystallographic ligand and the holo chain.")

    # Tabla B: reparto por tipo (% del total de cada bolsillo)
    tot = pd.DataFrame({c: groups[p][list(TYPE_LABELS)].sum() for p, c in zip(order, cols)})
    pct = 100 * tot / tot.sum()
    pct.index = [TYPE_LABELS[i] for i in pct.index]
    pct.index.name = "Interaction type"
    b = pct.round(1)
    b.loc["Total interactions (n)"] = tot.sum()
    b.to_csv(out / "interaction_types_pct.csv")
    fmt = b.copy().astype(object)
    for r in b.index:
        fmt.loc[r] = [f"{v:,.0f}" if "(n)" in r else f"{v:.1f}" for v in b.loc[r]]
    draw_table(fmt, "Interaction types (% of all interactions in each pocket type)",
               out / "interaction_types_pct.png",
               note="π–cation and salt bridges counted once per receptor residue and ligand group.")
    # Tabla C: grupos funcionales de las sondas
    rows = []
    for probe, sites in PROBE_SITES.items():
        for k, (_, types, _, group) in enumerate(sites):
            rows.append(("" if k else probe.capitalize(), group, type_label(types)))
    c = pd.DataFrame(rows, columns=["Probe", "Functional group", "Interaction type represented"]).set_index("Probe")
    c.to_csv(out / "probe_groups.csv")
    draw_table(c, "Probe functional groups and the interaction types they represent", out / "probe_groups.png",
               bold_last_col=False, first_col_w=1.75, col_w=[1.75, 2.3], text_cols=True, group_rules=True,
               note="Ring and carboxylate groups are represented by a single site at their centroid.")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
