#!/usr/bin/env python3
"""Paso 3: posicion de la sonda predicha respecto del ligando cristalografico.

Para cada prediccion de results/index.csv:
  1. Carga el cristal holo y ubica el ligando del caso (resname + resseq del nombre,
     cadena holo preferida).
  2. Superpone la cadena proteica de Boltz sobre la cadena holo (CA alineados por
     secuencia, con descarte iterativo de zonas que no superponen).
  3. Para cada copia de sonda mide, ya en el marco del cristal:
       min_dist       distancia minima atomo-atomo sonda-ligando (A)
       centroid_dist  distancia entre centroides sonda-ligando (A)
       in_site        algun atomo de la sonda a <= contact_cutoff de un residuo del
                      sitio (residuos holo a <= site_cutoff del ligando)
     y su confianza propia segun Boltz:
       probe_plddt    pLDDT medio de los atomos de la sonda (columna B, 0-100)
       probe_pae      PAE medio sonda <-> aminoacidos a <= 6 A de ella (A)

Salidas:
  results/probe_copies.csv    una fila por copia de sonda
  results/probe_atoms.csv.gz  atomos pesados de cada copia, ya en el marco del cristal (para %TI/%PS)
  results/probe_runs.csv      una fila por prediccion (mejor copia = menor min_dist)
  results/reference_sites.csv una fila por caso: ligando encontrado, cadena, n residuos del sitio
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from silc_boltz import structures as st  # noqa: E402
from silc_boltz.config import ROOT, load_config  # noqa: E402


def load_reference(cif, holo_chain, lig_resname, lig_resseq, site_cutoff):
    model = st.load_structure(cif)
    info = dict(ref_ok=False, ref_note="")
    if holo_chain not in model:
        info["ref_note"] = "holo_chain_missing"
        return None, info
    chain = model[holo_chain]
    lig, lig_chain, note = st.find_ligand(model, lig_resname, lig_resseq, holo_chain, cif_path=cif)
    info.update(lig_match=note, lig_chain=lig_chain)
    if lig is None:
        info["ref_note"] = "ligand_not_found"
        return None, info
    seq, cas = st.seq_ca(chain)
    lig_xyz = st.ligand_xyz(lig)
    site = st.site_residues(chain, lig_xyz, site_cutoff)
    site_xyz = np.vstack([st.heavy_xyz(r) for r in site]) if site else np.empty((0, 3))
    info.update(lig_residues=" ".join(f"{r.resname.strip()}{r.id[1]}" for r in lig),
                lig_units=" ".join(sorted({r.resname.strip() for r in lig})),
                ref_ok=True, n_ref_ca=len(cas), n_lig_atoms=len(lig_xyz), n_site_res=len(site),
                site_residues=" ".join(f"{r.resname}{r.id[1]}{r.id[2].strip()}" for r in site))
    return dict(seq=seq, cas=cas, lig_xyz=lig_xyz, lig_centroid=lig_xyz.mean(0), site_xyz=site_xyz), info


def process_case(args):
    """Procesa todas las predicciones de un caso (comparten cristal de referencia)."""
    case_rows, cfg = args
    first = case_rows[0]
    cif = cfg["references_dir"] / f"{first['holo']}.cif"
    base = dict(case_id=first["case_id"])
    if not cif.exists():
        return [], [dict(base, ref_ok=False, ref_note="cif_missing")], [], []
    try:
        ref, info = load_reference(cif, first["holo_chain"], first["lig_resname"],
                                   first["lig_resseq"], cfg["site_cutoff"])
    except Exception as e:  # cristal ilegible: se registra y sigue
        return [], [dict(base, ref_ok=False, ref_note=f"error:{type(e).__name__}")], [], []
    site_row = dict(base, **info)
    if ref is None:
        return [], [site_row], [], []

    copies, runs, atoms = [], [], []
    for r in case_rows:
        key = dict(run=r["run"], model=r["model"])
        model = st.load_structure(ROOT / r["pdb_path"])
        pchain = st.protein_chain(model)
        mseq, mcas = st.seq_ca(pchain)
        sp = st.superpose(ref["seq"], ref["cas"], mseq, mcas, cfg["min_aligned_ca"])
        if sp is None:
            runs.append(dict(key, status="superpose_failed"))
            continue
        rot, tran = sp["rot"], sp["tran"]
        pdb_path = ROOT / r["pdb_path"]
        pae_path = pdb_path.with_name("pae_" + pdb_path.stem + ".npz")
        pae = np.load(pae_path)["pae"] if pae_path.exists() else None
        run_copies = []
        for cid, res in st.probe_residues(model):
            heavy = [a for a in res if a.element != "H"]
            xyz = np.array([a.coord for a in heavy], dtype=float) @ rot + tran
            atoms += [dict(key, probe_chain=cid, atom=a.get_id(), x=v[0], y=v[1], z=v[2]) for a, v in zip(heavy, xyz)]
            c = dict(key, probe_chain=cid, probe_resname=res.resname.strip(), n_probe_atoms=len(xyz),
                     min_dist=st.min_dist(xyz, ref["lig_xyz"]),
                     centroid_dist=float(np.linalg.norm(xyz.mean(0) - ref["lig_centroid"])))
            c["min_dist_site"] = st.min_dist(xyz, ref["site_xyz"]) if len(ref["site_xyz"]) else np.nan
            c["in_site"] = bool(c["min_dist_site"] <= cfg["contact_cutoff"])
            c["probe_plddt"], c["probe_pae"] = st.probe_confidence(model, res, pae)
            run_copies.append(c)
        copies += run_copies
        sup = dict(align_rmsd=sp["rmsd"], n_aligned=sp["n_aligned"], n_core=sp["n_core"],
                   seq_identity=sp["identity"])
        if not run_copies:
            runs.append(dict(key, status="no_probe", **sup))
            continue
        best = min(run_copies, key=lambda c: c["min_dist"])
        runs.append(dict(key, status="ok", **sup, n_copies=len(run_copies),
                         n_copies_in_site=sum(c["in_site"] for c in run_copies),
                         best_min_dist=best["min_dist"], best_centroid_dist=best["centroid_dist"],
                         best_in_site=best["in_site"], best_probe_plddt=best["probe_plddt"],
                         best_probe_pae=best["probe_pae"]))
    return copies, [site_row], runs, atoms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit-cases", type=int, default=None, help="procesar solo N casos (prueba)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = cfg["results_dir"]

    index = pd.read_csv(out / "index.csv")
    groups = [g.to_dict("records") for _, g in index.groupby("case_id", sort=True)]
    if args.limit_cases:
        groups = groups[:args.limit_cases]

    copies, sites, runs, atoms = [], [], [], []
    with ProcessPoolExecutor(cfg["n_workers"]) as ex:
        for c, s, r, a in tqdm(ex.map(process_case, [(g, cfg) for g in groups]), total=len(groups), unit="caso"):
            copies += c
            atoms += a
            sites += s
            runs += r

    sites = pd.DataFrame(sites)
    sites.to_csv(out / "reference_sites.csv", index=False)
    pd.DataFrame(copies).round(3).to_csv(out / "probe_copies.csv", index=False)
    pd.DataFrame(atoms).round(3).to_csv(out / "probe_atoms.csv.gz", index=False)

    meta_cols = ["run", "model", "case_id", "holo", "holo_chain", "lig_resname", "lig_resseq",
                 "seq_state", "probe", "conc", "confidence_score", "ptm", "iptm", "ligand_iptm",
                 "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"]
    runs = index[meta_cols].merge(pd.DataFrame(runs), on=["run", "model"], how="inner")
    for cut in cfg["hit_cutoffs"]:
        runs[f"hit_{cut:g}A"] = runs.best_min_dist <= cut
    runs.round(3).to_csv(out / "probe_runs.csv", index=False)

    print(f"casos: {len(sites)}  con referencia OK: {int(sites.ref_ok.sum())}")
    print(sites.ref_note.fillna("").replace("", "ok").value_counts().to_string())
    print(runs.status.value_counts().to_string())
    ok = runs[runs.status == "ok"]
    if len(ok):
        agg = {f"hit_{c:g}A": "mean" for c in cfg["hit_cutoffs"]}
        agg.update(best_in_site="mean", best_min_dist="median", n_copies="size")
        print(ok.groupby(["probe", "seq_state"]).agg(agg).round(3).to_string())


if __name__ == "__main__":
    main()
