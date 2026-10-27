#!/usr/bin/env python3
"""Paso 5: interacciones proteina-ligando en los cristales holo con Interactions-search.

Para cada caso con ligando identificado (results/reference_sites.csv):
  1. Escribe el receptor (aminoacidos de la cadena holo, sin H) y el ligando (las
     unidades encontradas en el paso 3, como un unico residuo) en
     results/interactions/crystal/<case_id>/.
  2. Corre Interactions-search (modo receptor + ligando) con config/interactions_search.yml,
     usando el src/ del repo local. Si el ligando es una sola unidad, le pasa el SMILES
     del CCD (--ligand-smiles) para fijar enlaces, cargas y aromaticidad; si el grafo
     no coincide con el cristal (atomos faltantes, etc.) repite sin referencia.
  3. Lee las interacciones validadas (distancia + angulo) de Interaction_rec_lig_true.csv.

Solo se considera la cadena holo: contactos del ligando con otras cadenas no se cuentan.

Conteo: Interactions-search da una fila por par de atomos. Para pi_cation y salt_bridge
eso multiplica una misma interaccion (p. ej. un anillo frente a NE/NH1/NH2 de una Arg = 3
filas), asi que esas se cuentan una vez por (residuo, grupo del ligando). Puentes H se
cuentan por par de atomos y las hidrofobicas ya vienen agrupadas por residuo.

Salidas:
  results/interactions/crystal_interactions.csv  una fila por interaccion validada
  results/interactions/crystal_summary.csv       una fila por caso: conteos por tipo, quimica usada
  results/interactions/crystal_by_pocket.csv     media / mediana / total por tipo de bolsillo
  data/references/ccd_smiles.csv                 cache de SMILES del CCD
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from silc_boltz import structures as st  # noqa: E402
from silc_boltz.config import load_config  # noqa: E402
from silc_boltz.interactions import dedupe  # noqa: E402

TYPES = ["acceptor", "donor", "hydrophobic", "aromatic", "salt_bridge", "pi_cation"]


def ccd_smiles(comp_ids, url, cache_path):
    cache = pd.read_csv(cache_path, keep_default_na=False) if cache_path.exists() else pd.DataFrame(
        columns=["comp_id", "smiles"])
    known = dict(zip(cache.comp_id, cache.smiles))
    for cid in sorted(set(comp_ids) - set(known)):
        try:
            r = requests.get(url.format(comp_id=cid), timeout=30)
            d = r.json().get("rcsb_chem_comp_descriptor", {}) if r.ok else {}
            known[cid] = d.get("SMILES_stereo") or d.get("SMILES") or ""
        except requests.RequestException:
            continue  # no se cachea: se reintenta en la proxima corrida
    pd.DataFrame(sorted(known.items()), columns=["comp_id", "smiles"]).to_csv(cache_path, index=False)
    return known


def run_is(workdir, icfg, smiles=None):
    out = workdir / "rec_lig"
    if out.exists():
        shutil.rmtree(out)
    cmd = [sys.executable, "-c", "from interactions_search.cli import main; main()",
           "-r", "rec.pdb", "-l", "lig.pdb", "-c", "A", "--config", str(icfg["config"])]
    if smiles:
        cmd += ["--ligand-smiles", smiles]
    env = dict(os.environ, PYTHONPATH=str(icfg["repo"] / "src"))
    p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=icfg["timeout_s"])
    (workdir / ("is_smiles.log" if smiles else "is_pdb.log")).write_text(p.stdout + p.stderr)
    ok = p.returncode == 0 and (out / "Interaction_rec_lig_true.csv").exists()
    err = "" if ok else (p.stderr.strip().splitlines() or ["?"])[-1][:200]
    return ok, err


def process_case(args):
    row, cfg, smiles_map, force = args
    case = row["case_id"]
    wd = cfg["results_dir"] / "interactions" / "crystal" / case
    res = dict(case_id=case)
    true_csv = wd / "rec_lig" / "Interaction_rec_lig_true.csv"
    meta_path = wd / "run.json"
    if not force and true_csv.exists() and meta_path.exists():
        res.update(json.loads(meta_path.read_text()))
    else:
        wd.mkdir(parents=True, exist_ok=True)
        cif = cfg["references_dir"] / f"{row['holo']}.cif"
        model = st.load_structure(cif)
        lig, _, _ = st.find_ligand(model, row["lig_resname"], int(row["lig_resseq"]), row["holo_chain"], cif_path=cif)
        resnames = sorted({r.resname.strip() for r in lig})
        st.write_pdb(wd / "rec.pdb", st.receptor_records(model[row["holo_chain"]]))
        st.write_pdb(wd / "lig.pdb", st.ligand_records(lig, resname=resnames[0][:3] if len(lig) == 1 else "LIG"))
        smiles = smiles_map.get(resnames[0], "") if len(lig) == 1 else ""
        chem, note = "pdb_only", "multi_residue" if len(lig) > 1 else ("no_ccd_smiles" if not smiles else "")
        ok = False
        if smiles:
            ok, err = run_is(wd, cfg["interactions"], smiles)
            chem, note = ("ccd_smiles", "") if ok else ("pdb_only", f"smiles_rejected: {err}")
        if not ok:
            ok, err = run_is(wd, cfg["interactions"])
            if not ok:
                note = (note + " | " if note else "") + f"is_failed: {err}"
        res.update(status="ok" if ok else "failed", chem_source=chem, note=note,
                   lig_units=" ".join(resnames), n_lig_atoms=sum(1 for r in lig for a in r if a.element != "H"))
        meta_path.write_text(json.dumps({k: v for k, v in res.items() if k != "case_id"}))

    inter = pd.DataFrame()
    if res.get("status") == "ok":
        inter = pd.read_csv(true_csv)
        inter = inter[inter.Interaction.eq("Yes")].copy()
        inter.insert(0, "case_id", case)
        uniq = dedupe(inter)
        for t in TYPES:
            res[f"n_{t}"] = int((uniq.Type == t).sum())
        res["n_total"] = len(uniq)
        res["n_rows_raw"] = len(inter)
        res["n_residues"] = inter[["Pos R", "Res"]].drop_duplicates().shape[0]
    return res, inter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit-cases", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="recalcular aunque ya exista el resultado")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res_dir = cfg["results_dir"]

    sites = pd.read_csv(res_dir / "reference_sites.csv")
    index = pd.read_csv(res_dir / "index.csv")
    meta = index.drop_duplicates("case_id")[["case_id", "holo", "holo_chain", "lig_resname", "lig_resseq"]]
    cases = sites[sites.ref_ok.astype(bool)][["case_id", "lig_units"]].merge(meta, on="case_id")
    if args.limit_cases:
        cases = cases.head(args.limit_cases)

    units = {u for s in cases.lig_units for u in s.split()}
    smiles_map = ccd_smiles(units, cfg["interactions"]["ccd_api"], cfg["references_dir"].parent / "ccd_smiles.csv")

    rows, inters = [], []
    jobs = [(r, cfg, smiles_map, args.force) for r in cases.to_dict("records")]
    with ProcessPoolExecutor(cfg["n_workers"]) as ex:
        for r, i in tqdm(ex.map(process_case, jobs), total=len(jobs), unit="caso"):
            rows.append(r)
            if len(i):
                inters.append(i)

    out = res_dir / "interactions"
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "crystal_summary.csv", index=False)
    if inters:
        pd.concat(inters).to_csv(out / "crystal_interactions.csv", index=False)

    print(summ.status.value_counts().to_string())
    print(summ.chem_source.value_counts().to_string())
    ok = summ[summ.status == "ok"]
    print(ok[[f"n_{t}" for t in TYPES] + ["n_total", "n_residues"]].describe().round(1).to_string())
    cls = pd.read_csv(cfg["pocket_classification"], usecols=["system", "classification"])
    ok = ok.merge(cls.rename(columns={"system": "case_id", "classification": "pocket"}), on="case_id")
    ok = ok[ok.pocket.isin(cfg["pocket_classes"])]
    cols = [f"n_{t}" for t in TYPES] + ["n_total", "n_residues"]
    parts = []
    for pocket, g in list(ok.groupby("pocket")) + [("all", ok)]:
        for stat in ("mean", "median", "sum"):
            parts.append(dict(pocket=pocket, stat=stat, n_cases=len(g), **g[cols].agg(stat).round(2).to_dict()))
    bp = pd.DataFrame(parts)
    bp["_o"] = bp.pocket.map({p: i for i, p in enumerate(cfg["pocket_classes"] + ["all"])})
    bp.sort_values(["stat", "_o"]).drop(columns="_o").to_csv(out / "crystal_by_pocket.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
