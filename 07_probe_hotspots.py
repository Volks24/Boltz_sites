#!/usr/bin/env python3
"""Paso 7: %TI y %PS de cada sonda contra las interacciones del cristal.

Interacciones reales: results/interactions/crystal_interactions.csv (paso 5, coordenada
del lado del ligando), deduplicadas y traducidas al tipo que tendria que tener una sonda
en esa posicion (silc_boltz.interactions.probe_type).

Sitios de sonda: grupos funcionales de silc_boltz.probes.PROBE_SITES, con las coordenadas
de results/probe_atoms.csv.gz (paso 3, ya en el marco del cristal). Solo predicciones que
pasan el QC del paso 4 y casos con tipo de bolsillo.

Coincidencia: un sitio de sonda y una interaccion real coinciden si comparten tipo y el
sitio esta a < match_distance del atomo/centroide del ligando que hace la interaccion.

  %TI = posiciones encontradas / posiciones que la sonda podria cubrir, sumado sobre los
        pares (caso, sonda, concentracion):
          - posicion: coordenada del ligando que hace interacciones de los tipos que la sonda
            representa; varias interacciones en el mismo atomo (p. ej. un O con 3 puentes H)
            cuentan una vez. Se encuentra si alguna copia cubre alguna de sus interacciones.
          - capacidad: min(posiciones, sub-sitios utiles x copias). Un sub-sitio es util si
            alguno de sus tipos esta entre los del cristal (etanol en un cristal con 6
            posiciones de OH/CH3: 2 sub-sitios x 1 copia = 2, no 6).
          - encontradas efectivas (n_found_eff): emparejamiento maximo 1 a 1 entre sub-sitios
            y posiciones (mismo tipo, < match_distance). Un grupo cerca de varias posiciones
            (un CH3 entre dos C vecinos de un anillo) ocupa solo una; nunca supera la capacidad.
            n_pos_found sigue contando posiciones con algun grupo cerca, sin emparejar.
        pct_TI_raw conserva la definicion anterior (interacciones / interacciones).
  %PS = sitios de sonda que coinciden con alguna interaccion real de alguno de sus tipos /
        sitios de sonda totales (cada copia y cada grupo cuentan aparte).

Copias sobre su interaccion: de las copias que entraron (en el sitio = in_site del paso 3;
sobre el ligando = min_dist <= hit_cutoff_main), % con algun grupo a < match_distance de
una interaccion real de su tipo. Solo pares (caso, sonda) cuyo cristal tiene al menos una
interaccion de los tipos de la sonda. Salidas: copy_on_interaction.csv/.png (sonda x conc),
copy_on_interaction_pocket.csv/.png (bolsillo x conc, todas las sondas) y, por concentracion,
copy_on_interaction_by_pocket.csv/.png (sonda x bolsillo).

Si hay mas de una concentracion de sonda (Set_Replicas) todo se calcula por separado en
hotspots/x<n>/ y se agrega hotspots/probe_ti_ps_by_conc.csv/.png (sonda x concentracion), y
la misma tabla solo con los pares (caso, sonda) presentes en todas las concentraciones
(probe_ti_ps_by_conc_common.csv/.png; ahi las posiciones del cristal son las mismas en
cada columna y se muestran en la ultima).

Salidas (results/hotspots/):
  probe_ti_ps.csv            conteos y % por sonda x bolsillo (incluye 'all')
  ti_pairs.csv               posiciones, capacidad y encontradas por (caso, sonda)
  probe_ti_by_type.csv       %TI por sonda x tipo de interaccion
  matches_real.csv.gz        cada interaccion real considerada, con su sonda y si fue encontrada
  matches_sites.csv.gz       cada sitio de sonda, con su distancia a la interaccion mas cercana
  probe_ti_ps.png            tabla del paper
  category_ti_ps.csv/.png    %TI / %PS por categoria de interaccion (Aromatic, Aliphatic,
                             Both = Aromatic + Aliphatic, H-bond) x bolsillo
  case_hits_probe.csv/.png   % de casos con >= 1 interaccion recuperada, por sonda x bolsillo
  case_hits_category.csv/.png  idem por categoria de interaccion

% de casos: cada par (caso, sonda) es positivo si la sonda recupera al menos una
interaccion del cristal (de los tipos que representa) y negativo si no recupera ninguna.
Solo entran los pares donde el cristal tiene al menos una interaccion de esos tipos; los
demas se cuentan aparte (n_pairs_no_target). Por categoria, lo mismo restringido a la
categoria (en "Both", positivo si recupera al menos una aromatica o alifatica).
"""
import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from silc_boltz.config import load_config  # noqa: E402
from silc_boltz.interactions import dedupe, probe_type  # noqa: E402
from silc_boltz.probes import PROBE_SITES, TYPE_LABELS  # noqa: E402
from silc_boltz.tables import draw_table  # noqa: E402

_spec = importlib.util.spec_from_file_location("step4", HERE / "04_confidence_plots.py")
step4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step4)

POCKET_LABELS = {"rigid": "Rigid", "flexible": "Flexible", "cryptic": "Cryptic", "all": "All"}
# Categoria de interaccion a partir del tipo de sonda; "Both" = Aromatic + Aliphatic.
CATEGORY = {"aromatic": "Aromatic", "hydrophobic": "Aliphatic", "hbd": "H-bond", "hba": "H-bond",
            "cation": "Ionic", "anion": "Ionic"}
CATEGORY_ROWS = [("Aromatic", {"Aromatic"}), ("Aliphatic", {"Aliphatic"}),
                 ("Both", {"Aromatic", "Aliphatic"}), ("H-bond", {"H-bond"}), ("Ionic", {"Ionic"})]
MIN_REAL = 20  # %TI con menos interacciones reales que esto se muestra como "–" en la tabla PNG


def probe_sites(atoms, runs):
    """Un registro por (prediccion, copia, grupo funcional)."""
    meta = runs.set_index(["run", "model"])[["case_id", "probe", "conc", "pocket"]]
    out = []
    for (run, model, chain), g in atoms.groupby(["run", "model", "probe_chain"], sort=False):
        case_id, probe, conc, pocket = meta.loc[(run, model)]
        xyz = dict(zip(g.atom, g[["x", "y", "z"]].to_numpy()))
        for k, (names, types, mode, label) in enumerate(PROBE_SITES[probe]):
            pts = [xyz[n] for n in names if n in xyz]
            if not pts:
                continue
            pts = [np.mean(pts, axis=0)] if mode == "centroid" else pts
            for j, p in enumerate(pts):
                out.append(dict(run=run, model=model, probe_chain=chain, case_id=case_id, probe=probe,
                                conc=conc, pocket=pocket, site=f"{k}.{j}", group=label, types=types,
                                x=p[0], y=p[1], z=p[2]))
    return pd.DataFrame(out)


def match(real, sites, dist):
    """Marca interacciones reales encontradas y sitios de sonda correctos, por (caso, sonda, conc)."""
    real_rows, site_rows = [], []
    for (case_id, probe, conc), sg in sites.groupby(["case_id", "probe", "conc"]):
        covered = {t for ts in sg.types for t in ts}
        rg = real[(real.case_id == case_id) & real.ptype.isin(covered)]
        found = np.zeros(len(rg), bool)
        best = np.full(len(sg), np.inf)
        hits = [set() for _ in range(len(sg))]       # (posicion, tipo) a < dist de cada sitio
        for t in covered:
            rm = (rg.ptype == t).to_numpy()
            sm = sg.types.apply(lambda ts, t=t: t in ts).to_numpy()
            if not rm.any() or not sm.any():
                continue
            rxyz = rg.loc[rm, ["X", "Y", "Z"]].to_numpy()
            sxyz = sg.loc[sm, ["x", "y", "z"]].to_numpy()
            d = cdist(sxyz, rxyz)
            best[sm] = np.minimum(best[sm], d.min(axis=1))
            found[np.flatnonzero(rm)[(d < dist).any(axis=0)]] = True
            pos = rg.pos.to_numpy()[rm]
            for i, near in zip(np.flatnonzero(sm), d < dist):
                hits[i] |= {(q, t) for q in pos[near]}
        real_rows.append(rg.assign(probe=probe, conc=conc, found=found))
        site_rows.append(sg.assign(nearest_real=best, correct=best < dist, hits=hits))
    return pd.concat(real_rows), pd.concat(site_rows)


PAIR = ["case_id", "probe", "conc", "pocket"]


def n_matched(hits):
    """Emparejamiento maximo sub-sitio <-> posicion (caminos aumentantes de Kuhn)."""
    owner = {}

    def augment(i, seen):
        for q in hits[i]:
            if q not in seen:
                seen.add(q)
                if q not in owner or augment(owner[q], seen):
                    owner[q] = i
                    return True
        return False

    return sum(augment(i, set()) for i in range(len(hits)))


def ti_pairs(real_m, site_m, types=None):
    """Por (caso, sonda, conc): posiciones del cristal, encontradas y capacidad de la sonda.

    types restringe a esos tipos de sonda (interacciones reales y sub-sitios)."""
    r, s = real_m, site_m
    if types is not None:
        r = r[r.ptype.isin(types)]
        s = s[s.types.apply(lambda ts: bool(set(ts) & set(types)))]
    if r.empty:
        return pd.DataFrame(columns=PAIR + ["n_pos", "n_pos_found", "capacity", "n_found_eff"])
    pos = r.groupby(PAIR + ["pos"]).found.any().reset_index()
    out = pos.groupby(PAIR).found.agg(n_pos="size", n_pos_found="sum")
    present = r.groupby(PAIR).ptype.agg(set).rename("present")
    s = s.join(present, on=PAIR, how="inner")
    s["useful"] = [bool(set(ts) & pr) for ts, pr in zip(s.types, s.present)]
    out = out.join(s.groupby(PAIR).useful.sum().rename("useful_sites")).fillna({"useful_sites": 0})
    out["capacity"] = np.minimum(out.n_pos, out.useful_sites).astype(int)
    s["hit_pos"] = [{q for q, t in h if types is None or t in types} for h in s.hits]
    matched = s.groupby(PAIR).hit_pos.agg(lambda h: n_matched(list(h))).rename("n_found_eff")
    out = out.join(matched).fillna({"n_found_eff": 0})
    out["n_found_eff"] = out.n_found_eff.astype(int)
    return out.reset_index()


def ti_pct(t):
    return round(100 * t.n_found_eff.sum() / t.capacity.sum(), 1) if t.capacity.sum() else np.nan


def summarize(real_m, site_m, pairs, pockets, probes):
    rows = []
    for probe in probes + ["all"]:
        for pocket in pockets + ["all"]:
            sel = lambda d: d[((d.probe == probe) | (probe == "all")) & ((d.pocket == pocket) | (pocket == "all"))]
            r, s, t = sel(real_m), sel(site_m), sel(pairs)
            rows.append(dict(probe=probe, pocket=pocket, n_cases=s.case_id.nunique(),
                             n_real=len(r), n_real_found=int(r.found.sum()),
                             pct_TI_raw=round(100 * r.found.mean(), 1) if len(r) else np.nan,
                             n_pos=int(t.n_pos.sum()), n_pos_found=int(t.n_pos_found.sum()),
                             capacity=int(t.capacity.sum()), n_found_eff=int(t.n_found_eff.sum()),
                             pct_TI=ti_pct(t),
                             n_sites=len(s), n_sites_correct=int(s.correct.sum()),
                             pct_PS=round(100 * s.correct.mean(), 1) if len(s) else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = cfg["results_dir"]
    base = res / "hotspots"
    base.mkdir(parents=True, exist_ok=True)
    dist = cfg["match_distance"]

    with contextlib.redirect_stdout(io.StringIO()):
        runs, _ = step4.load(cfg)
    cases = runs.drop_duplicates("case_id")[["case_id", "pocket"]]

    real = pd.read_csv(res / "interactions" / "crystal_interactions.csv")
    real = dedupe(real[real.case_id.isin(cases.case_id)])
    real["ptype"] = real.apply(probe_type, axis=1)
    real = real.dropna(subset=["ptype", "X", "Y", "Z"]).merge(cases, on="case_id")
    real["pos"] = [f"{x:.2f},{y:.2f},{z:.2f}" for x, y, z in zip(real.X, real.Y, real.Z)]

    atoms = pd.read_csv(res / "probe_atoms.csv.gz").merge(runs[["run", "model"]], on=["run", "model"])
    sites = probe_sites(atoms, runs)
    copies = pd.read_csv(res / "probe_copies.csv", usecols=["run", "model", "probe_chain", "in_site", "min_dist"])
    sites = sites.merge(copies, on=["run", "model", "probe_chain"], how="left")
    real_m, site_m = match(real, sites, dist)

    pockets = [p for p in cfg["pocket_classes"] if p in set(runs.pocket)]
    concs = sorted(runs.conc.unique())
    split = len(concs) > 1
    summs = []
    for conc in concs:
        out = base / f"x{conc}" if split else base
        out.mkdir(parents=True, exist_ok=True)
        label = f" ({conc} probe cop{'y' if conc == 1 else 'ies'})" if split else ""
        print(f"\n== x{conc}")
        rm, sm = real_m[real_m.conc == conc].copy(), site_m[site_m.conc == conc].copy()
        summs.append(report(rm, sm, pockets, out, dist, label).assign(conc=conc))
        copy_on_interaction_by_pocket(rm, sm, pockets, out, dist, cfg["hit_cutoff_main"], label)
    cut = cfg["hit_cutoff_main"]
    copy_on_interaction(real_m, site_m, concs, base, dist, cut)
    copy_on_interaction_pocket(real_m, site_m, pockets, concs, base, dist, cut)
    if split:
        by_conc(pd.concat(summs), concs, base, dist)
        common = set.intersection(*[set(zip(g.case_id, g.probe)) for _, g in site_m.groupby("conc")])
        keep = lambda d: d[[k in common for k in zip(d.case_id, d.probe)]]
        rc, sc = keep(real_m), keep(site_m)
        copy_on_interaction(rc, sc, concs, base, dist, cut, common=(len(common), sc.case_id.nunique()))
        copy_on_interaction_pocket(rc, sc, pockets, concs, base, dist, cut, common=True)
        summs = []
        for conc in concs:
            r, st_ = rc[rc.conc == conc], sc[sc.conc == conc]
            probes = [p for p in PROBE_SITES if p in set(st_.probe)]
            summs.append(summarize(r, st_, ti_pairs(r, st_), [], probes).assign(conc=conc))
        by_conc(pd.concat(summs), concs, base, dist, common=(len(common), sc.case_id.nunique()))


def copy_flags(real_m, site_m, cut):
    """Una fila por copia de sonda elegible: on (sobre su interaccion), in_site, near (<= cut del ligando)."""
    elig = set(zip(real_m.case_id, real_m.probe, real_m.conc))
    c = (site_m.groupby(["run", "model", "probe_chain", "case_id", "probe", "conc", "pocket"])
         .agg(on=("correct", "any"), in_site=("in_site", "first"), min_dist=("min_dist", "first")).reset_index())
    c = c[[k in elig for k in zip(c.case_id, c.probe, c.conc)]].copy()
    c["near"] = c.min_dist <= cut
    return c


def on_rates(g):
    """Conteos y % de copias sobre su interaccion entre las que entraron (en el sitio / cerca)."""
    row = dict(n_copies=len(g), n_on=int(g.on.sum()))
    for key in ("in_site", "near"):
        e = g[g[key]]
        row.update({f"n_{key}": len(e), f"n_{key}_on": int(e.on.sum()),
                    f"pct_{key}_on": round(100 * e.on.mean(), 1) if len(e) else np.nan})
    return row


def fmt_pct(v, n):
    return "–" if n < MIN_REAL or pd.isna(v) else f"{v:.1f}"


def copy_on_interaction_pocket(real_m, site_m, pockets, concs, out, dist, cut, common=None):
    """Bolsillo x concentracion, todas las sondas juntas."""
    stem = "copy_on_interaction_pocket" + ("_common" if common else "")
    c = copy_flags(real_m, site_m, cut)
    tab = pd.DataFrame([dict(pocket=pocket, conc=conc,
                             **on_rates(c[(c.conc == conc) & ((c.pocket == pocket) | (pocket == "all"))]))
                        for pocket in pockets + ["all"] for conc in concs])
    tab.to_csv(out / f"{stem}.csv", index=False)
    labels = [f"{c_} cop{'y' if c_ == 1 else 'ies'}" for c_ in concs]
    cols = {}
    for key in ("in_site", "near"):
        for lab, conc in zip(labels, concs):
            g = tab[tab.conc == conc].set_index("pocket").reindex(pockets + ["all"])
            cols[(key, lab)] = [fmt_pct(v, n) for v, n in zip(g[f"pct_{key}_on"], g[f"n_{key}"])]
    t = pd.DataFrame(cols, index=[POCKET_LABELS[p] for p in pockets + ["all"]])
    t.columns = labels + labels
    t.index.name = "Pocket"
    n_cases = c.groupby("pocket").case_id.nunique()
    which = "common case–probe pairs" if common else "all predictions"
    draw_table(t, f"Probe copies sitting on a crystal interaction of their type, by pocket type ({which})",
               out / f"{stem}.png", first_col_w=1.6, col_w=0.95, bold_last_col=False,
               col_groups=[("Copies in the binding site (%)", len(concs)),
                           (f"Copies ≤ {cut:g} Å from the ligand (%)", len(concs))],
               note=f"Copy with ≥ 1 group < {dist:g} Å from a crystal interaction of its type; all probes pooled. "
                    f"Cases: " + ", ".join(f"{p} {n_cases.get(p, 0)}" for p in pockets)
                    + f". – : fewer than {MIN_REAL} copies.")
    print(f"\nCopias sobre su interaccion por bolsillo ({which}):")
    print(tab.pivot(index="pocket", columns="conc", values=["pct_in_site_on", "pct_near_on", "n_in_site", "n_near"])
          .reindex(pockets + ["all"]).to_string())
    print(f"-> {out / f'{stem}.png'}")


def copy_on_interaction_by_pocket(real_m, site_m, pockets, out, dist, cut, label):
    """Una concentracion: sonda x bolsillo, con los dos criterios de 'entrar'."""
    c = copy_flags(real_m, site_m, cut)
    probes = [p for p in PROBE_SITES if p in set(c.probe)]
    tab = pd.DataFrame([dict(probe=probe, pocket=pocket,
                             **on_rates(c[((c.probe == probe) | (probe == "all"))
                                          & ((c.pocket == pocket) | (pocket == "all"))]))
                        for probe in probes + ["all"] for pocket in pockets + ["all"]])
    tab.to_csv(out / "copy_on_interaction_by_pocket.csv", index=False)
    order = (tab[(tab.pocket == "all") & (tab.probe != "all")].sort_values("pct_near_on", ascending=False)
             .probe.tolist()) + ["all"]
    cols = {}
    for pocket in pockets + ["all"]:
        g = tab[tab.pocket == pocket].set_index("probe").reindex(order)
        cols[(pocket, "Site")] = [fmt_pct(v, n) for v, n in zip(g.pct_in_site_on, g.n_in_site)]
        cols[(pocket, f"≤ {cut:g} Å")] = [fmt_pct(v, n) for v, n in zip(g.pct_near_on, g.n_near)]
    t = pd.DataFrame(cols, index=[("All probes" if p == "all" else p.capitalize()) for p in order])
    t.columns = [c_[1] for c_ in t.columns]
    t.index.name = "Probe"
    draw_table(t, "Probe copies sitting on a crystal interaction of their type, by pocket type" + label,
               out / "copy_on_interaction_by_pocket.png", first_col_w=1.8, col_w=0.8, bold_last_col=False,
               col_groups=[(POCKET_LABELS[p], 2) for p in pockets + ["all"]],
               note=f"% of copies in the binding site (Site) or ≤ {cut:g} Å from the ligand with ≥ 1 group "
                    f"< {dist:g} Å from a crystal interaction of its type. – : fewer than {MIN_REAL} copies.")


def copy_on_interaction(real_m, site_m, concs, out, dist, cut, common=None):
    """% de copias que entraron (en el sitio / sobre el ligando) con algun grupo sobre su interaccion."""
    stem = "copy_on_interaction" + ("_common" if common else "")
    c = copy_flags(real_m, site_m, cut)
    probes = [p for p in PROBE_SITES if p in set(c.probe)]
    rows = []
    for probe in probes + ["all"]:
        for conc in concs:
            g = c[(c.conc == conc) & ((c.probe == probe) | (probe == "all"))]
            rows.append(dict(probe=probe, conc=conc, **on_rates(g)))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / f"{stem}.csv", index=False)

    order = (tab[(tab.conc == concs[0]) & (tab.probe != "all")].sort_values("pct_near_on", ascending=False)
             .probe.tolist()) + ["all"]
    labels = [f"{c_} cop{'y' if c_ == 1 else 'ies'}" for c_ in concs]
    parts = []
    for key in ("in_site", "near"):
        pct = tab.pivot(index="probe", columns="conc", values=f"pct_{key}_on").reindex(order)
        n = tab.pivot(index="probe", columns="conc", values=f"n_{key}").reindex(order)
        parts.append(pd.DataFrame({lab: [fmt_pct(v, k) for v, k in zip(pct[c_], n[c_])]
                                   for lab, c_ in zip(labels, concs)},
                                  index=order))
    t = pd.concat(parts, axis=1).astype(object)
    t.columns = labels + labels
    t.index = [("All probes" if p == "all" else p.capitalize()) for p in order]
    t.index.name = "Probe"
    which = (f"common case–probe pairs: {common[0]} pairs, {common[1]} cases" if common else "all predictions")
    draw_table(t, f"Probe copies sitting on a crystal interaction of their type ({which.split(':')[0]})",
               out / f"{stem}.png", first_col_w=1.9, col_w=0.95, bold_last_col=False,
               col_groups=[("Copies in the binding site (%)", len(concs)),
                           (f"Copies ≤ {cut:g} Å from the ligand (%)", len(concs))],
               note=f"Copy with ≥ 1 group < {dist:g} Å from a crystal interaction of its type. Only crystals with "
                    f"such interactions. – : fewer than {MIN_REAL} copies."
                    + (f" {common[0]} pairs ({common[1]} cases)." if common else ""))
    print(f"\nCopias sobre su interaccion ({which}):")
    print(tab.pivot(index="probe", columns="conc", values=["pct_in_site_on", "pct_near_on", "n_in_site", "n_near"])
          .reindex(order).to_string())
    print(f"-> {out / f'{stem}.png'}")


def by_conc(summ, concs, out, dist, common=None):
    """%TI y %PS por sonda x concentracion (todos los bolsillos).

    common=(n_pares, n_casos): tabla de pares comunes; agrega la columna de posiciones
    (iguales en todas las concentraciones) y no oculta %TI con pocas posiciones."""
    stem = "probe_ti_ps_by_conc" + ("_common" if common else "")
    summ = summ[summ.pocket == "all"]
    summ.insert(0, "conc", summ.pop("conc"))
    summ.to_csv(out / f"{stem}.csv", index=False)
    order = (summ[(summ.conc == concs[0]) & (summ.probe != "all")].sort_values("pct_TI", ascending=False)
             .probe.tolist()) + ["all"]
    labels = [f"{c} cop{'y' if c == 1 else 'ies'}" for c in concs]
    ti = summ.pivot(index="probe", columns="conc", values="pct_TI").reindex(order)
    ps = summ.pivot(index="probe", columns="conc", values="pct_PS").reindex(order)
    npos = summ.pivot(index="probe", columns="conc", values="n_pos").reindex(order)
    min_pos = 0 if common else MIN_REAL
    tab = pd.concat([pd.DataFrame([[("–" if n < min_pos or pd.isna(v) else f"{v:.1f}") for v, n in zip(ti[c], npos[c])]
                                   for c in concs], index=labels, columns=order).T,
                     ps.apply(lambda col: col.map("{:.1f}".format))], axis=1).astype(object)
    tab.columns = labels + labels
    groups = [("%TI", len(concs)), ("%PS", len(concs))]
    col_w = [0.95] * len(tab.columns)
    if common:
        tab["Positions (n)"] = npos[concs[0]].astype(int).astype(str).values
        groups.append(("Crystal", 1))
        col_w.append(1.35)
    tab.index = [("All probes" if p == "all" else p.capitalize()) for p in order]
    tab.index.name = "Probe"
    note = (f"%TI: crystal interaction positions recovered (< {dist:g} Å), over what the probe copies "
            "could cover. %PS: probe groups matching one.")
    if common:
        title = "Interaction hot-spots recovered by probe and probe concentration (common case–probe pairs)"
        note = (f"%TI: crystal positions recovered (< {dist:g} Å) over probe capacity. %PS: probe groups "
                f"matching one. {common[0]} pairs ({common[1]} cases) at every concentration.")
    else:
        title = "Interaction hot-spots recovered by probe and probe concentration (all predictions)"
        note += f" – : fewer than {MIN_REAL} positions."
    draw_table(tab, title, out / f"{stem}.png", first_col_w=1.9, col_w=col_w, bold_last_col=False,
               col_groups=groups, note=note)
    print(f"\n%TI / %PS por concentracion ({'pares comunes' if common else 'todas las predicciones'}):")
    print(summ.pivot(index="probe", columns="conc", values=["pct_TI", "pct_PS", "n_pos"]).reindex(order).to_string())
    print(f"-> {out / f'{stem}.png'}")


def report(real_m, site_m, pockets, out, dist, label):
    probes = [p for p in PROBE_SITES if p in set(site_m.probe)]
    pairs = ti_pairs(real_m, site_m)
    pairs.to_csv(out / "ti_pairs.csv", index=False)
    summ = summarize(real_m, site_m, pairs, pockets, probes)
    summ.to_csv(out / "probe_ti_ps.csv", index=False)

    rows = []
    for t in sorted(real_m.ptype.unique()):
        pt = ti_pairs(real_m, site_m, types={t})
        for probe, g in pt.groupby("probe"):
            rows.append(dict(probe=probe, ptype=t, n_real=int((real_m.ptype.eq(t) & real_m.probe.eq(probe)).sum()),
                             n_pos=int(g.n_pos.sum()), capacity=int(g.capacity.sum()),
                             n_found_eff=int(g.n_found_eff.sum()), pct_TI=ti_pct(g)))
    by_type = pd.DataFrame(rows)
    by_type.to_csv(out / "probe_ti_by_type.csv", index=False)
    real_m.to_csv(out / "matches_real.csv.gz", index=False)
    site_m.drop(columns=["types", "hits"]).assign(types=site_m.types.str.join("+")).to_csv(out / "matches_sites.csv.gz", index=False)

    # Tabla del paper: filas = sondas (+ todas), columnas = %TI / %PS por bolsillo
    order = summ[(summ.pocket == "all") & (summ.probe != "all")].sort_values("pct_TI", ascending=False).probe.tolist()
    tab = {}
    for pocket in pockets + ["all"]:
        g = summ[summ.pocket == pocket].set_index("probe").reindex(order + ["all"])
        tab[(pocket, "%TI")] = [("–" if n < MIN_REAL else f"{v:.1f}") for v, n in zip(g.pct_TI, g.n_pos)]
        tab[(pocket, "%PS")] = g.pct_PS.map("{:.1f}".format)
    tab = pd.DataFrame(tab)
    tab.columns = [c[1] for c in tab.columns]
    tab.index = [("All probes" if p == "all" else p.capitalize()) for p in tab.index]
    tab.index.name = "Probe"
    draw_table(tab, "Interaction hot-spots recovered by AISM-placed probes, by pocket type" + label,
               out / "probe_ti_ps.png", first_col_w=1.8, col_w=0.8, bold_last_col=False,
               col_groups=[(POCKET_LABELS[p], 2) for p in pockets + ["all"]],
               note=f"%TI: crystal interaction positions recovered (< {dist:g} Å), over what the probe copies could "
                    f"cover. %PS: probe groups matching one. – : fewer than {MIN_REAL} positions.")

    # Por categoria de interaccion: interacciones reales por su tipo; sitios por el tipo de su grupo.
    real_m["category"] = real_m.ptype.map(CATEGORY)
    site_m["category"] = site_m.types.apply(lambda ts: CATEGORY[ts[0]])
    rows = []
    for name, cats in CATEGORY_ROWS:
        cat_pairs = ti_pairs(real_m, site_m, types={t for t, c in CATEGORY.items() if c in cats})
        for pocket in pockets + ["all"]:
            r = real_m[real_m.category.isin(cats) & ((real_m.pocket == pocket) | (pocket == "all"))]
            st_ = site_m[site_m.category.isin(cats) & ((site_m.pocket == pocket) | (pocket == "all"))]
            t = cat_pairs[(cat_pairs.pocket == pocket) | (pocket == "all")]
            rows.append(dict(category=name, pocket=pocket, probes=", ".join(sorted(st_.probe.unique())),
                             n_real=len(r), n_real_found=int(r.found.sum()),
                             pct_TI_raw=round(100 * r.found.mean(), 1) if len(r) else np.nan,
                             n_pos=int(t.n_pos.sum()), capacity=int(t.capacity.sum()),
                             n_found_eff=int(t.n_found_eff.sum()), pct_TI=ti_pct(t),
                             n_sites=len(st_), n_sites_correct=int(st_.correct.sum()),
                             pct_PS=round(100 * st_.correct.mean(), 1) if len(st_) else np.nan))
    cat = pd.DataFrame(rows)
    cat.to_csv(out / "category_ti_ps.csv", index=False)
    ctab = {}
    for pocket in pockets + ["all"]:
        g = cat[cat.pocket == pocket].set_index("category").reindex([n for n, _ in CATEGORY_ROWS])
        ctab[(pocket, "%TI")] = [("–" if n < MIN_REAL else f"{v:.1f}") for v, n in zip(g.pct_TI, g.n_pos)]
        ctab[(pocket, "%PS")] = g.pct_PS.map("{:.1f}".format)
    ctab = pd.DataFrame(ctab)
    ctab.columns = [c[1] for c in ctab.columns]
    ctab.index.name = "Interaction"
    draw_table(ctab, "Interaction hot-spots recovered by AISM-placed probes, by interaction category" + label,
               out / "category_ti_ps.png", first_col_w=1.5, col_w=0.8, bold_last_col=False, group_rules=False,
               col_groups=[(POCKET_LABELS[p], 2) for p in pockets + ["all"]],
               note="Aromatic: phenol and benzene rings. Aliphatic: CH₃ groups. Both: aromatic + aliphatic. "
                    "H-bond: OH, NH and C=O groups. Ionic: NH₃⁺ and COO⁻.")

    # % de casos con al menos una interaccion recuperada
    all_pairs = site_m[["case_id", "probe", "pocket"]].drop_duplicates()
    pair_hits = real_m.groupby(["case_id", "probe", "pocket"]).found.any().reset_index()
    rows = []
    for probe in probes + ["all"]:
        for pocket in pockets + ["all"]:
            sel = lambda d: d[((d.probe == probe) | (probe == "all")) & ((d.pocket == pocket) | (pocket == "all"))]
            h, a = sel(pair_hits), sel(all_pairs)
            rows.append(dict(probe=probe, pocket=pocket, n_pairs=len(h), n_positive=int(h.found.sum()),
                             pct_cases=round(100 * h.found.mean(), 1) if len(h) else np.nan,
                             n_pairs_no_target=len(a) - len(h)))
    ch = pd.DataFrame(rows)
    ch.to_csv(out / "case_hits_probe.csv", index=False)

    cat_hits = []
    for name, cats in CATEGORY_ROWS:
        g = real_m[real_m.category.isin(cats)].groupby(["case_id", "probe", "pocket"]).found.any().reset_index()
        cat_hits.append(g.assign(category=name))
    cat_hits = pd.concat(cat_hits)
    rows = []
    for name, _ in CATEGORY_ROWS:
        for pocket in pockets + ["all"]:
            h = cat_hits[(cat_hits.category == name) & ((cat_hits.pocket == pocket) | (pocket == "all"))]
            rows.append(dict(category=name, pocket=pocket, n_pairs=len(h), n_positive=int(h.found.sum()),
                             pct_cases=round(100 * h.found.mean(), 1) if len(h) else np.nan))
    chc = pd.DataFrame(rows)
    chc.to_csv(out / "case_hits_category.csv", index=False)

    def pct_table(df, key, index, labels):
        t = {}
        for pocket in pockets + ["all"]:
            g = df[df.pocket == pocket].set_index(key).reindex(index)
            t[POCKET_LABELS[pocket]] = [("–" if n < MIN_REAL else f"{v:.1f}") for v, n in zip(g.pct_cases, g.n_pairs)]
        t = pd.DataFrame(t)
        t.index = labels
        return t

    order_c = ch[(ch.pocket == "all") & (ch.probe != "all")].sort_values("pct_cases", ascending=False).probe.tolist()
    t = pct_table(ch, "probe", order_c + ["all"], [p.capitalize() for p in order_c] + ["All probes"])
    t.index.name = "Probe"
    draw_table(t, "Cases in which the probe recovered at least one crystal interaction (%)" + label,
               out / "case_hits_probe.png", first_col_w=1.9, col_w=0.95, bold_last_col=True,
               note="Only case–probe pairs with at least one crystal interaction of a type the probe represents. "
                    f"– : fewer than {MIN_REAL} pairs.")
    t = pct_table(chc, "category", [n for n, _ in CATEGORY_ROWS], [n for n, _ in CATEGORY_ROWS])
    t.index.name = "Interaction"
    draw_table(t, "Cases in which the probe recovered at least one interaction, by category (%)" + label,
               out / "case_hits_category.png", first_col_w=1.6, col_w=0.95, bold_last_col=True,
               note="Case–probe pairs with at least one crystal interaction of the category. Both: aromatic + aliphatic.")

    print(ch[ch.pocket == "all"].to_string(index=False))
    print(chc.pivot(index="category", columns="pocket", values="pct_cases").to_string())
    print(f"interacciones reales consideradas (por sonda): {len(real_m)}  sitios de sonda: {len(site_m)}")
    print(summ[summ.pocket == "all"].to_string(index=False))
    print(by_type.pivot(index="probe", columns="ptype", values="pct_TI").rename(columns=TYPE_LABELS).to_string())
    print(cat[cat.pocket == "all"].to_string(index=False))
    print(f"-> {out}")
    return summ


if __name__ == "__main__":
    main()
