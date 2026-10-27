#!/usr/bin/env python3
"""Paso 4: tasas de acierto por sonda y relacion con la confianza de Boltz.

Usa results/probe_runs.csv y probe_copies.csv, filtrados por el QC de config.yaml
(ligando bien identificado, align_rmsd y seq_identity). "Acierto" = distancia minima
sonda-ligando <= hit_cutoff_main.

- Tasas de acierto: por prediccion (mejor copia).
- Confianza: por copia de sonda. Las metricas globales de Boltz (iptm, plddt, ...)
  se repiten en todas las copias de una prediccion; probe_plddt / probe_pae son
  propias de cada copia. Con x1 copia = prediccion.
  ligand_iptm no se analiza aparte: con un unico ligando es identico a iptm.
- Si config tiene `models` (p. ej. [0]) solo se usan esas muestras de Boltz.
- Si hay mas de una concentracion de sonda (Set_Replicas), todo se calcula por
  separado para cada una en confidence/x<n>/, y se juntan en confidence/ las tablas
  hit_rates_by_conc.csv y auc_by_conc.csv, y la comparacion sonda x concentracion
  (hit_rates_probe_conc.csv, fig6_hit_rates_probe_conc.png) con todos los datos y
  solo con los pares caso-sonda presentes en todas las concentraciones; esta ultima
  tambien como tablas PNG (table_hit_rates_common_pairs.png, table_hit_rates_all.png).

Salidas (results/confidence/):
  hit_rates.csv               % y conteos por sonda x tipo de bolsillo ('all' = agrupado)
  hit_rates_pct_by_pocket.csv solo %, filas = sondas + Total, columnas = bolsillo x criterio
  auc_by_probe.csv            AUC de cada metrica para separar aciertos (por sonda y global)
  thresholds.csv              precision/recall al filtrar por cuantiles de cada metrica
  fig1_hit_rates.png          tasas de acierto por sonda (apo vs holo)
  fig2_distance_ecdf.png      distribucion de la distancia minima por sonda
  fig3_auc_heatmap.png        AUC metrica x sonda
  fig4_hit_rate_vs_conf.png   tasa de acierto por decil de las mejores metricas
  fig5_conf_vs_distance.png   densidad confianza vs distancia
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from silc_boltz.config import load_config  # noqa: E402
from silc_boltz.tables import draw_table  # noqa: E402

# Metricas: (columna, etiqueta, signo) con signo -1 si "menor = mas confianza".
METRICS = [
    ("probe_plddt", "probe pLDDT", 1),
    ("probe_pae", "probe interface PAE", -1),
    ("iptm", "ipTM", 1),
    ("confidence_score", "confidence score", 1),
    ("complex_iplddt", "complex ipLDDT", 1),
    ("complex_ipde", "complex iPDE", -1),
    ("complex_plddt", "complex pLDDT", 1),
    ("complex_pde", "complex PDE", -1),
    ("ptm", "pTM", 1),
]

# Paleta de referencia (skill dataviz), modo claro.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
# Tipo de bolsillo: slots categoricos 1-3 de la paleta de referencia (validados all-pairs).
POCKET_COLOR = {"rigid": "#2a78d6", "flexible": "#eb6834", "cryptic": "#1baf7a"}
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
# Concentracion de sonda (ordinal): pasos de la rampa azul, claro -> oscuro.
CONC_COLOR = ["#6da7ec", "#2a78d6", "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#c43b3a", "#e34948", "#f0efec", "#2a78d6", "#1c5cab"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def auc(score, label):
    """AUC ROC (Mann-Whitney) con empates promediados; NaN si falta una clase."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    ok = ~np.isnan(score)
    score, label = score[ok], label[ok]
    npos, nneg = label.sum(), (~label).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(score)
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


def load(cfg):
    res = cfg["results_dir"]
    runs = pd.read_csv(res / "probe_runs.csv")
    copies = pd.read_csv(res / "probe_copies.csv")
    sites = pd.read_csv(res / "reference_sites.csv")
    qc = cfg["qc"]
    good_cases = set(sites.loc[sites.lig_match.isin(qc["lig_match_ok"]), "case_id"])
    runs["qc_pass"] = (runs.status.eq("ok") & runs.case_id.isin(good_cases)
                       & (runs.align_rmsd <= qc["max_align_rmsd"])
                       & (runs.seq_identity >= qc["min_seq_identity"]))
    if cfg.get("models") is not None:
        runs = runs[runs.model.isin(cfg["models"])].copy()
        print(f"Muestras de Boltz: {cfg['models']}")
    n0 = len(runs)
    runs = runs[runs.qc_pass].copy()
    print(f"QC: {len(runs)}/{n0} predicciones, {runs.case_id.nunique()} casos")
    glob_cols = [m for m, _, _ in METRICS if m in runs.columns]
    cls = pd.read_csv(cfg["pocket_classification"], usecols=["system", "classification"])
    runs = runs.merge(cls.rename(columns={"system": "case_id", "classification": "pocket"}), on="case_id", how="left")
    n1 = len(runs)
    runs = runs[runs.pocket.isin(cfg["pocket_classes"])].copy()
    print(f"Con tipo de bolsillo {cfg['pocket_classes']}: {len(runs)}/{n1} predicciones, "
          f"{runs.case_id.nunique()} casos")
    cut = cfg["hit_cutoff_main"]
    runs["hit"] = runs.best_min_dist <= cut
    copies = copies.merge(runs[["run", "model", "probe", "conc", "seq_state", "pocket", "case_id"] + glob_cols],
                          on=["run", "model"], how="inner")
    copies["hit"] = copies.min_dist <= cut
    return runs, copies


def probe_order(runs):
    return runs.groupby("probe").hit.mean().sort_values(ascending=False).index.tolist()


def fig_hit_rates(runs, order, cfg, out):
    cols = [(f"hit_{c:g}A", f"min. distance ≤ {c:g} Å", f"le_{c:g}A") for c in cfg["hit_cutoffs"]]
    cols.append(("best_in_site", f"in binding site (≤ {cfg['contact_cutoff']:g} Å of site)", "in_site"))
    pockets = [p for p in cfg["pocket_classes"] if p in set(runs.pocket)]

    # Tabla larga: % y conteos por sonda x tipo de bolsillo; 'all' = agrupado.
    rows = []
    for probe in order + ["all"]:
        for pocket in pockets + ["all"]:
            g = runs[((runs.probe == probe) | (probe == "all")) & ((runs.pocket == pocket) | (pocket == "all"))]
            row = dict(probe=probe, pocket=pocket, n_predictions=len(g), n_cases=g.case_id.nunique())
            for col, _, short in cols:
                row[f"n_{short}"] = int(g[col].sum())
                row[f"pct_{short}"] = round(100 * g[col].mean(), 1) if len(g) else np.nan
            rows.append(row)
    rates = pd.DataFrame(rows)
    rates.to_csv(out / "hit_rates.csv", index=False)

    # Tabla ancha solo con %, lista para el paper: filas = sondas + Total, columnas = bolsillo x criterio.
    wide = rates.pivot(index="probe", columns="pocket", values=[f"pct_{s}" for _, _, s in cols])
    wide = wide.reindex(index=order + ["all"], columns=[(f"pct_{s}", p) for p in pockets + ["all"] for _, _, s in cols])
    wide.columns = [f"{p}_{m[4:]}" for m, p in wide.columns]
    wide.index = [("Total" if p == "all" else p) for p in wide.index]
    n_row = rates[rates.probe == "all"].set_index("pocket").n_cases
    wide.index.name = "probe"
    wide.to_csv(out / "hit_rates_pct_by_pocket.csv")
    print("casos por tipo de bolsillo:", n_row.to_dict())

    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 1.2 + 0.3 * len(order) * len(pockets)),
                             sharey=True)
    y = np.arange(len(order))[::-1]
    h = 0.8 / len(pockets)
    for ax, (col, title, short) in zip(axes, cols):
        for k, pocket in enumerate(pockets):
            v = rates[rates.pocket == pocket].set_index("probe").reindex(order)[f"pct_{short}"].values
            yy = y + (len(pockets) - 1) / 2 * h - k * h
            ax.barh(yy, v, height=h - 0.04, color=POCKET_COLOR.get(pocket, MUTED),
                    label=f"{pocket} (n = {n_row[pocket]} cases)")
            for yi, vi in zip(yy, v):
                ax.text(vi + 1, yi, f"{vi:.0f}%", va="center", fontsize=6.5, color=INK2)
        ax.set_xlim(0, 100)
        ax.set_title(title, loc="left")
        ax.set_xlabel("% of predictions")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, order)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", ncol=len(pockets), fontsize=8,
               bbox_to_anchor=(0.005, 0.965))
    fig.suptitle("Where Boltz places the probe relative to the crystal ligand, by pocket type" + cfg["_label"],
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig1_hit_rates.png")
    plt.close(fig)


def fig_ecdf(runs, order, cfg, out):
    n = len(order)
    ncol = 4
    pockets = [p for p in cfg["pocket_classes"] if p in set(runs.pocket)]
    fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(12, 5.6), sharex=True, sharey=True)
    for ax, probe in zip(axes.flat, order):
        for pocket in pockets:
            d = np.sort(runs[(runs.probe == probe) & (runs.pocket == pocket)].best_min_dist.clip(lower=0.05))
            ax.step(d, 100 * np.arange(1, len(d) + 1) / len(d), where="post", lw=2,
                    color=POCKET_COLOR.get(pocket, MUTED), label=pocket)
        for c in cfg["hit_cutoffs"]:
            ax.axvline(c, color=AXIS, lw=1, ls="--", zorder=0)
        ax.set_xscale("log")
        ax.set_title(probe, loc="left")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("min. probe–ligand distance (Å)")
    for ax in axes[:, 0]:
        ax.set_ylabel("cumulative % of predictions")
    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Distribution of the probe–ligand distance by pocket type (dashed: "
                 + ", ".join(f"{c:g} Å" for c in cfg["hit_cutoffs"]) + ")" + cfg["_label"], x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig2_distance_ecdf.png")
    plt.close(fig)


def auc_table(copies, order):
    rows = []
    for probe in order + ["all"]:
        sub = copies if probe == "all" else copies[copies.probe == probe]
        for m, label, sign in METRICS:
            rows.append(dict(probe=probe, metric=m, label=label, n=len(sub), n_hit=int(sub.hit.sum()),
                             auc=auc(sign * sub[m], sub.hit)))
    return pd.DataFrame(rows)


def fig_auc(tab, order, cfg, out):
    mat = tab.pivot(index="label", columns="probe", values="auc")
    metric_order = (tab[tab.probe == "all"].sort_values("auc", ascending=False).label.tolist())
    mat = mat.loc[metric_order, order + ["all"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    im = ax.imshow(mat.values, cmap=DIVERGING, vmin=0.2, vmax=0.8, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(mat.shape[1]), [c if c != "all" else "ALL" for c in mat.columns], rotation=30, ha="right")
    ax.set_yticks(range(mat.shape[0]), mat.index)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(len(order) - 0.5, color=SURFACE, lw=4)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color=SURFACE if abs(v - 0.5) > 0.2 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("ROC AUC (0.5 = no better than random)")
    cb.outline.set_visible(False)
    ax.set_title(f"How well each Boltz confidence metric separates probes ≤ {cfg['hit_cutoff_main']:g} Å "
                 "from the crystal ligand" + cfg["_label"], loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig3_auc_heatmap.png")
    plt.close(fig)


def fig_deciles(copies, tab, cfg, out, n_metrics=4):
    top = tab[tab.probe == "all"].sort_values("auc", ascending=False).head(n_metrics)
    base = copies.hit.mean()
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 3.6), sharey=True)
    for ax, (_, t) in zip(axes, top.iterrows()):
        sub = copies[[t.metric, "hit"]].dropna()
        sub["bin"] = pd.qcut(sub[t.metric], 10, labels=False, duplicates="drop")
        g = sub.groupby("bin").agg(x=(t.metric, "median"), k=("hit", "sum"), n=("hit", "size"))
        lo, hi = 100 * np.array(wilson(g.k.values, g.n.values))
        p = 100 * g.k / g.n
        ax.fill_between(g.x, lo, hi, color=POCKET_COLOR["rigid"], alpha=0.15, lw=0)
        ax.plot(g.x, p, color=POCKET_COLOR["rigid"], lw=2, marker="o", ms=4)
        ax.axhline(100 * base, color=MUTED, lw=1, ls="--")
        ax.set_title(f"{t.label}  (AUC {t.auc:.2f})", loc="left")
        ax.set_xlabel(t.label + ("  (lower = more confident)" if dict((m, s) for m, _, s in METRICS)[t.metric] < 0 else ""))
        ax.set_ylim(0, 100)
    axes[0].set_ylabel(f"% of probes ≤ {cfg['hit_cutoff_main']:g} Å from ligand")
    axes[-1].text(axes[-1].get_xlim()[1], 100 * base + 2, "overall", ha="right", fontsize=7, color=MUTED)
    fig.suptitle("Hit rate by confidence decile, all probes pooled (band: 95% Wilson CI)" + cfg["_label"],
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig4_hit_rate_vs_conf.png")
    plt.close(fig)


def fig_density(copies, tab, cfg, out):
    top = tab[tab.probe == "all"].sort_values("auc", ascending=False)
    picks = [top.iloc[0]] + [r for _, r in top.iterrows() if r.metric in ("iptm",) and r.metric != top.iloc[0].metric][:1]
    fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, picks):
        sub = copies[[t.metric, "min_dist"]].dropna()
        hb = ax.hexbin(sub[t.metric], sub.min_dist.clip(lower=0.1), gridsize=45, yscale="log",
                       cmap=SEQ_BLUE, mincnt=1, bins="log", linewidths=0.2, edgecolors=SURFACE)
        ax.axhline(cfg["hit_cutoff_main"], color=INK2, lw=1, ls="--")
        ax.set_xlabel(t.label)
        ax.set_title(f"{t.label} vs distance" + cfg["_label"], loc="left")
        ax.grid(False)
    axes[0].set_ylabel("min. probe–ligand distance (Å)")
    cb = fig.colorbar(hb, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("probe copies (log)")
    cb.outline.set_visible(False)
    fig.savefig(out / "fig5_conf_vs_distance.png")
    plt.close(fig)


def probe_conc_rates(runs, cfg, out):
    """% de acierto por sonda x concentracion: todos los datos y pares caso-sonda comunes."""
    concs = sorted(runs.conc.unique())
    common = set.intersection(*[set(zip(g.case_id, g.probe)) for _, g in runs.groupby("conc")])
    sets = {"all": runs, "common": runs[[k in common for k in zip(runs.case_id, runs.probe)]]}
    rows = []
    for subset, df in sets.items():
        for probe in sorted(df.probe.unique()) + ["all"]:
            for conc in concs:
                g = df[(df.conc == conc) & ((df.probe == probe) | (probe == "all"))]
                k, n = int(g.hit.sum()), len(g)
                lo, hi = wilson(k, n) if n else (np.nan, np.nan)
                rows.append(dict(subset=subset, probe=probe, conc=conc, n_predictions=n,
                                 n_cases=g.case_id.nunique(), n_hit=k, pct_hit=100 * k / n if n else np.nan,
                                 ci_low=100 * lo, ci_high=100 * hi))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "hit_rates_probe_conc.csv", index=False, float_format="%.1f")

    order = (tab[(tab.subset == "all") & (tab.conc == concs[0]) & (tab.probe != "all")]
             .sort_values("pct_hit", ascending=False).probe.tolist()) + ["all"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 1.4 + 0.42 * len(order)), sharey=True)
    y = np.arange(len(order))[::-1].astype(float)
    y[-1] -= 0.4                                     # separa la fila "all"
    h = 0.8 / len(concs)
    titles = {"all": "All predictions",
              "common": f"Case–probe pairs present at every concentration ({len(common)})"}
    for ax, (subset, _) in zip(axes, sets.items()):
        t = tab[tab.subset == subset]
        for k, conc in enumerate(concs):
            v = t[t.conc == conc].set_index("probe").reindex(order)
            yy = y + (len(concs) - 1) / 2 * h - k * h
            ax.barh(yy, v.pct_hit, height=h - 0.04, color=CONC_COLOR[k % len(CONC_COLOR)],
                    label=f"{conc} cop{'y' if conc == 1 else 'ies'}")
            ax.errorbar(v.pct_hit, yy, xerr=[v.pct_hit - v.ci_low, v.ci_high - v.pct_hit],
                        fmt="none", ecolor=INK2, elinewidth=0.7, capsize=0)
            for yi, vi, ni in zip(yy, v.pct_hit, v.n_predictions):
                ax.text(101, yi, f"{vi:.0f}% (n={ni})", va="center", fontsize=6, color=INK2)
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"% of predictions with min. distance ≤ {cfg['hit_cutoff_main']:g} Å")
        ax.set_title(titles[subset], loc="left")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [o if o != "all" else "ALL" for o in order])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", ncol=len(concs), fontsize=8,
               bbox_to_anchor=(0.005, 0.95))
    fig.suptitle("Hit rate by probe and probe concentration (bars: 95% Wilson CI)" + cfg["_label"],
                 x=0.01, y=0.995, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.93, 0.92), w_pad=6)
    fig.savefig(out / "fig6_hit_rates_probe_conc.png")
    plt.close(fig)

    # Misma informacion como grafico de puntos: una fila por sonda, un punto por concentracion.
    fig, axes = plt.subplots(1, 2, figsize=(10, 1.4 + 0.42 * len(order)), sharey=True)
    dy = 0.18
    for ax, subset in zip(axes, ("common", "all")):
        t = tab[tab.subset == subset]
        for k, conc in enumerate(concs):
            v = t[t.conc == conc].set_index("probe").reindex(order)
            yy = y + ((len(concs) - 1) / 2 - k) * dy
            ax.errorbar(v.pct_hit, yy, xerr=[v.pct_hit - v.ci_low, v.ci_high - v.pct_hit],
                        fmt="none", ecolor=CONC_COLOR[k % len(CONC_COLOR)], alpha=0.45, elinewidth=1, zorder=2)
            ax.scatter(v.pct_hit, yy, s=42, color=CONC_COLOR[k % len(CONC_COLOR)], edgecolor="white",
                       linewidth=1, zorder=3, label=f"{conc} cop{'y' if conc == 1 else 'ies'}")
        ax.axhline((y[-1] + y[-2]) / 2, color=GRID, lw=0.8)
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"% of predictions with min. distance ≤ {cfg['hit_cutoff_main']:g} Å")
        ax.set_title(titles[subset], loc="left")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [o.capitalize() if o != "all" else "ALL" for o in order])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", ncol=len(concs), fontsize=8,
               bbox_to_anchor=(0.005, 0.95))
    fig.suptitle("Hit rate by probe and probe concentration (lines: 95% Wilson CI)" + cfg["_label"],
                 x=0.01, y=0.995, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.945), w_pad=3)
    fig.savefig(out / "fig7_hit_rates_probe_conc_dots.png")
    plt.close(fig)

    # Tabla PNG (paper) de los pares comunes, mismo orden de sondas que la figura.
    t = tab[tab.subset == "common"]
    pct = t.pivot(index="probe", columns="conc", values="pct_hit").reindex(order)
    fmt = pct.apply(lambda col: col.map(lambda v: f"{v:.0f}")).astype(object)
    fmt.columns = [f"{c} cop{'y' if c == 1 else 'ies'} (%)" for c in pct.columns]
    fmt["Pairs (n)"] = t[t.conc == concs[0]].set_index("probe").reindex(order).n_predictions.astype(int).astype(str)
    fmt.index = ["Total" if o == "all" else o.capitalize() for o in order]
    fmt.index.name = "Probe"
    draw_table(fmt, "Hit rate by probe and probe concentration (common case–probe pairs)",
               out / "table_hit_rates_common_pairs.png", bold_last_col=False, first_col_w=1.9, col_w=1.35,
               note=f"Hit: best probe copy ≤ {cfg['hit_cutoff_main']:g} Å from the crystal ligand. "
                    f"{len(common)} pairs ({sets['common'].case_id.nunique()} cases) predicted at every "
                    "concentration" + cfg["_label"].replace(",", ";", 1) + ".")

    # Tabla PNG con todas las predicciones de cada concentracion (el n cambia entre columnas).
    t = tab[tab.subset == "all"]
    labels = [f"{c} cop{'y' if c == 1 else 'ies'}" for c in concs]
    pct = t.pivot(index="probe", columns="conc", values="pct_hit").reindex(order)
    num = t.pivot(index="probe", columns="conc", values="n_predictions").reindex(order)
    fmt = pd.concat([pct.apply(lambda col: col.map(lambda v: f"{v:.0f}")),
                     num.apply(lambda col: col.map(lambda v: f"{v:,.0f}"))], axis=1).astype(object)
    fmt.columns = labels + labels
    fmt.index = ["Total" if o == "all" else o.capitalize() for o in order]
    fmt.index.name = "Probe"
    draw_table(fmt, "Hit rate by probe and probe concentration (all predictions)",
               out / "table_hit_rates_all.png", bold_last_col=False, first_col_w=1.9, col_w=1.0,
               col_groups=[("Hit rate (%)", len(concs)), ("Predictions (n)", len(concs))],
               note=f"Hit: best probe copy ≤ {cfg['hit_cutoff_main']:g} Å from the crystal ligand. "
                    "All predictions per concentration (cases differ between columns)"
                    + cfg["_label"].replace(",", ";", 1) + ".")
    return tab


def thresholds(copies, out):
    rows = []
    total_hits = copies.hit.sum()
    for m, label, sign in METRICS:
        s = sign * copies[m]
        for q in (0.0, 0.25, 0.5, 0.75, 0.9):
            thr = s.quantile(q)
            keep = s >= thr
            rows.append(dict(metric=m, keep_top_fraction=round(1 - q, 2), threshold=sign * thr,
                             n_kept=int(keep.sum()), precision=copies.hit[keep].mean(),
                             recall=copies.hit[keep].sum() / total_hits))
    pd.DataFrame(rows).to_csv(out / "thresholds.csv", index=False, float_format="%.4f")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = cfg["results_dir"] / "confidence"
    base.mkdir(parents=True, exist_ok=True)

    all_runs, all_copies = load(cfg)
    concs = sorted(all_runs.conc.unique())
    split = len(concs) > 1
    models = cfg.get("models")
    model_label = f", Boltz sample {', '.join(map(str, models))}" if models is not None else ""
    rates_all, tabs_all = [], []
    for conc in concs:
        runs, copies = all_runs[all_runs.conc == conc], all_copies[all_copies.conc == conc]
        out = base / f"x{conc}" if split else base
        out.mkdir(parents=True, exist_ok=True)
        cfg["_label"] = (f" — {conc} probe cop{'y' if conc == 1 else 'ies'}" if split else "") + model_label
        print(f"\n== x{conc}: {len(runs)} predicciones, {len(copies)} copias de sonda")
        order = probe_order(runs)
        fig_hit_rates(runs, order, cfg, out)
        fig_ecdf(runs, order, cfg, out)
        tab = auc_table(copies, order)
        tab.to_csv(out / "auc_by_probe.csv", index=False, float_format="%.4f")
        fig_auc(tab, order, cfg, out)
        fig_deciles(copies, tab, cfg, out)
        fig_density(copies, tab, cfg, out)
        thresholds(copies, out)
        print(tab[tab.probe == "all"].sort_values("auc", ascending=False)[["label", "auc", "n", "n_hit"]]
              .to_string(index=False, float_format="%.3f"))
        print(f"-> {out}")
        rates_all.append(pd.read_csv(out / "hit_rates.csv").assign(conc=conc))
        tabs_all.append(tab.assign(conc=conc))

    if split:
        rates = pd.concat(rates_all)
        rates.insert(0, "conc", rates.pop("conc"))
        rates.to_csv(base / "hit_rates_by_conc.csv", index=False)
        tabs = pd.concat(tabs_all)
        tabs.insert(0, "conc", tabs.pop("conc"))
        tabs.to_csv(base / "auc_by_conc.csv", index=False, float_format="%.4f")
        print("\n% acierto (min. distance <= 2 A), todas las sondas:")
        print(rates[rates.probe == "all"].pivot(index="pocket", columns="conc", values="pct_le_2A").to_string())
        print("\nAUC global por concentracion:")
        print(tabs[tabs.probe == "all"].pivot(index="label", columns="conc", values="auc")
              .sort_values(concs[0], ascending=False).to_string(float_format="%.3f"))
        print(f"-> {base / 'hit_rates_by_conc.csv'}, {base / 'auc_by_conc.csv'}")
        auc_all = tabs[tabs.probe == "all"].pivot(index="label", columns="conc", values="auc")
        auc_all = auc_all.sort_values(concs[-1], ascending=False)
        n_hit = tabs[tabs.probe == "all"].groupby("conc")[["n", "n_hit"]].first()
        fmt = auc_all.apply(lambda col: col.map("{:.2f}".format)).astype(object)
        fmt.columns = [f"{c} cop{'y' if c == 1 else 'ies'}" for c in auc_all.columns]
        fmt.index.name = "Metric"
        draw_table(fmt, "ROC AUC of Boltz confidence metrics for hits, by probe concentration",
                   base / "table_auc_by_conc.png", bold_last_col=False, first_col_w=2.3, col_w=1.15,
                   note=f"Per probe copy; hit ≤ {cfg['hit_cutoff_main']:g} Å. Copies (hits): "
                        + ", ".join(f"x{c} {int(r.n)} ({int(r.n_hit)})" for c, r in n_hit.iterrows())
                        + model_label.replace(",", ";", 1) + ".")
        cfg["_label"] = model_label
        pc = probe_conc_rates(all_runs, cfg, base)
        print("\n% acierto por sonda x concentracion (todos | pares comunes):")
        print(pc.pivot_table(index="probe", columns=["subset", "conc"], values="pct_hit").round(1).to_string())
        print(f"-> {base / 'hit_rates_probe_conc.csv'}, {base / 'fig6_hit_rates_probe_conc.png'}")


if __name__ == "__main__":
    main()
