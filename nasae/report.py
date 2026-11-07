"""Build the paper's tables and figures from REAL results.   python -m nasae.report --results results"""
import argparse, glob, json, os
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import DIMS, DEPTHS

NAMES = {"cifar10": "CIFAR-10 (image)", "ecg5000": "ECG5000 (time-series)", "kdd99": "KDD 99 (tabular)"}
PRIM = {"cifar10": "acc", "ecg5000": "f1", "kdd99": "acc"}
METHODS = [("manual", "Manual AE"), ("grid", "Grid Search AE"), ("vae", "VAE"), ("random", "Random NAS"), ("nasae", "NAS-AE")]


def load(root):
    R = {}
    for d in sorted(glob.glob(os.path.join(root, "*", "seed*"))):
        ds = os.path.basename(os.path.dirname(d)).replace("_synthetic", "")
        for f in glob.glob(os.path.join(d, "*.json")):
            R.setdefault(ds, {}).setdefault(os.path.basename(f)[:-5], []).append(json.load(open(f)))
    return R


def ms(vals, fmt="{:.2f}"):
    v = np.array(vals, float)
    return (fmt + " ± " + fmt).format(v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)


def metric(r, ds): return r["test"][PRIM[ds]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "report"); os.makedirs(out, exist_ok=True)
    R = load(a.results)
    md = ["# NAS-AE: results from real runs\n", "Values are mean ± std over seeds (n = number of seeds shown).\n"]

    # ---------- Table: all methods (Fig. 2 numbers)
    md += ["## Table A. Task metric, all methods (test set, linear probe on frozen z)\n",
           "| Dataset (metric) | n | " + " | ".join(n for _, n in METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
    for ds, rr in R.items():
        mult = 1 if PRIM[ds] == "f1" else 1
        fmt = "{:.3f}" if PRIM[ds] == "f1" else "{:.2f}"
        row = [f"{NAMES[ds]} ({PRIM[ds]})", str(len(rr.get("nasae", [])))]
        for k, _ in METHODS:
            row.append(ms([metric(r, ds) for r in rr[k]], fmt) if k in rr else "-")
        md.append("| " + " | ".join(row) + " |")

    # ---------- Table II
    md += ["\n## Table B. Architectures found by NAS-AE (per seed) and gain over Manual AE\n",
           "| Dataset | seed genotypes (dim/depth/act/skips) | Recon MSE | Task metric | Δ vs Manual (abs) | Δ vs Manual (relative %) | Δ vs Grid (abs) |", "|---|---|---|---|---|---|---|"]
    for ds, rr in R.items():
        if "nasae" not in rr: continue
        g = "; ".join(f"{r['genotype']['dim']}/{r['genotype']['depth']}/{r['genotype']['act']}/{''.join(map(str, r['genotype']['skips']))}" for r in rr["nasae"])
        n = [metric(r, ds) for r in rr["nasae"]]; m = [metric(r, ds) for r in rr["manual"]]; gr = [metric(r, ds) for r in rr["grid"]]
        k = min(len(n), len(m), len(gr))
        d_abs = np.array(n[:k]) - np.array(m[:k]); d_rel = 100 * d_abs / np.array(m[:k]); dg = np.array(n[:k]) - np.array(gr[:k])
        fm = "{:.3f}" if PRIM[ds] == "f1" else "{:.2f}"
        md.append(f"| {NAMES[ds]} | {g} | {ms([r['test']['recon_mse'] for r in rr['nasae']], '{:.4f}')} | {ms(n, fm)} | {ms(d_abs, fm)} | {ms(d_rel)} | {ms(dg, fm)} |")
    md.append("\nInterpretation guide: if the std of Δ is as large as its mean, the 'gain' is not distinguishable from seed noise.\n")

    # ---------- Table III (ablation)
    if "cifar10" in R and "abl_no_complexity" in R["cifar10"]:
        rr = R["cifar10"]
        md += ["## Table C. Ablation on CIFAR-10\n", "| Configuration | Recon MSE | Accuracy (%) | Params (M) | MFLOPs/sample |", "|---|---|---|---|---|"]
        for key, name in [("nasae", "Full NAS-AE"), ("abl_no_complexity", "No complexity penalty (λ3=0)"), ("abl_no_task", "No task loss (λ2=0)"), ("random", "Random NAS (no bilevel)")]:
            if key in rr:
                md.append(f"| {name} | {ms([r['test']['recon_mse'] for r in rr[key]], '{:.4f}')} | {ms([r['test']['acc'] for r in rr[key]])} | "
                          f"{ms([r['params_M'] for r in rr[key]])} | {ms([r['mflops'] for r in rr[key]], '{:.1f}')} |")

    # ---------- Fig 1: performance vs bottleneck dim
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for i, (ds, rr) in enumerate(R.items()):
        if i >= 3 or "sweep_d8" not in rr: continue
        M = np.array([[metric(r, ds) for r in rr[f"sweep_d{d}"]] for d in DIMS])
        ax[i].errorbar(range(len(DIMS)), M.mean(1), yerr=M.std(1, ddof=1) if M.shape[1] > 1 else None, marker="o", capsize=3)
        ax[i].set_xticks(range(len(DIMS))); ax[i].set_xticklabels(DIMS); ax[i].set_xlabel("bottleneck dim")
        ax[i].set_title(NAMES[ds]); ax[i].set_ylabel(PRIM[ds]); ax[i].grid(alpha=.3)
    fig.suptitle("Fig. 1  Task performance vs bottleneck dimension (Manual-AE structure)"); fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_dim_sweep.png"), dpi=160); plt.close(fig)
    md.append("\n![fig1](fig1_dim_sweep.png)\n")

    # ---------- Fig 2: baselines bar chart
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for i, (ds, rr) in enumerate(R.items()):
        if i >= 3 or "nasae" not in rr: continue
        ks = [(k, n) for k, n in METHODS if k in rr]
        v = [[metric(r, ds) for r in rr[k]] for k, _ in ks]
        ax[i].bar(range(len(ks)), [np.mean(x) for x in v], yerr=[np.std(x, ddof=1) if len(x) > 1 else 0 for x in v], capsize=3,
                  color=["#999"] * (len(ks) - 1) + ["#6a3fb5"])
        ax[i].set_xticks(range(len(ks))); ax[i].set_xticklabels([n for _, n in ks], rotation=30, ha="right", fontsize=8)
        lo = min(np.mean(x) for x in v); hi = max(np.mean(x) for x in v)
        ax[i].set_ylim(lo - 0.5 * (hi - lo + 1e-3) - 0.02 * abs(lo), hi + 0.3 * (hi - lo + 1e-3)); ax[i].set_title(NAMES[ds]); ax[i].set_ylabel(PRIM[ds])
    fig.suptitle("Fig. 2  Task metric vs baselines (y-axis truncated)"); fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_baselines.png"), dpi=160); plt.close(fig)
    md.append("![fig2](fig2_baselines.png)\n")

    # ---------- Fig 3: convergence (search epochs then retrain epochs), CIFAR-10
    ds = "cifar10" if "cifar10" in R else next(iter(R))
    if "nasae" in R[ds]:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        H = [r["search_history"] + r["history"] for r in R[ds]["nasae"]]
        T = min(len(h) for h in H); ns = len(R[ds]["nasae"][0]["search_history"])
        for j, (key, lab) in enumerate([("val_recon", "val recon MSE"), ("val_acc", "val head accuracy")]):
            Y = np.array([[e[key] for e in h[:T]] for h in H])
            ax[j].plot(range(1, T + 1), Y.mean(0)); ax[j].fill_between(range(1, T + 1), Y.min(0), Y.max(0), alpha=.2)
            ax[j].axvline(ns, ls="--", c="k"); ax[j].set_xlabel("epoch (search | retrain)"); ax[j].set_ylabel(lab); ax[j].grid(alpha=.3)
        fig.suptitle(f"Fig. 3  Bilevel training convergence, {NAMES[ds]} (dashed = discretisation)"); fig.tight_layout()
        fig.savefig(os.path.join(out, "fig3_convergence.png"), dpi=160); plt.close(fig)
        md.append("![fig3](fig3_convergence.png)\n")

    # ---------- Fig 4: alpha heatmaps  P(dim) x P(depth)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    for i, (ds, rr) in enumerate(R.items()):
        if i >= 3 or "nasae" not in rr: continue
        d = np.mean([r["alpha"]["d"] for r in rr["nasae"]], 0); L = np.mean([r["alpha"]["L"] for r in rr["nasae"]], 0)
        im = ax[i].imshow(np.outer(L, d), aspect="auto", cmap="viridis"); ax[i].set_xticks(range(len(DIMS))); ax[i].set_xticklabels(DIMS)
        ax[i].set_yticks(range(len(DEPTHS))); ax[i].set_yticklabels(DEPTHS); ax[i].set_xlabel("dim"); ax[i].set_ylabel("depth"); ax[i].set_title(NAMES[ds])
        fig.colorbar(im, ax=ax[i])
    fig.suptitle("Fig. 4  softmax(alpha_L) x softmax(alpha_d) (seed-averaged)"); fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_alpha.png"), dpi=160); plt.close(fig)
    md.append("![fig4](fig4_alpha.png)\n")

    # ---------- Search-cost
    md += ["## Table D. Cost\n", "| Dataset | search time (s) | retrain time (s) | derived params (M) | derived MFLOPs |", "|---|---|---|---|---|"]
    for ds, rr in R.items():
        if "nasae" in rr:
            md.append(f"| {NAMES[ds]} | {ms([r['search_s'] for r in rr['nasae']], '{:.0f}')} | {ms([r['wall_s'] - r['search_s'] for r in rr['nasae']], '{:.0f}')} | "
                      f"{ms([r['params_M'] for r in rr['nasae']])} | {ms([r['mflops'] for r in rr['nasae']], '{:.1f}')} |")
    open(os.path.join(out, "RESULTS.md"), "w").write("\n".join(md))
    print("\n".join(md)); print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
