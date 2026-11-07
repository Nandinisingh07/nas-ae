"""Run every experiment of the paper for one dataset and several seeds. Results are cached as JSON (resumable).

  python -m nasae.experiments --dataset cifar10 --seeds 0 1 2
"""
import argparse, json, os, time
import numpy as np
import torch
from .config import HP, LAMBDA, DIMS, manual_spec
from .data import load_dataset
from .models import FixedAE, match_width, random_genotype
from .train import search, train_fixed, evaluate
from .utils import set_seed


def cached(path, fn):
    if os.path.exists(path):
        print(f"  (cached) {path}")
        return json.load(open(path))
    t0 = time.time()
    res = fn()
    res["wall_s"] = time.time() - t0
    json.dump(res, open(path, "w"), indent=1)
    return res


def run_fixed(data, spec, geno, lam, hp, seed, dev, tag, beta=0.0):
    model, hist = train_fixed(data, spec, geno, lam, hp, seed, dev, beta=beta, tag=tag)
    res = evaluate(model, data, spec, hp, seed)
    res.update(genotype=geno, history=hist)
    return res


def run_nasae(data, spec, lam, hp, seed, dev, tag):
    geno, alpha, shist, t = search(data, spec, lam, hp, seed, dev)
    print(f"  derived architecture: {geno}")
    res = run_fixed(data, spec, geno, lam, hp, seed, dev, tag)
    res.update(alpha=alpha, search_history=shist, search_s=t)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar10", "ecg5000", "kdd99"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="results")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--ablation", action="store_true", help="also run lambda3=0 and lambda2=0 variants (paper Table III)")
    ap.add_argument("--quick", action="store_true", help="tiny run (few epochs, small data) to check everything works")
    ap.add_argument("--synthetic", action="store_true", help="random fake data; ONLY to smoke-test the code")
    ap.add_argument("--max_train", type=int, default=None, help="subsample training set (speed)")
    ap.add_argument("--search_epochs", type=int); ap.add_argument("--retrain_epochs", type=int)
    ap.add_argument("--baseline_loss", choices=["dual", "recon"], default="dual",
                    help="dual: baselines use the same L_recon+L_task loss as NAS-AE (architecture-only comparison, default)")
    a = ap.parse_args()

    hp = dict(HP)
    if a.quick:
        hp.update(search_epochs=2, retrain_epochs=3, probe_steps=200, random_samples=2)
        a.max_train = a.max_train or 3000
    if a.search_epochs: hp["search_epochs"] = a.search_epochs
    if a.retrain_epochs: hp["retrain_epochs"] = a.retrain_epochs
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  hp={hp}")
    base_lam = dict(LAMBDA) if a.baseline_loss == "dual" else dict(LAMBDA, l2=0.0)

    for seed in a.seeds:
        out = os.path.join(a.out, a.dataset + ("_synthetic" if a.synthetic else ""), f"seed{seed}")
        os.makedirs(out, exist_ok=True)
        set_seed(seed)
        spec, data = load_dataset(a.dataset, seed, a.data_root, a.synthetic, a.max_train, dev)
        print(f"\n=== {a.dataset} seed {seed}: train {len(data['ytr'])} val {len(data['yva'])} test {len(data['yte'])} ===")
        P = lambda n: os.path.join(out, n + ".json")

        # 1) NAS-AE: search -> discretize -> retrain from scratch
        print("[NAS-AE]")
        nas = cached(P("nasae"), lambda: run_nasae(data, spec, LAMBDA, hp, seed, dev, "nasae"))

        # 2) fixed-dimension sweep (paper Fig.1). d=32 is the Manual AE; d in {16,32,64} is the Grid-Search AE
        sweep = {}
        for d in DIMS:
            print(f"[sweep/manual dim={d}]")
            sweep[d] = cached(P(f"sweep_d{d}"), lambda d=d: run_fixed(data, spec, manual_spec(d, hp["base_channels"]), base_lam, hp, seed, dev, f"d{d}"))
        manual = sweep[32]
        json.dump(manual, open(P("manual"), "w"))
        gd = max([16, 32, 64], key=lambda d: sweep[d]["val"][spec.primary])       # pick by VALIDATION metric
        json.dump(dict(sweep[gd], chosen_dim=gd), open(P("grid"), "w"))

        # 3) VAE with ~ same parameter count as the derived NAS-AE
        print("[VAE]")
        def _vae():
            g = manual_spec(32); g["vae"] = True
            g["C"] = match_width(spec, g, int(nas["params_M"] * 1e6))
            return run_fixed(data, spec, g, base_lam, hp, seed, dev, "vae", beta=hp["vae_beta"])
        cached(P("vae"), _vae)

        # 4) Random NAS: random architectures from same space, no bilevel optimisation; pick best on validation
        print("[Random NAS]")
        def _random():
            rng = np.random.RandomState(1000 + seed)
            runs = []
            for k in range(hp["random_samples"]):
                g = random_genotype(rng, hp["base_channels"])
                runs.append(run_fixed(data, spec, g, LAMBDA, hp, seed, dev, f"rand{k}"))
            best = max(runs, key=lambda r: r["val"][spec.primary])
            return dict(best, all_test=[r["test"][spec.primary] for r in runs],
                        all_genotypes=[r["genotype"] for r in runs])
        cached(P("random"), _random)

        # 5) ablations (Table III): lambda3 = 0 and lambda2 = 0
        if a.ablation:
            for name, lam in [("abl_no_complexity", dict(LAMBDA, l3=0.0)), ("abl_no_task", dict(LAMBDA, l2=0.0))]:
                print(f"[{name}]")
                cached(P(name), lambda lam=lam, name=name: run_nasae(data, spec, lam, hp, seed, dev, name))
        print(f"seed {seed} done -> {out}")


if __name__ == "__main__":
    main()
