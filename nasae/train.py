"""Bilevel search (Algorithm 1), retraining, linear-probe evaluation."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, balanced_accuracy_score
from .config import HP, LAMBDA, BATCH
from .models import SuperAE, FixedAE
from .utils import count_params, count_flops


def batches(X, y, bs, shuffle, drop_last=False):
    n = len(X)
    idx = torch.randperm(n, device=X.device) if shuffle else torch.arange(n, device=X.device)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        if drop_last and len(j) < bs:
            break
        yield X[j], y[j]


def task_loss(logits, y, spec):
    if spec.task == "binary":
        return F.binary_cross_entropy_with_logits(logits.squeeze(1), y.float())
    return F.cross_entropy(logits, y)


def predict(logits, spec):
    return (logits.squeeze(1) > 0).long() if spec.task == "binary" else logits.argmax(1)


def total_loss(model, x, y, spec, lam, beta=0.0, with_complexity=False):
    """Eq. 5.  L_recon is per-element MSE (Eq. 6 divided by input size, so lambdas are on a sane scale)."""
    recon, logits, z, kl = model(x)
    lr, lt = F.mse_loss(recon, x), task_loss(logits, y, spec)
    loss = lam["l1"] * lr + lam["l2"] * lt + beta * kl
    if with_complexity:
        loss = loss + lam["l3"] * model.complexity()
    return loss, lr.detach(), (predict(logits, spec) == y).float().mean().detach()


@torch.no_grad()
def eval_split(model, X, y, spec, bs=1024):
    model.eval()
    se, n, correct, zs = 0., 0, 0., []
    for xb, yb in batches(X, y, bs, False):
        recon, logits, z, _ = model(xb)
        se += F.mse_loss(recon, xb, reduction="sum").item()
        n += xb.numel()
        correct += (predict(logits, spec) == yb).sum().item()
        zs.append(z)
    return se / n, correct / len(X), torch.cat(zs)


# ------------------------------------------------------------------ bilevel search
def search(data, spec, lam, hp, seed, device, log=print):
    """First-order DARTS: per iteration 1 step on W (train half) then 1 step on alpha (val half)."""
    model = SuperAE(spec, hp["base_channels"]).to(device)
    optW = torch.optim.Adam(model.weights(), lr=hp["lr_w"])
    optA = torch.optim.Adam(model.alphas(), lr=hp["lr_alpha"])
    X, y = data["Xtr"], data["ytr"]
    perm = torch.randperm(len(X), device=X.device)
    h = len(X) // 2
    Xw, yw, Xa, ya = X[perm[:h]], y[perm[:h]], X[perm[h:]], y[perm[h:]]
    bs = BATCH[spec.name]
    hist, t0 = [], time.time()
    for ep in range(hp["search_epochs"]):
        model.train()
        a_iter = batches(Xa, ya, bs, True, drop_last=True)
        tl, tr_, ta, k = 0., 0., 0., 0
        for xb, yb in batches(Xw, yw, bs, True, drop_last=True):
            try:
                xa, yb_a = next(a_iter)
            except StopIteration:
                a_iter = batches(Xa, ya, bs, True, drop_last=True); xa, yb_a = next(a_iter)
            # 1) W step on D_train
            loss, lr_, acc = total_loss(model, xb, yb, spec, lam, with_complexity=True)
            optW.zero_grad(); optA.zero_grad(); loss.backward(); optW.step()
            # 2) alpha step on D_val
            loss_a, _, _ = total_loss(model, xa, yb_a, spec, lam, with_complexity=True)
            optW.zero_grad(); optA.zero_grad(); loss_a.backward(); optA.step()
            tl += loss.item(); tr_ += lr_.item(); ta += acc.item(); k += 1
        vr, vacc, _ = eval_split(model, data["Xva"], data["yva"], spec)
        rec = dict(epoch=ep + 1, phase="search", train_loss=tl / k, train_recon=tr_ / k, train_acc=ta / k,
                   val_recon=vr, val_acc=vacc, complexity_mflops=float(model.complexity().detach()),
                   genotype=model.genotype(), alpha=model.alpha_snapshot())
        hist.append(rec)
        log(f"  [search {ep + 1:2d}/{hp['search_epochs']}] loss {rec['train_loss']:.4f} val_recon {vr:.4f} "
            f"val_acc {vacc:.3f} C(a) {rec['complexity_mflops']:.1f}MF  geno {rec['genotype']['dim']}/{rec['genotype']['depth']}/{rec['genotype']['act']}")
    return model.genotype(), model.alpha_snapshot(), hist, time.time() - t0


# ------------------------------------------------------------------ retrain derived / baseline AE
def train_fixed(data, spec, geno, lam, hp, seed, device, beta=0.0, log=print, tag=""):
    torch.manual_seed(seed)
    model = FixedAE(spec, geno).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr_w"])
    bs = BATCH[spec.name]
    hist = []
    for ep in range(hp["retrain_epochs"]):
        model.train()
        tl, tr_, ta, k = 0., 0., 0., 0
        for xb, yb in batches(data["Xtr"], data["ytr"], bs, True, drop_last=True):
            loss, lr_, acc = total_loss(model, xb, yb, spec, lam, beta=beta)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); tr_ += lr_.item(); ta += acc.item(); k += 1
        vr, vacc, _ = eval_split(model, data["Xva"], data["yva"], spec)
        hist.append(dict(epoch=ep + 1, phase="retrain", train_loss=tl / k, train_recon=tr_ / k, train_acc=ta / k,
                         val_recon=vr, val_acc=vacc))
        if (ep + 1) % 10 == 0 or ep == 0 or ep + 1 == hp["retrain_epochs"]:
            log(f"  [{tag} {ep + 1:2d}/{hp['retrain_epochs']}] loss {hist[-1]['train_loss']:.4f} val_recon {vr:.4f} val_acc {vacc:.3f}")
    return model, hist


# ------------------------------------------------------------------ linear probe on frozen latent z
def linear_probe(ztr, ytr, evals, spec, steps, seed):
    """Train a linear classifier on frozen z (train split), return predictions for each (z, y) in evals."""
    g = torch.Generator(device=ztr.device).manual_seed(seed)
    mu, sd = ztr.mean(0, keepdim=True), ztr.std(0, keepdim=True) + 1e-6
    f = lambda z: (z - mu) / sd
    clf = torch.nn.Linear(ztr.shape[1], spec.n_out).to(ztr.device)
    opt = torch.optim.Adam(clf.parameters(), lr=5e-3)
    n, bs = len(ztr), 1024
    for _ in range(steps):
        i = torch.randint(0, n, (min(bs, n),), device=ztr.device, generator=g)
        loss = task_loss(clf(f(ztr[i])), ytr[i], spec)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return [predict(clf(f(z)), spec).cpu().numpy() for z, _ in evals]


def score(pred, y, spec):
    y = y.cpu().numpy()
    out = dict(acc=float((pred == y).mean() * 100))
    if spec.task == "binary":
        out["f1"] = float(f1_score(y, pred, pos_label=1))          # anomalous (positive) class
    else:
        out["bal_acc"] = float(balanced_accuracy_score(y, pred) * 100)
    return out


def evaluate(model, data, spec, hp, seed):
    """All methods are scored identically: linear probe on frozen z, test-set metric from the paper."""
    r = {}
    vr, _, zva = eval_split(model, data["Xva"], data["yva"], spec)
    tr, _, zte = eval_split(model, data["Xte"], data["yte"], spec)
    _, _, ztr = eval_split(model, data["Xtr"], data["ytr"], spec)
    pv, pt = linear_probe(ztr, data["ytr"], [(zva, None), (zte, None)], spec, hp["probe_steps"], seed)
    r["val"] = score(pv, data["yva"], spec); r["val"]["recon_mse"] = vr
    r["test"] = score(pt, data["yte"], spec); r["test"]["recon_mse"] = tr
    r["params_M"] = count_params(model) / 1e6
    r["mflops"] = count_flops(model, spec) / 1e6
    r["primary"] = spec.primary
    return r
