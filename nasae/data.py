"""Dataset loading. Real data by default; --synthetic gives tiny random data ONLY for smoke-testing code."""
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from .config import DataSpec

KDD_COLS = ("duration protocol_type service flag src_bytes dst_bytes land wrong_fragment urgent hot "
            "num_failed_logins logged_in num_compromised root_shell su_attempted num_root "
            "num_file_creations num_shells num_access_files num_outbound_cmds is_host_login "
            "is_guest_login count srv_count serror_rate srv_serror_rate rerror_rate srv_rerror_rate "
            "same_srv_rate diff_srv_rate srv_diff_host_rate dst_host_count dst_host_srv_count "
            "dst_host_same_srv_rate dst_host_diff_srv_rate dst_host_same_src_port_rate "
            "dst_host_srv_diff_host_rate dst_host_serror_rate dst_host_srv_serror_rate "
            "dst_host_rerror_rate dst_host_srv_rerror_rate").split()
KDD_CAT = ["protocol_type", "service", "flag"]
ATTACK_GROUP = {"normal": 0}
for g, names in {1: "back land neptune pod smurf teardrop", 2: "ipsweep nmap portsweep satan",
                 3: "ftp_write guess_passwd imap multihop phf spy warezclient warezmaster",
                 4: "buffer_overflow loadmodule perl rootkit"}.items():
    for n in names.split():
        ATTACK_GROUP[n] = g   # 0 normal, 1 dos, 2 probe, 3 r2l, 4 u2r


def _strat_split(y, seed, f_val, f_test):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=f_test, stratify=y, random_state=seed)
    tr, va = train_test_split(tr, test_size=f_val / (1 - f_test), stratify=y[tr], random_state=seed)
    return tr, va, te


def _tensorize(parts, task):
    out = {}
    for k, (X, y) in parts.items():
        out["X" + k] = torch.tensor(np.ascontiguousarray(X), dtype=torch.float32)
        out["y" + k] = torch.tensor(y, dtype=torch.long)
    return out


# ------------------------------------------------------------------ CIFAR-10
def _load_cifar(root, seed):
    from torchvision.datasets import CIFAR10
    tr = CIFAR10(root, train=True, download=True)
    te = CIFAR10(root, train=False, download=True)
    Xtr = tr.data.astype(np.float32) / 255.; ytr = np.array(tr.targets)
    Xte = te.data.astype(np.float32) / 255.; yte = np.array(te.targets)
    mean, std = Xtr.mean((0, 1, 2)), Xtr.std((0, 1, 2))
    f = lambda X: ((X - mean) / std).transpose(0, 3, 1, 2)
    i_tr, i_va = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr, random_state=seed)
    spec = DataSpec("cifar10", 2, 3, (32, 32), 10, "multiclass")
    return spec, {"tr": (f(Xtr[i_tr]), ytr[i_tr]), "va": (f(Xtr[i_va]), ytr[i_va]), "te": (f(Xte), yte)}


# ------------------------------------------------------------------ ECG5000
def _load_ecg(root, seed):
    d = os.path.join(root, "ECG5000")
    ftr, fte = os.path.join(d, "ECG5000_TRAIN.txt"), os.path.join(d, "ECG5000_TEST.txt")
    if os.path.exists(ftr) and os.path.exists(fte):
        A = np.vstack([np.loadtxt(ftr), np.loadtxt(fte)])
        y_raw, X = A[:, 0].astype(int), A[:, 1:]
    else:
        from aeon.datasets import load_classification   # downloads from timeseriesclassification.com
        Xa, ya = load_classification("ECG5000")
        X = np.asarray(Xa)[:, 0, :]; y_raw = ya.astype(int)
    y = (y_raw != 1).astype(int)          # class 1 = normal; classes 2-5 = abnormal beats -> anomaly=1
    tr, va, te = _strat_split(y, seed, 0.1, 0.2)
    mu, sd = X[tr].mean(), X[tr].std()
    f = lambda i: ((X[i] - mu) / sd)[:, None, :]
    spec = DataSpec("ecg5000", 1, 1, (X.shape[1],), 1, "binary")
    return spec, {"tr": (f(tr), y[tr]), "va": (f(va), y[va]), "te": (f(te), y[te])}


# ------------------------------------------------------------------ KDD Cup 99 (10% = 494,021 rows)
def _load_kdd(root, seed):
    import pandas as pd
    local = os.path.join(root, "kddcup.data_10_percent.gz")
    if os.path.exists(local):
        df = pd.read_csv(local, header=None, names=KDD_COLS + ["label"])
    else:
        from sklearn.datasets import fetch_kddcup99
        b = fetch_kddcup99(percent10=True, as_frame=True, data_home=root)
        df = b.frame if hasattr(b, "frame") and b.frame is not None else b.data.assign(labels=b.target)
        df = df.rename(columns={df.columns[-1]: "label"})
    dec = lambda v: v.decode() if isinstance(v, bytes) else v
    for c in KDD_CAT + ["label"]:
        df[c] = df[c].map(dec).astype(str)
    y = df["label"].str.rstrip(".").map(ATTACK_GROUP).values.astype(int)
    X = df.drop(columns=["label"])
    num = [c for c in X.columns if c not in KDD_CAT]
    X[num] = X[num].astype(float)
    for c in num:   # heavy-tailed counters -> log
        if X[c].min() >= 0 and X[c].max() > 100:
            X[c] = np.log1p(X[c])
    X = pd.get_dummies(X, columns=KDD_CAT, dtype=float)
    X = X.values.astype(np.float32)
    tr, va, te = _strat_split(y, seed, 0.1, 0.2)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    f = lambda i: ((X[i] - mu) / sd)[:, None, :]
    spec = DataSpec("kdd99", 1, 1, (X.shape[1],), 5, "multiclass")
    return spec, {"tr": (f(tr), y[tr]), "va": (f(va), y[va]), "te": (f(te), y[te])}


# ------------------------------------------------------------------ synthetic (smoke test only)
def _synthetic(name, seed):
    rng = np.random.RandomState(seed)
    cfg = {"cifar10": (DataSpec("cifar10", 2, 3, (32, 32), 10, "multiclass"), 1200),
           "ecg5000": (DataSpec("ecg5000", 1, 1, (140,), 1, "binary"), 1000),
           "kdd99": (DataSpec("kdd99", 1, 1, (118,), 5, "multiclass"), 1500)}
    spec, n = cfg[name]
    K = max(2, spec.n_out if spec.task == "multiclass" else 2)
    y = rng.randint(0, K, n)
    protos = rng.randn(K, spec.in_ch, *spec.size)
    X = (protos[y] + rng.randn(n, spec.in_ch, *spec.size)).astype(np.float32)
    a, b = int(.7 * n), int(.8 * n)
    return spec, {"tr": (X[:a], y[:a]), "va": (X[a:b], y[a:b]), "te": (X[b:], y[b:])}


def load_dataset(name, seed=0, root="data", synthetic=False, max_train=None, device="cpu"):
    if synthetic:
        print("!!! SYNTHETIC RANDOM DATA - for code smoke-testing only, results are meaningless !!!")
        spec, parts = _synthetic(name, seed)
    else:
        spec, parts = {"cifar10": _load_cifar, "ecg5000": _load_ecg, "kdd99": _load_kdd}[name](root, seed)
    if max_train and len(parts["tr"][1]) > max_train:
        rs = np.random.RandomState(seed)
        i = rs.choice(len(parts["tr"][1]), max_train, replace=False)
        parts["tr"] = (parts["tr"][0][i], parts["tr"][1][i])
        for k in ("va", "te"):
            m = max(max_train // 4, 500)
            if len(parts[k][1]) > m:
                j = rs.choice(len(parts[k][1]), m, replace=False)
                parts[k] = (parts[k][0][j], parts[k][1][j])
    data = _tensorize(parts, spec.task)
    data = {k: v.to(device) for k, v in data.items()}
    return spec, data
