"""All constants taken from the paper live here."""
from dataclasses import dataclass

# ---- Search space (paper Sec. III-A) ----
DIMS = [8, 16, 32, 64, 128]
DEPTHS = [1, 2, 3]
ACTS = ["relu", "gelu", "tanh"]
OPS = ["conv", "linear", "skip", "zero"]   # Conv3x3, Linear(1x1), Skip(identity), Zero(pruned)
DMAX = max(DIMS)

# ---- Hyper-parameters (paper Sec. III-C/D, IV-B) ----
LAMBDA = dict(l1=1.0, l2=0.5, l3=0.01)
HP = dict(
    search_epochs=25,      # supernet training
    retrain_epochs=50,     # derived arch + all baselines
    lr_w=3e-4,
    lr_alpha=1e-3,
    base_channels=32,      # channel width C of conv edges (NOT specified in paper)
    probe_steps=3000,      # linear-probe training steps (NOT specified in paper)
    vae_beta=1e-3,         # KL weight for VAE baseline (NOT specified in paper)
    random_samples=5,      # Random-NAS architectures sampled per seed
)
BATCH = {"cifar10": 128, "ecg5000": 64, "kdd99": 512}


@dataclass
class DataSpec:
    name: str
    ndim: int          # 1 -> Conv1d (ECG, KDD feature vector), 2 -> Conv2d (images)
    in_ch: int
    size: tuple        # spatial size, e.g. (32, 32) or (140,)
    n_out: int         # classes (multiclass) or 1 (binary)
    task: str          # "multiclass" | "binary"

    @property
    def primary(self):  # metric name used in the paper for this dataset
        return "f1" if self.task == "binary" else "acc"


def manual_spec(dim=32, C=None):
    """Paper baseline (1): bottleneck=32, 1 layer, ReLU, no skip."""
    return dict(dim=dim, depth=1, act="relu", skips=[0], enc=["conv", "conv"],
                dec=["conv", "conv"], C=C or HP["base_channels"], vae=False)
