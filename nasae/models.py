"""NAS-AE supernet (paper Sec. III-A/B) and the discrete AE derived from it (also used by every baseline)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import DIMS, DEPTHS, ACTS, OPS, DMAX

ACT_FN = {"relu": F.relu, "gelu": F.gelu, "tanh": torch.tanh}


def prod(t):
    return int(math.prod(t))


def stage_sizes(size):
    """spatial size after 0,1,2 stride-2 convs (k=3,p=1)"""
    s = [tuple(size)]
    for _ in range(2):
        s.append(tuple((n - 1) // 2 + 1 for n in s[-1]))
    return s


def _conv(ndim): return nn.Conv1d if ndim == 1 else nn.Conv2d
def _bn(ndim): return nn.BatchNorm1d if ndim == 1 else nn.BatchNorm2d


class Zero(nn.Module):
    def forward(self, x): return x * 0.0


def make_op(name, C, ndim, affine):
    Conv, BN = _conv(ndim), _bn(ndim)
    if name == "conv":    # Conv3x3
        return nn.Sequential(nn.ReLU(), Conv(C, C, 3, padding=1, bias=False), BN(C, affine=affine))
    if name == "linear":  # pointwise linear (1x1 conv)
        return nn.Sequential(nn.ReLU(), Conv(C, C, 1, bias=False), BN(C, affine=affine))
    if name == "skip":
        return nn.Identity()
    if name == "zero":
        return Zero()
    raise ValueError(name)


def op_flops(name, C, size, ndim):
    """FLOPs (2*MACs) of one candidate op, used in C(alpha), Eq. 7."""
    n = prod(size)
    k = 3 if ndim == 1 else 9
    if name == "conv": return 2. * k * C * C * n
    if name == "linear": return 2. * C * C * n
    return 0.


class Edge(nn.Module):
    """Eq. 2: y = x + sum_o softmax(alpha_e)_o * op_o(x)   (mixed)   or   y = x + op(x)   (derived).
    The residual x makes the 'Zero' op a real 'pruned edge' (signal still flows)."""
    def __init__(self, C, ndim, choice=None, affine=True):
        super().__init__()
        self.mixed = choice is None
        if self.mixed:
            self.ops = nn.ModuleList([make_op(o, C, ndim, affine) for o in OPS])
        else:
            self.choice = choice
            self.op = make_op(choice, C, ndim, affine) if choice != "zero" else None

    def forward(self, x, w=None):
        if self.mixed:
            out = sum(wi * op(x) for wi, op, name in zip(w, self.ops, OPS) if name != "zero")
            return x + out
        return x if self.op is None else x + self.op(x)


class Encoder(nn.Module):
    def __init__(self, spec, C, ops=None, affine=True):
        super().__init__()
        Conv, BN = _conv(spec.ndim), _bn(spec.ndim)
        self.stem = nn.Sequential(Conv(spec.in_ch, C, 3, padding=1, bias=False), BN(C, affine=affine), nn.ReLU())
        self.edges = nn.ModuleList([Edge(C, spec.ndim, None if ops is None else ops[i], affine) for i in range(2)])
        self.downs = nn.ModuleList([nn.Sequential(Conv(C, C, 3, stride=2, padding=1, bias=False),
                                                  BN(C, affine=affine), nn.ReLU()) for _ in range(2)])

    def forward(self, x, w=None):
        x = self.stem(x)
        for i in range(2):
            x = self.edges[i](x, None if w is None else w[i])
            x = self.downs[i](x)
        return x.flatten(1)


class Decoder(nn.Module):
    def __init__(self, spec, C, d_in, ops=None, affine=True):
        super().__init__()
        Conv, BN = _conv(spec.ndim), _bn(spec.ndim)
        self.sizes = stage_sizes(spec.size)
        self.C = C
        self.fc = nn.Linear(d_in, C * prod(self.sizes[2]))
        self.edges = nn.ModuleList([Edge(C, spec.ndim, None if ops is None else ops[i], affine) for i in range(2)])
        up = lambda s: nn.Upsample(size=s[0] if spec.ndim == 1 else s, mode="nearest")
        self.ups = nn.ModuleList([nn.Sequential(up(self.sizes[1]), Conv(C, C, 3, padding=1, bias=False), BN(C, affine=affine), nn.ReLU()),
                                  nn.Sequential(up(self.sizes[0]), Conv(C, C, 3, padding=1, bias=False), BN(C, affine=affine), nn.ReLU())])
        self.out = Conv(C, spec.in_ch, 3, padding=1)

    def forward(self, z, w=None):
        x = F.relu(self.fc(z)).view(z.shape[0], self.C, *self.sizes[2])
        for i in range(2):
            x = self.edges[i](x, None if w is None else w[i])
            x = self.ups[i](x)
        return self.out(x)


# ====================================================================== SUPERNET
class SuperAE(nn.Module):
    """Continuous relaxation over: bottleneck dim, depth, activation, per-layer skip gate, encoder/decoder edge ops."""
    def __init__(self, spec, C=32):
        super().__init__()
        self.spec, self.C = spec, C
        self.encoder = Encoder(spec, C, None, affine=False)
        self.decoder = Decoder(spec, C, DMAX, None, affine=False)
        self.H = C * prod(self.decoder.sizes[2])
        assert self.H >= DMAX, "flattened feature size must be >= 128 (needed for skip slice h[:d])"
        self.proj = nn.ModuleList([nn.Linear(self.H, d) for d in DIMS])              # Eq. 1 candidates
        self.W = nn.ModuleList([nn.Linear(DMAX, DMAX) for _ in range(max(DEPTHS) - 1)])  # layers 2..3
        self.head = nn.Linear(DMAX, spec.n_out)
        # architecture parameters  (Eq. 4)
        r = lambda *s: nn.Parameter(1e-3 * torch.randn(*s))
        self.alpha_d, self.alpha_L, self.alpha_sigma = r(len(DIMS)), r(len(DEPTHS)), r(len(ACTS))
        self.alpha_skip = r(max(DEPTHS))
        self.alpha_e_enc, self.alpha_e_dec = r(2, len(OPS)), r(2, len(OPS))
        masks = torch.zeros(len(DIMS), DMAX)
        for i, d in enumerate(DIMS): masks[i, :d] = 1
        self.register_buffer("masks", masks)
        s = self.decoder.sizes
        self.register_buffer("fl_lin", torch.tensor([2. * self.H * d for d in DIMS]) / 1e6)
        enc_sz, dec_sz = [s[0], s[1]], [s[2], s[1]]
        self.register_buffer("fl_enc", torch.tensor([[op_flops(o, C, z, spec.ndim) for o in OPS] for z in enc_sz]) / 1e6)
        self.register_buffer("fl_dec", torch.tensor([[op_flops(o, C, z, spec.ndim) for o in OPS] for z in dec_sz]) / 1e6)

    def alphas(self):
        return [p for n, p in self.named_parameters() if n.startswith("alpha_")]

    def weights(self):
        return [p for n, p in self.named_parameters() if not n.startswith("alpha_")]

    def forward(self, x):
        sm = lambda a: F.softmax(a, -1)
        wd, wL, wa = sm(self.alpha_d), sm(self.alpha_L), sm(self.alpha_sigma)
        s = torch.sigmoid(self.alpha_skip)
        we_enc, we_dec = sm(self.alpha_e_enc), sm(self.alpha_e_dec)
        act = lambda u: sum(wa[k] * ACT_FN[a](u) for k, a in enumerate(ACTS))
        mask = (wd[:, None] * self.masks).sum(0)                      # soft "keep first d units"

        h = self.encoder(x, we_enc)
        u = sum(wd[i] * F.pad(p(h), (0, DMAX - DIMS[i])) for i, p in enumerate(self.proj))   # Eq. 1
        z = act(u) * (1 - s[0]) + h[:, :DMAX] * mask * s[0]            # Eq. 3, layer 1
        zs = [z]
        for l in range(1, max(DEPTHS)):                                 # layers 2..3
            z = (act(self.W[l - 1](z)) * (1 - s[l]) + z * s[l]) * mask
            zs.append(z)
        z = sum(wL[l] * zs[l] for l in range(len(DEPTHS)))
        recon = self.decoder(z, we_dec)
        return recon, self.head(z), z, x.new_zeros(())

    def complexity(self):
        """Eq. 7 (+decoder edges). Unit: MFLOPs, so lambda_3=0.01 is on a sensible scale."""
        sm = lambda a: F.softmax(a, -1)
        c = (sm(self.alpha_d) * self.fl_lin).sum()
        c = c + (sm(self.alpha_e_enc) * self.fl_enc).sum() + (sm(self.alpha_e_dec) * self.fl_dec).sum()
        return c

    @torch.no_grad()
    def genotype(self):
        am = lambda a: int(a.argmax())
        L = DEPTHS[am(self.alpha_L)]
        sk = (torch.sigmoid(self.alpha_skip) > 0.5).int().tolist()[:L]
        return dict(dim=DIMS[am(self.alpha_d)], depth=L, act=ACTS[am(self.alpha_sigma)], skips=sk,
                    enc=[OPS[am(a)] for a in self.alpha_e_enc], dec=[OPS[am(a)] for a in self.alpha_e_dec],
                    C=self.C, vae=False)

    @torch.no_grad()
    def alpha_snapshot(self):
        sm = lambda a: F.softmax(a, -1).tolist()
        return dict(d=sm(self.alpha_d), L=sm(self.alpha_L), sigma=sm(self.alpha_sigma),
                    skip=torch.sigmoid(self.alpha_skip).tolist(),
                    enc=[sm(a) for a in self.alpha_e_enc], dec=[sm(a) for a in self.alpha_e_dec])


# ====================================================================== DERIVED / BASELINE AE
class FixedAE(nn.Module):
    """Discrete AE for a genotype dict: dim, depth, act, skips, enc[2], dec[2], C, vae."""
    def __init__(self, spec, g):
        super().__init__()
        self.spec, self.g = spec, g
        C, d, L = g["C"], g["dim"], g["depth"]
        self.encoder = Encoder(spec, C, g["enc"], affine=True)
        self.decoder = Decoder(spec, C, d, g["dec"], affine=True)
        H = C * prod(self.decoder.sizes[2])
        assert H >= d
        self.act = ACT_FN[g["act"]]
        self.layers = nn.ModuleList([nn.Linear(H if l == 0 else d, d) for l in range(L)])
        self.skips = list(g["skips"]) + [0] * (L - len(g["skips"]))
        self.vae = g.get("vae", False)
        if self.vae:
            self.mu, self.lv = nn.Linear(d, d), nn.Linear(d, d)
        self.head = nn.Linear(d, spec.n_out)

    def forward(self, x):
        z = self.encoder(x)
        for l, W in enumerate(self.layers):            # Eq. 3 with hard s_l in {0,1}
            z = z[:, :self.g["dim"]] if self.skips[l] else self.act(W(z))
        kl = x.new_zeros(())
        if self.vae:
            mu, lv = self.mu(z), self.lv(z).clamp(-10, 10)
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
            z_dec = mu + torch.randn_like(mu) * (0.5 * lv).exp() if self.training else mu
            z = mu
        else:
            z_dec = z
        return self.decoder(z_dec), self.head(z), z, kl

    def complexity(self):
        return 0.


def match_width(spec, g, target_params, lo=8, hi=160, step=4):
    """pick channel width C so that the AE has ~target_params parameters (VAE baseline: 'equivalent parameter count')."""
    best = None
    for C in range(lo, hi + 1, step):
        gg = dict(g, C=C)
        try:
            n = sum(p.numel() for p in FixedAE(spec, gg).parameters())
        except AssertionError:
            continue
        if best is None or abs(n - target_params) < abs(best[1] - target_params):
            best = (C, n)
    return best[0]


def random_genotype(rng, C):
    L = int(rng.choice(DEPTHS))
    return dict(dim=int(rng.choice(DIMS)), depth=L, act=str(rng.choice(ACTS)), skips=[int(v) for v in rng.randint(0, 2, L)],
                enc=[str(rng.choice(OPS)) for _ in range(2)], dec=[str(rng.choice(OPS)) for _ in range(2)], C=C, vae=False)
