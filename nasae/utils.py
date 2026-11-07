import random, numpy as np, torch


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def count_flops(model, spec):
    """Real FLOPs per sample (encoder+bottleneck+decoder+head) measured by torch's FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    dev = next(model.parameters()).device
    x = torch.randn(1, spec.in_ch, *spec.size, device=dev)
    with FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops()
