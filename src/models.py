"""
src/models.py - Mô hình cDCGAN / DCGAN.
"""
from models import (
    weights_init,
    ConditionalGenerator,
    ConditionalDiscriminator,
    build_cDCGAN,
)

Generator = ConditionalGenerator

if __name__ == "__main__":
    import torch
    from utils import ensure_utf8_stdio

    ensure_utf8_stdio()
    g, d = build_cDCGAN(nz=100, num_attrs=6)
    z = g.sample_noise(4)
    c = torch.randint(0, 2, (4, 6), dtype=torch.float32)
    x = g(z, c)
    print("G:", tuple(z.shape), "->", tuple(x.shape), "| D:", tuple(d(x, c).shape))
    print(
        "Tham số G: %.2fM | D: %.2fM"
        % (sum(p.numel() for p in g.parameters()) / 1e6, sum(p.numel() for p in d.parameters()) / 1e6)
    )
