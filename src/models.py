"""Giai đoạn 2 - Kiến trúc DCGAN (Radford et al., 2016) cho ảnh 64x64.

Generator      : z (nz,1,1) --ConvTranspose2d x5--> ảnh (3,64,64), Tanh.
Discriminator  : ảnh (3,64,64) --Conv2d x5--> xác suất thật (Sigmoid).

Các nguyên tắc DCGAN được áp dụng: không pooling (dùng strided conv), BatchNorm,
ReLU trong G / LeakyReLU(0.2) trong D, Tanh ở đầu ra G, khởi tạo N(0, 0.02).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def weights_init(m: nn.Module) -> None:
    """Khởi tạo trọng số theo N(0, 0.02); BatchNorm weight ~ N(1, 0.02), bias = 0."""
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


class Generator(nn.Module):
    """z: (B, nz, 1, 1) -> ảnh: (B, nc, 64, 64) trong [-1, 1]."""

    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.nz = nz
        self.net = nn.Sequential(
            # (nz) x 1 x 1 -> (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            # -> (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            # -> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            # -> ngf x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            # -> nc x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def sample_noise(self, n: int, device=None) -> torch.Tensor:
        return torch.randn(n, self.nz, 1, 1, device=device)


class Discriminator(nn.Module):
    """ảnh: (B, nc, 64, 64) -> xác suất thật: (B,)."""

    def __init__(self, ndf: int = 64, nc: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            # nc x 64 x 64 -> ndf x 32 x 32  (lớp đầu không dùng BatchNorm)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            # -> (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            # -> (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
            # -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def build_models(nz=100, ngf=64, ndf=64, nc=3, device="cpu"):
    g = Generator(nz, ngf, nc).to(device)
    d = Discriminator(ndf, nc).to(device)
    g.apply(weights_init)
    d.apply(weights_init)
    return g, d


if __name__ == "__main__":  # kiểm tra nhanh kích thước tensor
    g, d = build_models()
    z = g.sample_noise(4)
    x = g(z)
    print("G:", tuple(z.shape), "->", tuple(x.shape), "| D:", tuple(d(x).shape))
    print("Tham số G: %.2fM | D: %.2fM" % (sum(p.numel() for p in g.parameters()) / 1e6,
                                          sum(p.numel() for p in d.parameters()) / 1e6))
