"""Tiện ích: lưu lưới ảnh, đo độ đa dạng (phát hiện mode collapse), vẽ đường cong huấn luyện."""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils

from .dataset import to_unit_range


def save_grid(images: torch.Tensor, path: str, nrow: int = 8, padding: int = 2) -> None:
    """images: (N,3,H,W) trong [-1, 1]."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    vutils.save_image(to_unit_range(images.detach().cpu()), path, nrow=nrow, padding=padding)


@torch.no_grad()
def diversity_score(images: torch.Tensor) -> float:
    """Khoảng cách RMS trung bình giữa các cặp ảnh trong tập (theo pixel, miền [-1,1]).

    Nếu G bị mode collapse, các ảnh gần giống nhau -> điểm này tụt mạnh so với ảnh thật.
    """
    x = images.detach().float().cpu().flatten(1)
    n = x.shape[0]
    d = torch.cdist(x, x) / np.sqrt(x.shape[1])
    return (d.sum() / (n * (n - 1))).item()


def slerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """Nội suy cầu giữa hai véc-tơ tiềm ẩn (phù hợp với phân phối Gaussian)."""
    a, b = z0.flatten(), z1.flatten()
    omega = torch.acos(torch.clamp(torch.dot(a / a.norm(), b / b.norm()), -1, 1))
    so = torch.sin(omega)
    if so.abs() < 1e-6:
        return (1 - t) * z0 + t * z1
    return (torch.sin((1 - t) * omega) / so) * z0 + (torch.sin(t * omega) / so) * z1


def plot_training(out_dir: str) -> None:
    """Vẽ loss G/D theo iteration, D(x)/D(G(z)) và FID/diversity theo epoch từ file CSV."""
    it_path, ep_path = os.path.join(out_dir, "iter_losses.csv"), os.path.join(out_dir, "metrics.csv")
    if not (os.path.isfile(it_path) and os.path.isfile(ep_path)):
        return
    it = np.genfromtxt(it_path, delimiter=",", names=True)
    with open(ep_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ep = [int(r["epoch"]) for r in rows]
    col = lambda k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    k = max(1, len(it["iter"]) // 300)  # làm mượt bằng trung bình trượt
    sm = lambda v: np.convolve(v, np.ones(k) / k, mode="valid") if k > 1 else v
    ax[0, 0].plot(sm(it["loss_g"]), label="Generator")
    ax[0, 0].plot(sm(it["loss_d"]), label="Discriminator")
    ax[0, 0].set(title="Loss theo iteration (trung bình trượt)", xlabel="iteration", ylabel="BCE loss")
    ax[0, 0].legend()
    ax[0, 1].plot(ep, col("d_x"), label="D(x)  (ảnh thật)")
    ax[0, 1].plot(ep, col("d_gz"), label="D(G(z))  (ảnh giả)")
    ax[0, 1].axhline(0.5, color="gray", ls="--", lw=0.8)
    ax[0, 1].set(title="Cân bằng G–D (lý tưởng → 0.5)", xlabel="epoch", ylim=(0, 1))
    ax[0, 1].legend()
    fid = col("fid")
    ok = ~np.isnan(fid)
    ax[1, 0].plot(np.array(ep)[ok], fid[ok], "o-", color="tab:red")
    ax[1, 0].set(title="FID (thấp hơn = tốt hơn)", xlabel="epoch")
    ax[1, 1].plot(ep, col("diversity"), color="tab:green", label="ảnh sinh")
    real_div = col("diversity_real")
    if not np.isnan(real_div).all():
        ax[1, 1].axhline(np.nanmean(real_div), color="gray", ls="--", label="ảnh thật")
    ax[1, 1].set(title="Độ đa dạng (tụt mạnh → nghi mode collapse)", xlabel="epoch")
    ax[1, 1].legend()
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=130)
    plt.close(fig)
