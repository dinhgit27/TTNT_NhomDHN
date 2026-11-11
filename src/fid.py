"""Module độc lập tính chỉ số FID (Fréchet Inception Distance).

    FID = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^{1/2})

``mu, S`` là trung bình và hiệp phương sai của đặc trưng pool3 (2048 chiều) của
Inception-v3 trên tập ảnh thật (r) và ảnh sinh (g). FID càng thấp càng tốt.

Lưu ý: dùng trọng số Inception-v3 ImageNet của torchvision nên số FID có thể lệch
nhẹ so với bản TensorFlow gốc (pytorch-fid / paper); vẫn hợp lệ để so sánh giữa các
cấu hình/epoch trong cùng đồ án.

CLI::

    python -m src.fid --real data/processed/celeba_64.npy --fake outputs/run1/generated   # thư mục ảnh hoặc .npy
"""
from __future__ import annotations

import argparse
import os
import warnings
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

from .utils import ensure_utf8_stdio

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class InceptionFeatures(torch.nn.Module):
    """Inception-v3 (ImageNet) trả về vector đặc trưng 2048 chiều (sau avg-pool)."""

    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights

        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = torch.nn.Identity()
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net.to(device)
        self.device = device
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1).to(device))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) trong [-1, 1]."""
        x = ((x.to(self.device).clamp(-1, 1) + 1) / 2)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.net(x)


@torch.no_grad()
def get_activations(batches: Iterable[torch.Tensor], model: InceptionFeatures) -> np.ndarray:
    feats = [model(b).cpu().numpy().astype(np.float64) for b in batches]
    return np.concatenate(feats, axis=0)


def stats_from_activations(act: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return act.mean(axis=0), np.cov(act, rowvar=False)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():  # ổn định số học
        offset = np.eye(sigma1.shape[0]) * eps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", linalg.LinAlgWarning)
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def _chunks(t: torch.Tensor, bs: int):
    for i in range(0, len(t), bs):
        yield t[i:i + bs]


def stats_from_tensor(images: torch.Tensor, model: InceptionFeatures, batch_size: int = 100):
    return stats_from_activations(get_activations(_chunks(images, batch_size), model))


def real_stats(npy_path: str, model: InceptionFeatures, n: int = 10000, cache: str | None = None,
               batch_size: int = 100, seed: int = 0):
    """Thống kê của ảnh thật (có cache ra .npz để không tính lại)."""
    from .dataset import CelebA64

    ds = CelebA64(npy_path, flip=False)
    actual_n = min(n, len(ds))
    if actual_n < 2:
        raise ValueError(f"Cần ít nhất 2 ảnh thật để tính FID, hiện có {actual_n}.")

    if cache and os.path.isfile(cache):
        d = np.load(cache)
        cached_total = int(d["total"]) if "total" in d else -1
        if int(d["n"]) == actual_n and cached_total == len(ds):
            return d["mu"], d["sigma"]

    images = ds.sample(actual_n, seed=seed)
    mu, sigma = stats_from_tensor(images, model, batch_size)
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        np.savez(cache, mu=mu, sigma=sigma, n=actual_n, total=len(ds))
    return mu, sigma


@torch.no_grad()
def generator_images(G: torch.nn.Module, n: int, device, batch_size: int = 250):
    """Sinh n ảnh từ G, trả về từng batch (CPU) để tiết kiệm bộ nhớ."""
    was_training = G.training
    G.eval()
    num_attrs = int(getattr(G, "num_attrs", 0))
    done = 0
    while done < n:
        b = min(batch_size, n - done)
        z = G.sample_noise(b, device) if hasattr(G, "sample_noise") else torch.randn(b, G.nz, 1, 1, device=device)
        if num_attrs > 0:
            cond = torch.randint(0, 2, (b, num_attrs), dtype=torch.float32, device=device)
            yield G(z, cond).cpu()
        else:
            yield G(z).cpu()
        done += b
    G.train(was_training)


def fid_of_generator(G, mu_r, sigma_r, model: InceptionFeatures, n: int = 5000, device="cpu") -> float:
    act = get_activations(generator_images(G, n, device), model)
    mu_g, sigma_g = stats_from_activations(act)
    return frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


def _load_folder_or_npy(path: str, limit: int) -> torch.Tensor:
    if path.lower().endswith(".npy"):
        arr = np.load(path, mmap_mode="r")[:limit]
        arr = torch.from_numpy(np.array(arr, copy=True))
    else:
        import cv2

        if not os.path.isdir(path):
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {path}")
        files = sorted(f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg")))[:limit]
        imgs = []
        for f in files:
            buf = np.fromfile(os.path.join(path, f), dtype=np.uint8)
            raw = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if raw is None:
                continue
            imgs.append(cv2.cvtColor(cv2.resize(raw, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        if not imgs:
            raise ValueError(f"Không tìm thấy ảnh hợp lệ nào trong '{path}'.")
        arr = torch.from_numpy(np.stack(imgs))
    return arr.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


def main() -> None:
    ensure_utf8_stdio()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", required=True, help=".npy ảnh thật đã tiền xử lý")
    p.add_argument("--fake", required=True, help="thư mục ảnh hoặc .npy (N,H,W,3 uint8) ảnh sinh")
    p.add_argument("--n-real", type=int, default=10000)
    p.add_argument("--n-fake", type=int, default=10000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    model = InceptionFeatures(a.device)
    mu_r, s_r = real_stats(a.real, model, a.n_real, cache=os.path.splitext(a.real)[0] + f"_fidstats_{a.n_real}.npz")
    mu_g, s_g = stats_from_tensor(_load_folder_or_npy(a.fake, a.n_fake), model)
    print(f"FID = {frechet_distance(mu_r, s_r, mu_g, s_g):.3f}")


if __name__ == "__main__":
    main()
