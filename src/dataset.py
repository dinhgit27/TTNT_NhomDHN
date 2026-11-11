"""
src/dataset.py - Dataset cDCGAN và bộ đọc .npy cho đánh giá/FID.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import (
    SELECTED_ATTRS,
    get_transforms,
    CelebAAttributeDataset,
    get_celeba_dataloader,
)
from utils import denormalize

to_unit_range = denormalize


class CelebA64(Dataset):
    """Ảnh CelebA 64x64 từ file .npy đã tiền xử lý. Trả về tensor (3, 64, 64) trong [-1, 1]."""

    def __init__(self, path: str, flip: bool = True, limit: int = 0):
        self.path = path
        self.flip = flip
        self._data: np.ndarray | None = None
        n = len(np.load(path, mmap_mode="r"))
        self.n = min(n, limit) if limit > 0 else n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._data is None:
            self._data = np.load(self.path, mmap_mode="r")
        img = torch.from_numpy(np.array(self._data[idx], copy=True))  # bản sao ghi được
        x = img.permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        if self.flip and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        return x

    def sample(self, n: int, seed: int = 0) -> torch.Tensor:
        """Lấy ngẫu nhiên n ảnh thật (không lật) -> (n, 3, 64, 64) trong [-1, 1]."""
        if self.n == 0 or n <= 0:
            return torch.empty((0, 3, 64, 64), dtype=torch.float32)
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(self.n, size=min(n, self.n), replace=False))
        data = np.load(self.path, mmap_mode="r")
        arr = torch.from_numpy(np.array(data[idx], copy=True))
        return arr.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
