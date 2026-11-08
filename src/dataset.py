"""Dataset PyTorch đọc file ``.npy`` đã tiền xử lý và chuẩn hóa về [-1, 1]."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] (dùng khi hiển thị / lưu ảnh)."""
    return (x.clamp(-1, 1) + 1) / 2


class CelebA64(Dataset):
    """Ảnh CelebA 64x64. Trả về tensor float32 (3, 64, 64) trong [-1, 1].

    Công thức chuẩn hóa: ``x_norm = x / 127.5 - 1``  (tương đương Normalize(0.5, 0.5)),
    khớp với hàm kích hoạt Tanh ở đầu ra của Generator.
    """

    def __init__(self, path: str, flip: bool = True, limit: int = 0):
        self.path = path
        self.flip = flip
        self._data: np.ndarray | None = None  # mở lười (lazy) để an toàn khi dùng nhiều worker
        n = len(np.load(path, mmap_mode="r"))
        self.n = min(n, limit) if limit > 0 else n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._data is None:
            self._data = np.load(self.path, mmap_mode="r")
        img = torch.from_numpy(np.ascontiguousarray(self._data[idx]))  # HWC uint8
        x = img.permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        if self.flip and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        return x

    def sample(self, n: int, seed: int = 0) -> torch.Tensor:
        """Lấy ngẫu nhiên n ảnh thật (không lật) -> (n, 3, 64, 64) trong [-1, 1]."""
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(self.n, size=min(n, self.n), replace=False))
        data = np.load(self.path, mmap_mode="r")
        arr = torch.from_numpy(np.ascontiguousarray(data[idx]))
        return arr.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
