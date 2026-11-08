"""Giai đoạn 1 - Tiền xử lý CelebA (Align & Cropped).

Pipeline cho mỗi ảnh:  đọc JPEG -> center crop -> resize 64x64 -> lưu uint8.
Toàn bộ 202.599 ảnh được gom vào 1 file ``.npy`` (N, 64, 64, 3) để huấn luyện
nhanh (không phải giải mã JPEG mỗi epoch). Bước chuẩn hóa về [-1, 1] được thực
hiện ở :mod:`src.dataset` để file lưu trữ giữ nguyên dạng uint8 (gọn, ~2.5 GB).

Ví dụ::

    python -m src.preprocess --src archive.zip --dst data/processed/celeba_64.npy
    python -m src.preprocess --src path/to/img_align_celeba --limit 2000   # chạy thử
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

_ZIP: zipfile.ZipFile | None = None
_CFG: dict = {}


def list_images(src: str) -> list[str]:
    """Liệt kê tên ảnh (đã sắp xếp) trong file zip hoặc thư mục."""
    if os.path.isfile(src) and src.lower().endswith(".zip"):
        with zipfile.ZipFile(src) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    elif os.path.isdir(src):
        names = []
        for root, _, files in os.walk(src):
            names += [os.path.join(root, f) for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    else:
        raise FileNotFoundError(f"Không tìm thấy nguồn dữ liệu: {src}")
    return sorted(names)


def center_crop_resize(img: np.ndarray, crop: int, size: int) -> np.ndarray:
    """Cắt vuông ``crop x crop`` ở chính giữa rồi resize về ``size x size``."""
    h, w = img.shape[:2]
    c = min(crop, h, w)
    top, left = (h - c) // 2, (w - c) // 2
    img = img[top:top + c, left:left + c]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _init_worker(src: str, crop: int, size: int) -> None:
    global _ZIP, _CFG
    _CFG = {"crop": crop, "size": size}
    _ZIP = zipfile.ZipFile(src) if src.lower().endswith(".zip") and os.path.isfile(src) else None


def _process(name: str) -> np.ndarray | None:
    try:
        if _ZIP is not None:
            buf = np.frombuffer(_ZIP.read(name), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(name, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return center_crop_resize(img, _CFG["crop"], _CFG["size"])
    except Exception:  # ảnh lỗi -> bỏ qua, báo ở cuối
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="archive.zip", help="file zip hoặc thư mục chứa ảnh CelebA aligned")
    p.add_argument("--dst", default="data/processed/celeba_64.npy")
    p.add_argument("--crop", type=int, default=148, help="cạnh vùng center-crop (ảnh gốc 178x218)")
    p.add_argument("--size", type=int, default=64, help="kích thước ảnh đầu ra")
    p.add_argument("--limit", type=int, default=0, help="chỉ xử lý N ảnh đầu (0 = tất cả)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = p.parse_args()

    names = list_images(a.src)
    if a.limit > 0:
        names = names[: a.limit]
    print(f"Tìm thấy {len(names)} ảnh trong {a.src}")

    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    out = np.lib.format.open_memmap(a.dst, mode="w+", dtype=np.uint8, shape=(len(names), a.size, a.size, 3))

    n_ok, n_bad = 0, 0
    with Pool(a.workers, initializer=_init_worker, initargs=(a.src, a.crop, a.size)) as pool:
        for arr in tqdm(pool.imap(_process, names, chunksize=256), total=len(names), desc="preprocess"):
            if arr is None:
                n_bad += 1
                continue
            out[n_ok] = arr
            n_ok += 1
    out.flush()
    del out

    if n_bad:  # cắt bớt phần thừa nếu có ảnh lỗi
        data = np.load(a.dst, mmap_mode="r")[:n_ok]
        np.save(a.dst, np.ascontiguousarray(data))
    meta = {"count": n_ok, "bad": n_bad, "size": a.size, "crop": a.crop, "source": os.path.basename(a.src)}
    with open(os.path.splitext(a.dst)[0] + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Xong: {n_ok} ảnh -> {a.dst}  (lỗi: {n_bad})")


if __name__ == "__main__":
    main()
