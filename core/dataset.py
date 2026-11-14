import os
import io
import json
import csv
import random
import zipfile
import threading
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image

SELECTED_ATTRIBUTES = [
    ("Male", "Nam giới (Male)"),
    ("Smiling", "Đang cười (Smiling)"),
    ("Young", "Trẻ tuổi (Young)"),
    ("Eyeglasses", "Đeo kính (Eyeglasses)"),
    ("Black_Hair", "Tóc đen (Black Hair)"),
    ("Blond_Hair", "Tóc vàng (Blond Hair)"),
    ("Brown_Hair", "Tóc nâu (Brown Hair)"),
    ("Wavy_Hair", "Tóc gợn sóng (Wavy Hair)"),
    ("Heavy_Makeup", "Trang điểm (Heavy Makeup)"),
    ("Mustache", "Ria mép / Râu (Mustache)"),
]

ATTR_KEYS = [k for k, _ in SELECTED_ATTRIBUTES]
ATTR_LABELS = {k: v for k, v in SELECTED_ATTRIBUTES}


class CelebAZipDataset:
    """
    Quản lý truy xuất bộ dữ liệu CelebA từ archive.zip (202,599 ảnh).
    Trích xuất một lần duy nhất bộ mẫu chuẩn (300 ảnh + metadata) vào data_cache/
    giúp Web App khởi động và truy xuất trong < 0.05 giây mà không phải đọc lại file zip 1.4GB.
    """

    def __init__(self, zip_path: str = "archive.zip", cache_dir: str = "data_cache", max_indexed: int = 300):
        self.zip_path = os.path.abspath(zip_path)
        self.cache_dir = os.path.abspath(cache_dir)
        self.samples_dir = os.path.join(self.cache_dir, "samples")
        self.meta_cache_path = os.path.join(self.cache_dir, "celeba_meta_cache.json")
        self.max_indexed = max_indexed
        self._lock = threading.Lock()

        os.makedirs(self.samples_dir, exist_ok=True)

        self.total_images = 202599
        self.all_attr_names: List[str] = []
        self.attr_counts: Dict[str, int] = {}
        self.partition_counts: Dict[str, int] = {"Train (0)": 162770, "Val (1)": 19867, "Test (2)": 19962}
        self.records: Dict[str, Dict] = {}
        self.image_ids: List[str] = []

        self._initialize_metadata_and_samples()

    def _initialize_metadata_and_samples(self):
        if os.path.exists(self.meta_cache_path):
            try:
                with open(self.meta_cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.total_images = data.get("total_images", 202599)
                self.all_attr_names = data.get("all_attr_names", [])
                self.attr_counts = data.get("attr_counts", {})
                self.partition_counts = data.get("partition_counts", self.partition_counts)
                self.records = data.get("records", {})
                # Chỉ giữ các ảnh đã được trích xuất sẵn ra ổ cứng để truy xuất tức thì
                valid_ids = [
                    img_id for img_id in sorted(self.records.keys())
                    if os.path.exists(os.path.join(self.samples_dir, img_id))
                ]
                if len(valid_ids) >= min(100, self.max_indexed):
                    self.records = {k: self.records[k] for k in valid_ids}
                    self.image_ids = valid_ids
                    return
            except Exception as e:
                print(f"[Dataset] Cache reload notice: {e}")

        if not os.path.exists(self.zip_path):
            print(f"[Dataset] Warning: {self.zip_path} not found!")
            return

        print(f"[Dataset] One-time setup: indexing metadata & caching {self.max_indexed} CelebA samples from archive.zip...")
        records = {}
        attr_counts = {}
        all_attr_names = []

        with zipfile.ZipFile(self.zip_path, "r") as z:
            with z.open("list_attr_celeba.csv") as f:
                reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                header = next(reader)
                all_attr_names = header[1:]
                attr_indices = {name: i + 1 for i, name in enumerate(all_attr_names)}
                for name in all_attr_names:
                    attr_counts[name] = 0

                count = 0
                for row in reader:
                    if not row:
                        continue
                    img_id = row[0]
                    count += 1
                    if count <= 5000:
                        for i, name in enumerate(all_attr_names):
                            if row[i + 1] == "1":
                                attr_counts[name] += 1

                    if count <= self.max_indexed:
                        attrs = {}
                        for k in ATTR_KEYS:
                            idx = attr_indices.get(k)
                            attrs[k] = 1 if (idx is not None and row[idx] == "1") else 0
                        records[img_id] = {"attrs": attrs, "landmarks": None}
                    elif count >= 5000:
                        break

            scale = 202599.0 / max(1, min(count, 5000))
            for name in all_attr_names:
                attr_counts[name] = int(round(attr_counts[name] * scale))

            with z.open("list_landmarks_align_celeba.csv") as f:
                reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                _ = next(reader)
                for i, row in enumerate(reader):
                    if i >= self.max_indexed:
                        break
                    img_id = row[0]
                    if img_id in records:
                        records[img_id]["landmarks"] = [float(x) for x in row[1:11]]

            # Trích xuất ngay các ảnh mẫu trong cùng 1 lần mở ZipFile
            for img_id in records.keys():
                dst_path = os.path.join(self.samples_dir, img_id)
                if not os.path.exists(dst_path):
                    zip_member = f"img_align_celeba/img_align_celeba/{img_id}"
                    try:
                        with open(dst_path, "wb") as out_f:
                            out_f.write(z.read(zip_member))
                    except KeyError:
                        pass

        self.all_attr_names = all_attr_names
        self.attr_counts = attr_counts
        self.records = records
        self.image_ids = sorted(list(records.keys()))

        cache_payload = {
            "total_images": self.total_images,
            "all_attr_names": self.all_attr_names,
            "attr_counts": self.attr_counts,
            "partition_counts": self.partition_counts,
            "records": self.records,
        }
        with open(self.meta_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_payload, f, ensure_ascii=False)
        print(f"[Dataset] Ready! Cached {len(self.image_ids)} CelebA images & metadata.")

    def ensure_cached_samples(self, count: int = 200) -> List[str]:
        return self.image_ids[: min(count, len(self.image_ids))]

    def get_image_pil(self, img_id: str, size: Optional[Tuple[int, int]] = None, center_crop: bool = True) -> Image.Image:
        local_path = os.path.join(self.samples_dir, img_id)
        if not os.path.exists(local_path):
            # Fallback về ảnh đầu tiên đã cache để tránh mở lại file zip 1.4GB
            fallback_id = self.image_ids[0] if self.image_ids else "000001.jpg"
            local_path = os.path.join(self.samples_dir, fallback_id)

        img = Image.open(local_path).convert("RGB")

        if center_crop:
            w, h = img.size
            crop_size = min(w, h)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2 + 8
            top = min(max(0, top), h - crop_size)
            img = img.crop((left, top, left + crop_size, top + crop_size))

        if size is not None:
            img = img.resize(size, Image.Resampling.LANCZOS)
        return img

    def get_landmarks_cropped(self, img_id: str, target_size: int = 128) -> np.ndarray:
        rec = self.records.get(img_id)
        if not rec or not rec.get("landmarks"):
            s = target_size / 128.0
            return np.array([
                [44.0 * s, 52.0 * s],
                [84.0 * s, 52.0 * s],
                [64.0 * s, 74.0 * s],
                [48.0 * s, 94.0 * s],
                [80.0 * s, 94.0 * s],
            ], dtype=np.float32)

        lm = np.array(rec["landmarks"], dtype=np.float32).reshape(5, 2)
        top_offset = 28.0
        crop_size = 178.0
        lm[:, 1] -= top_offset
        lm = lm * (target_size / crop_size)
        return np.clip(lm, 4.0, target_size - 4.0)

    def get_attribute_vector(self, img_id: str) -> np.ndarray:
        rec = self.records.get(img_id, {})
        attrs = rec.get("attrs", {})
        return np.array([float(attrs.get(k, 0)) for k in ATTR_KEYS], dtype=np.float32)

    def query_by_attributes(
        self,
        desired_attrs: Dict[str, Optional[int]],
        limit: int = 24,
        randomize: bool = True,
        seed: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        scored = []
        active_filters = {k: v for k, v in desired_attrs.items() if v in (0, 1) and k in ATTR_KEYS}

        for img_id in self.image_ids:
            attrs = self.records[img_id]["attrs"]
            if not active_filters:
                score = 1.0
            else:
                matches = 0.0
                total_weight = 0.0
                for k, val in active_filters.items():
                    w = 3.0 if k in ("Male", "Eyeglasses", "Smiling", "Mustache") else 1.5
                    total_weight += w
                    if attrs.get(k, 0) == val:
                        matches += w
                score = matches / max(1e-6, total_weight)
            scored.append((img_id, score))

        if randomize:
            rng = random.Random(seed)
            scored.sort(key=lambda x: x[1] + rng.uniform(0, 0.04), reverse=True)
        else:
            scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:limit]

    def get_dataset_summary(self) -> Dict:
        top_attrs = sorted(self.attr_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        selected_stats = [
            {
                "key": k,
                "label": ATTR_LABELS[k],
                "positive_count": self.attr_counts.get(k, 0),
                "percentage": round(self.attr_counts.get(k, 0) * 100.0 / max(1, self.total_images), 2),
            }
            for k in ATTR_KEYS
        ]
        return {
            "dataset_name": "CelebA (Large-scale CelebFaces Attributes Dataset)",
            "source_file": os.path.basename(self.zip_path),
            "file_size_mb": round(os.path.getsize(self.zip_path) / (1024 * 1024), 2) if os.path.exists(self.zip_path) else 0,
            "total_images": self.total_images,
            "indexed_fast_access": len(self.image_ids),
            "original_resolution": "178 × 218 px (RGB)",
            "model_resolutions": ["64 × 64 px (DCGAN)", "128 × 128 px (Conditional DCGAN / Hybrid Decoder)"],
            "num_attributes": len(self.all_attr_names),
            "num_landmarks": 5,
            "partitions": self.partition_counts,
            "selected_attributes": selected_stats,
            "top_attributes": [{"name": k, "count": v, "pct": round(v * 100.0 / self.total_images, 1)} for k, v in top_attrs],
        }
