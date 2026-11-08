"""
dataset.py - Tiền xử lý dữ liệu và Xây dựng Custom Dataset CelebA có Điều kiện (cDCGAN).

Tác giả: Senior AI Engineer & Full-stack Developer
Nhiệm vụ:
  1. Đọc và lọc các nhãn thuộc tính từ CelebA (Male, Smiling, Eyeglasses, Blond_Hair, Black_Hair, Brown_Hair).
  2. Áp dụng các biến đổi (transforms):
     - CenterCrop(178): Cắt vùng trọng tâm khuôn mặt (178x178).
     - Resize(64x64): Đưa về độ phân giải 64x64 chuẩn cho DCGAN.
     - ToTensor(): Chuyển đổi thành PyTorch Tensor trong khoảng [0, 1].
     - Normalize([-1, -1, -1], [1, 1, 1]): Chuẩn hóa về dải [-1, 1] khớp với hàm Tanh của Generator.
  3. Cung cấp DataLoader và cơ chế chạy thử nghiệm (mock/smoke test) trên máy cục bộ khi chưa có tập dữ liệu đầy đủ.
"""

import os
from typing import List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# 6 thuộc tính khuôn mặt quan trọng được lựa chọn làm véc-tơ điều kiện
# Thuộc tính trong CelebA gốc: 1 = Có thuộc tính, -1 = Không có thuộc tính
SELECTED_ATTRS = [
    "Male",         # Giới tính: 1 = Nam, 0 = Nữ
    "Smiling",      # Biểu cảm: 1 = Cười, 0 = Không cười
    "Eyeglasses",   # Phụ kiện: 1 = Đeo kính, 0 = Không kính
    "Blond_Hair",   # Tóc vàng: 1 = Có, 0 = Không
    "Black_Hair",   # Tóc đen: 1 = Có, 0 = Không
    "Brown_Hair"    # Tóc nâu: 1 = Có, 0 = Không
]

def get_transforms(image_size: int = 64) -> transforms.Compose:
    """
    Tạo pipeline biến đổi ảnh theo đúng chuẩn CelebA:
    1. CenterCrop 178: Cắt bỏ viền thừa, tập trung vào khuôn mặt.
    2. Resize 64x64: Kích thước chuẩn huấn luyện cDCGAN.
    3. ToTensor: Chuyển sang FloatTensor [0.0, 1.0].
    4. Normalize: Ánh xạ từ [0, 1] sang [-1, 1] (phù hợp với hàm kích hoạt Tanh).
    """
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


class CelebAAttributeDataset(Dataset):
    """
    Custom Dataset nạp ảnh CelebA kèm theo véc-tơ thuộc tính điều kiện (condition vector).
    Hỗ trợ đọc từ thư mục local hoặc fallback về dữ liệu giả lập (dummy) nếu đang test code.
    """
    def __init__(
        self,
        img_dir: str,
        attr_file: Optional[str] = None,
        selected_attrs: List[str] = SELECTED_ATTRS,
        transform: Optional[transforms.Compose] = None,
        limit: int = 0
    ):
        """
        Tham số:
            img_dir: Đường dẫn đến thư mục chứa ảnh (vd: data/celeba/img_align_celeba)
            attr_file: Đường dẫn đến file list_attr_celeba.txt
            selected_attrs: Danh sách tên thuộc tính cần trích xuất
            transform: Chuỗi biến đổi áp dụng lên ảnh
            limit: Giới hạn số lượng ảnh (dùng để test nhanh/smoke test)
        """
        self.img_dir = img_dir
        self.attr_file = attr_file
        self.selected_attrs = selected_attrs
        self.transform = transform if transform is not None else get_transforms()
        
        self.image_names: List[str] = []
        self.labels: List[torch.Tensor] = []
        
        # Kiểm tra xem thư mục dữ liệu thật có tồn tại không
        if os.path.isdir(img_dir) and attr_file and os.path.isfile(attr_file):
            self._load_dataset(limit)
        else:
            # Fallback: Nếu chưa có bộ dữ liệu thật, tạo 100 mẫu giả lập để debug trên CPU mà không bị crash
            print(f"[CẢNH BÁO] Không tìm thấy dữ liệu CelebA tại '{img_dir}' hoặc file '{attr_file}'.")
            print("Đang khởi tạo 100 mẫu dữ liệu giả lập (Synthetic/Dummy Data) để kiểm thử luồng code...")
            self.image_names = [f"dummy_{i:04d}.jpg" for i in range(100)]
            self.labels = [torch.randint(0, 2, (len(self.selected_attrs),), dtype=torch.float32) for _ in range(100)]
            self.is_dummy = True
            return

        self.is_dummy = False

    def _load_dataset(self, limit: int):
        """Đọc và trích xuất nhãn từ file list_attr_celeba.txt."""
        with open(self.attr_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # Dòng 1: Tổng số ảnh trong dataset
        # Dòng 2: Danh sách 40 tên thuộc tính
        all_attr_names = lines[1].strip().split()
        attr_indices = [all_attr_names.index(attr) for attr in self.selected_attrs if attr in all_attr_names]
        
        print(f"Đã trích xuất {len(attr_indices)} thuộc tính điều kiện: {self.selected_attrs}")
        
        # Các dòng tiếp theo: [image_name, attr_1, attr_2, ..., attr_40]
        data_lines = lines[2:]
        if limit > 0:
            data_lines = data_lines[:limit]
            
        for line in data_lines:
            parts = line.strip().split()
            img_name = parts[0]
            # Kiểm tra ảnh có tồn tại trong img_dir không
            img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                continue
                
            # Trích xuất nhãn: CelebA lưu -1 và 1 -> Ta chuyển -1 thành 0.0, 1 thành 1.0
            raw_vals = [int(parts[idx + 1]) for idx in attr_indices]
            binary_vals = [1.0 if val == 1 else 0.0 for val in raw_vals]
            
            self.image_names.append(img_name)
            self.labels.append(torch.tensor(binary_vals, dtype=torch.float32))

        print(f"-> Nạp thành công {len(self.image_names)} ảnh CelebA hợp lệ kèm nhãn điều kiện.")

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trả về:
            image: Tensor kích thước (3, 64, 64) với giá trị [-1.0, 1.0]
            label: Tensor véc-tơ điều kiện kích thước (num_attrs,) với giá trị 0.0 hoặc 1.0
        """
        if getattr(self, "is_dummy", False):
            # Tạo ảnh giả lập ngẫu nhiên (3, 64, 64) trong khoảng [-1, 1]
            dummy_img = torch.rand(3, 64, 64) * 2.0 - 1.0
            return dummy_img, self.labels[idx]

        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Đọc ảnh RGB
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img_tensor = self.transform(img)
                
        label_tensor = self.labels[idx]
        return img_tensor, label_tensor


def get_celeba_dataloader(
    img_dir: str = "data/celeba/img_align_celeba",
    attr_file: str = "data/celeba/list_attr_celeba.txt",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 2,
    limit: int = 0
) -> DataLoader:
    """Hàm tiện ích tạo DataLoader chuẩn cho quá trình huấn luyện cDCGAN."""
    dataset = CelebAAttributeDataset(
        img_dir=img_dir,
        attr_file=attr_file,
        selected_attrs=SELECTED_ATTRS,
        limit=limit
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    return dataloader


if __name__ == "__main__":
    # Test kiểm tra nhanh chức năng Dataset trên máy cá nhân
    print("--- KIỂM TRA NHANH DATASET.PY ---")
    dataloader = get_celeba_dataloader(batch_size=8, limit=16, num_workers=0)
    for imgs, attrs in dataloader:
        print(f"Kích thước Batch ảnh: {imgs.shape} (Kỳ vọng: [8, 3, 64, 64])")
        print(f"Dải giá trị ảnh: min={imgs.min().item():.2f}, max={imgs.max().item():.2f} (Kỳ vọng: [-1, 1])")
        print(f"Kích thước Batch nhãn điều kiện: {attrs.shape} (Kỳ vọng: [8, {len(SELECTED_ATTRS)}])")
        print(f"Mẫu nhãn điều kiện đầu tiên: {attrs[0].tolist()}")
        break
    print("-> Dataset kiểm thử thành công!")
