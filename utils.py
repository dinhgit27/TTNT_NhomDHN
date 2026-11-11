"""
utils.py - Các công cụ hỗ trợ trực quan hóa, lưu ảnh và đánh giá chỉ số FID (cDCGAN).

Tác giả: Senior AI Engineer & Full-stack Developer
Nhiệm vụ:
  1. denormalize: Chuyển đổi dải giá trị từ [-1, 1] về [0, 1] để lưu ảnh hoặc hiển thị.
  2. save_image_grid: Lưu lưới ảnh sinh ra để theo dõi chất lượng qua từng epoch.
  3. generate_and_save_samples: Sinh ảnh từ Generator với noise và điều kiện cố định.
  4. generate_fake_images_for_fid: Sinh N ảnh fake và lưu vào thư mục để sẵn sàng tính FID.
  5. calculate_fid_score: Gọi thư viện pytorch-fid để đo lường khoảng cách Fréchet Inception Distance
     giữa thư mục ảnh thật (real_images) và ảnh sinh ra (fake_images).
"""

import os
from typing import List, Optional
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Chuyển tensor từ khoảng [-1.0, 1.0] (đầu ra của hàm Tanh) sang [0.0, 1.0] (chuẩn hiển thị ảnh).
    Công thức: x_unit = (x + 1) / 2
    """
    return (tensor.clamp(-1.0, 1.0) + 1.0) / 2.0


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8, padding: int = 2) -> None:
    """
    Lưu tensor tập hợp ảnh (B, 3, H, W) thành một tấm ảnh lưới (grid) duy nhất.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    images_unit = denormalize(images.detach().cpu())
    vutils.save_image(images_unit, path, nrow=nrow, padding=padding)


def get_preset_conditions(num_attrs: int = 6, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Tạo bộ các vector điều kiện mẫu tiêu biểu (16 mẫu) để theo dõi chất lượng qua các epoch:
    Các thuộc tính: [Male, Smiling, Eyeglasses, Blond_Hair, Black_Hair, Brown_Hair]
    """
    presets = [
        # Nam, Cười, Kính, Tóc Đen
        [1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        # Nam, Cười, Không Kính, Tóc Đen
        [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        # Nam, Không Cười, Không Kính, Tóc Nâu
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        # Nam, Không Cười, Có Kính, Tóc Vàng
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        
        # Nữ, Cười, Không Kính, Tóc Vàng
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        # Nữ, Cười, Không Kính, Tóc Đen
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        # Nữ, Không Cười, Không Kính, Tóc Nâu
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        # Nữ, Cười, Có Kính, Tóc Đen
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],

        # Các tổ hợp đa dạng khác
        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ]
    
    # Điều chỉnh kích thước nếu số lượng thuộc tính khác 6
    adjusted = []
    for p in presets:
        if len(p) >= num_attrs:
            adjusted.append(p[:num_attrs])
        else:
            adjusted.append(p + [0.0] * (num_attrs - len(p)))
            
    return torch.tensor(adjusted, dtype=torch.float32, device=device)


@torch.no_grad()
def generate_and_save_samples(
    generator: torch.nn.Module,
    noise: torch.Tensor,
    conditions: torch.Tensor,
    output_path: str,
    nrow: int = 4
) -> None:
    """
    Sinh ảnh từ Generator với noise và conditions cho trước và lưu vào file ảnh.
    """
    generator.eval()
    fake_imgs = generator(noise, conditions)
    save_image_grid(fake_imgs, output_path, nrow=nrow)
    generator.train()


@torch.no_grad()
def generate_fake_images_for_fid(
    generator: torch.nn.Module,
    output_dir: str,
    num_samples: int = 1000,
    nz: int = 100,
    num_attrs: int = 6,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu")
) -> None:
    """
    Sinh hàng loạt ảnh từ Generator và lưu vào thư mục dưới dạng các file PNG riêng lẻ
    để phục vụ cho việc tính toán chỉ số FID bằng thư viện pytorch-fid.
    """
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    
    saved_count = 0
    print(f"Đang sinh {num_samples} ảnh giả lập lưu vào thư mục '{output_dir}' để tính FID...")

    while saved_count < num_samples:
        current_batch = min(batch_size, num_samples - saved_count)
        noise = torch.randn(current_batch, nz, device=device)
        conditions = torch.randint(0, 2, (current_batch, num_attrs), dtype=torch.float32, device=device)

        fake_imgs = generator(noise, conditions)
        fake_imgs = denormalize(fake_imgs).cpu()

        for i in range(current_batch):
            img_tensor = fake_imgs[i]
            # Chuyển tensor (3, H, W) [0, 1] sang PIL Image (H, W, 3) [0, 255]
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_pil.save(os.path.join(output_dir, f"fake_{saved_count + i:05d}.png"))

        saved_count += current_batch

    print(f"-> Đã lưu hoàn tất {saved_count} ảnh vào '{output_dir}'.")
    generator.train()


def calculate_fid_score(
    real_images_dir: str,
    fake_images_dir: str,
    batch_size: int = 50,
    device: Optional[str] = None
) -> float:
    """
    Tính toán chỉ số Fréchet Inception Distance (FID) giữa hai thư mục ảnh thật và ảnh sinh ra.
    Sử dụng trực tiếp thư viện chuẩn 'pytorch-fid'.
    
    Giá trị FID càng thấp chứng tỏ ảnh sinh ra càng giống với phân phối ảnh thật cả về chất lượng
    lẫn độ đa dạng.
    """
    try:
        from pytorch_fid import fid_score
    except ImportError:
        print("[LỖI] Thư viện 'pytorch-fid' chưa được cài đặt.")
        print("Vui lòng cài đặt bằng lệnh: pip install pytorch-fid")
        return -1.0

    if not os.path.isdir(real_images_dir):
        print(f"[CẢNH BÁO] Không tìm thấy thư mục ảnh thật: {real_images_dir}")
        return -1.0

    if not os.path.isdir(fake_images_dir):
        print(f"[CẢNH BÁO] Không tìm thấy thư mục ảnh sinh ra: {fake_images_dir}")
        return -1.0

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- BẮT ĐẦU TÍNH CHỈ SỐ FID TRÊN THIẾT BỊ: {device} ---")
    print(f"Ảnh thật: {real_images_dir}")
    print(f"Ảnh sinh: {fake_images_dir}")

    try:
        fid_value = fid_score.calculate_fid_given_paths(
            paths=[real_images_dir, fake_images_dir],
            batch_size=batch_size,
            device=torch.device(device),
            dims=2048
        )
        print(f"-> KẾT QUẢ FID: {fid_value:.4f}")
        return float(fid_value)
    except Exception as e:
        print(f"[LỖI khi tính FID]: {e}")
        return -1.0


if __name__ == "__main__":
    print("--- KIỂM TRA NHANH CÁC HÀM TIỆN ÍCH TRONG UTILS.PY ---")
    dummy_imgs = torch.randn(4, 3, 64, 64)
    out_test_path = "outputs/test_grid.png"
    save_image_grid(dummy_imgs, out_test_path, nrow=2)
    print(f"Đã lưu ảnh lưới thử nghiệm vào: {out_test_path}")

    presets = get_preset_conditions(num_attrs=6)
    print(f"Tạo thành công preset conditions: shape={presets.shape}")
    print("-> utils.py hoạt động chuẩn xác!")
