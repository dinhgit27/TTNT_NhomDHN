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

import io
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image, ImageFilter


def ensure_utf8_stdio() -> None:
    """Đảm bảo stdout/stderr dùng UTF-8 trên Windows để tránh lỗi UnicodeEncodeError (cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def extract_checkpoint_info(
    checkpoint: Any,
    nz: int = 100,
) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, Any]]:
    """Tách state_dict, suy luận số thuộc tính điều kiện (num_attrs) và metadata từ checkpoint."""
    if isinstance(checkpoint, dict) and "G" in checkpoint:
        state_dict = checkpoint["G"]
        args = checkpoint.get("args") or {}
        meta = {
            "epoch": checkpoint.get("epoch"),
            "fid": checkpoint.get("fid"),
            "nz": args.get("nz", nz),
            "ngf": args.get("ngf", 64),
        }
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        meta = {"epoch": None, "fid": None, "nz": nz, "ngf": 64}
    else:
        raise ValueError("Cấu trúc file trọng số không hợp lệ.")

    first_weight = state_dict.get("net.0.weight")
    if first_weight is None:
        for k, v in state_dict.items():
            if k == "0.weight" or k.endswith(".0.weight"):
                first_weight = v
                break

    in_ch = int(first_weight.shape[0]) if first_weight is not None else int(meta["nz"])
    num_attrs = max(0, in_ch - int(meta["nz"]))
    return state_dict, num_attrs, meta


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
    conditions: Optional[torch.Tensor],
    output_path: str,
    nrow: int = 4
) -> None:
    """
    Sinh ảnh từ Generator với noise và conditions cho trước và lưu vào file ảnh.
    """
    was_training = generator.training
    generator.eval()
    fake_imgs = generator(noise, conditions)
    save_image_grid(fake_imgs, output_path, nrow=nrow)
    generator.train(was_training)


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
    was_training = generator.training
    generator.eval()

    saved_count = 0
    print(f"Đang sinh {num_samples} ảnh giả lập lưu vào thư mục '{output_dir}' để tính FID...")

    while saved_count < num_samples:
        current_batch = min(batch_size, num_samples - saved_count)
        noise = torch.randn(current_batch, nz, device=device)
        conditions = (
            torch.randint(0, 2, (current_batch, num_attrs), dtype=torch.float32, device=device)
            if num_attrs > 0
            else None
        )

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
    generator.train(was_training)


def slerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """Nội suy hình cầu (Spherical Linear Interpolation) trong không gian tiềm ẩn."""
    orig_shape = z0.shape
    if z0.dim() == 2:
        a_flat, b_flat = z0, z1
    else:
        a_flat, b_flat = z0.reshape(1, -1), z1.reshape(1, -1)
    a = a_flat / a_flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = b_flat / b_flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    omega = torch.acos(torch.clamp((a * b).sum(dim=-1, keepdim=True), -1.0, 1.0))
    so = torch.sin(omega)
    lerp = (1.0 - t) * a_flat + t * b_flat
    sph = (
        (torch.sin((1.0 - t) * omega) / so.clamp_min(1e-8)) * a_flat
        + (torch.sin(t * omega) / so.clamp_min(1e-8)) * b_flat
    )
    out = torch.where(so.abs() < 1e-5, lerp, sph)
    return out.reshape(orig_shape)


@torch.no_grad()
def diversity_score(images: torch.Tensor) -> float:
    """Khoảng cách RMS trung bình giữa các cặp ảnh trong tập (phát hiện mode collapse)."""
    x = images.detach().float().cpu().flatten(1)
    n = x.shape[0]
    if n < 2:
        return 0.0
    d = torch.cdist(x, x) / np.sqrt(x.shape[1])
    return (d.sum() / (n * (n - 1))).item()


def tensor_to_pil(tensor: torch.Tensor, upscale_size: int = 256, apply_smooth: bool = True) -> Image.Image:
    """Chuyển tensor (3, H, W) trong [-1, 1] sang PIL Image RGB với tùy chọn phóng to và làm mịn."""
    tensor_unit = denormalize(tensor.detach().cpu())
    np_img = (tensor_unit.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img)
    if upscale_size != 64:
        pil_img = pil_img.resize((upscale_size, upscale_size), Image.Resampling.LANCZOS)
        if apply_smooth and upscale_size >= 256:
            pil_img = pil_img.filter(ImageFilter.SMOOTH)
    return pil_img


@torch.no_grad()
def generate_morph_gif_bytes(
    generator: torch.nn.Module,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    conditions: Optional[torch.Tensor] = None,
    steps: int = 16,
    duration: int = 80,
    upscale_size: int = 256,
) -> bytes:
    """Tạo file GIF động biến hình từ Face A sang Face B và quay vòng mượt mà."""
    if steps < 1:
        raise ValueError("steps phải >= 1")
    alphas = np.linspace(0.0, 1.0, steps)
    loop_alphas = list(alphas) + list(reversed(alphas[1:-1]))
    frames = []
    for a in loop_alphas:
        z_cur = slerp(z_a, z_b, float(a))
        img_tensor = generator(z_cur, conditions)[0] if conditions is not None else generator(z_cur)[0]
        frames.append(tensor_to_pil(img_tensor, upscale_size=upscale_size, apply_smooth=True))

    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    return buf.getvalue()


def set_seed(seed: int = 42) -> None:
    """Thiết lập seed ngẫu nhiên cho tính tái lập (reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    ensure_utf8_stdio()
    print("--- KIỂM TRA NHANH CÁC HÀM TIỆN ÍCH TRONG UTILS.PY ---")
    dummy_imgs = torch.randn(4, 3, 64, 64)
    out_test_path = "outputs/test_grid.png"
    save_image_grid(dummy_imgs, out_test_path, nrow=2)
    print(f"Đã lưu ảnh lưới thử nghiệm vào: {out_test_path}")

    presets = get_preset_conditions(num_attrs=6)
    print(f"Tạo thành công preset conditions: shape={presets.shape}")
    print("-> utils.py hoạt động chuẩn xác!")
