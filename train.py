"""
train.py - Quy trình huấn luyện mạng cDCGAN trên tập dữ liệu CelebA có Điều kiện.

Tác giả: Senior AI Engineer & Full-stack Developer
Nhiệm vụ:
  1. Tự động phát hiện phần cứng tối ưu: CUDA (NVIDIA GPU), MPS (Apple Silicon), hoặc CPU.
  2. Khởi tạo Criterion (BCELoss) và 2 bộ tối ưu Adam(lr=0.0002, betas=(0.5, 0.999)).
  3. Vòng lặp huấn luyện xen kẽ kinh điển của GAN:
     - Cập nhật Discriminator: Phân biệt ảnh thật + nhãn thật vs ảnh giả + nhãn điều kiện.
     - Cập nhật Generator: Đánh lừa Discriminator tin rằng ảnh sinh ra là thật ứng với nhãn điều kiện.
  4. Ghi nhận log chi tiết (Loss D, Loss G, D(x), D(G(z))) trong quá trình huấn luyện.
  5. Sau mỗi epoch:
     - Lưu ảnh mẫu sinh ra từ fixed_noise + fixed_conditions để theo dõi sự tiến hóa của khuôn mặt.
     - Lưu trọng số mạng sinh ra file 'generator.pth' sẵn sàng cho Web Streamlit.
"""

import argparse
import os
import time
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_celeba_dataloader, SELECTED_ATTRS
from models import build_cDCGAN, ConditionalGenerator
from utils import ensure_utf8_stdio, generate_and_save_samples, get_preset_conditions


def auto_detect_device() -> torch.device:
    """Tự động phát hiện và lựa chọn thiết bị tính toán tối ưu."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"-> Sử dụng phần cứng tăng tốc NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("-> Sử dụng phần cứng tăng tốc Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("-> Không phát hiện GPU, sử dụng vi xử lý CPU.")
    return device


def get_training_args():
    """Thiết lập các tham số dòng lệnh cho quá trình huấn luyện."""
    parser = argparse.ArgumentParser(description="Huấn luyện cDCGAN trên tập dữ liệu CelebA")
    parser.add_argument("--img-dir", type=str, default="data/celeba/img_align_celeba", help="Thư mục ảnh CelebA")
    parser.add_argument("--attr-file", type=str, default="data/celeba/list_attr_celeba.txt", help="File thuộc tính")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Thư mục lưu kết quả và checkpoint")
    parser.add_argument("--epochs", type=int, default=25, help="Số lượng epoch huấn luyện")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước mini-batch")
    parser.add_argument("--lr", type=float, default=0.0002, help="Tốc độ học (learning rate)")
    parser.add_argument("--beta1", type=float, default=0.5, help="Hệ số beta1 cho optimizer Adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="Hệ số beta2 cho optimizer Adam")
    parser.add_argument("--nz", type=int, default=100, help="Số chiều véc-tơ nhiễu (latent dimension)")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số mẫu nạp (0 = toàn bộ, >0 để test nhanh)")
    parser.add_argument("--log-interval", type=int, default=20, help="Tần suất in log (số batch)")
    parser.add_argument("--save-name", type=str, default="generator.pth", help="Tên file trọng số Generator lưu sau mỗi epoch")
    return parser.parse_args()


def train():
    """Hàm điều khiển toàn bộ quy trình huấn luyện cDCGAN."""
    ensure_utf8_stdio()
    args = get_training_args()
    device = auto_detect_device()
    
    # Tạo các thư mục lưu kết quả
    samples_dir = os.path.join(args.out_dir, "samples")
    checkpoints_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    num_attrs = len(SELECTED_ATTRS)
    print(f"\n========================================================")
    print(f"   BẮT ĐẦU HUẤN LUYỆN cDCGAN TRÊN CELEBA")
    print(f"   - Số thuộc tính điều kiện: {num_attrs} {SELECTED_ATTRS}")
    print(f"   - Số epoch: {args.epochs} | Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.lr} | Latent dim (nz): {args.nz}")
    print(f"========================================================\n")

    # 1. Nạp dữ liệu
    dataloader = get_celeba_dataloader(
        img_dir=args.img_dir,
        attr_file=args.attr_file,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if device.type != "cuda" else 2,
        limit=args.limit
    )
    total_batches = len(dataloader)
    if total_batches == 0:
        raise RuntimeError("DataLoader có 0 batch; kiểm tra lại dữ liệu hoặc batch_size.")
    print(f"Tổng số batch mỗi epoch: {total_batches}")

    # 2. Xây dựng mô hình cDCGAN
    netG, netD = build_cDCGAN(
        nz=args.nz,
        num_attrs=num_attrs,
        ngf=64,
        ndf=64,
        nc=3,
        device=device
    )

    # 3. Thiết lập hàm mất mát (BCELoss) và Optimizer (Adam)
    criterion = nn.BCELoss()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # 4. Chuẩn bị véc-tơ cố định để theo dõi chất lượng ảnh qua từng epoch
    num_fixed_samples = 16
    fixed_noise = torch.randn(num_fixed_samples, args.nz, device=device)
    fixed_conditions = get_preset_conditions(num_attrs=num_attrs, device=device)

    # Lưu ảnh ban đầu (khi chưa huấn luyện)
    init_sample_path = os.path.join(samples_dir, "epoch_000_init.png")
    generate_and_save_samples(netG, fixed_noise, fixed_conditions, init_sample_path, nrow=4)
    print(f"Đã lưu ảnh khởi tạo ngẫu nhiên ban đầu tại: {init_sample_path}")

    # 5. Vòng lặp huấn luyện chính
    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        running_loss_D = 0.0
        running_loss_G = 0.0
        running_D_x = 0.0
        running_D_G_z1 = 0.0

        for i, (real_images, conditions) in enumerate(dataloader, start=1):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            conditions = conditions.to(device)

            # Nhãn mục tiêu: Thật = 1 (hoặc 0.9 label smoothing), Giả = 0
            label_real = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)
            label_fake = torch.full((batch_size,), 0.0, dtype=torch.float32, device=device)

            # -----------------------------------------------------------
            # (1) Cập nhật Discriminator: Tối đa hóa log(D(x|c)) + log(1 - D(G(z|c)|c))
            # -----------------------------------------------------------
            netD.zero_grad()

            # Huấn luyện với ảnh Thật
            output_real = netD(real_images, conditions)
            loss_D_real = criterion(output_real, label_real)
            loss_D_real.backward()
            D_x = output_real.mean().item()

            # Huấn luyện với ảnh Giả
            noise = torch.randn(batch_size, args.nz, device=device)
            fake_images = netG(noise, conditions)
            # Dùng detach() để KHÔNG tính gradient ngược vào Generator ở bước này
            output_fake = netD(fake_images.detach(), conditions)
            loss_D_fake = criterion(output_fake, label_fake)
            loss_D_fake.backward()
            D_G_z1 = output_fake.mean().item()

            loss_D = loss_D_real + loss_D_fake
            optimizerD.step()

            # -----------------------------------------------------------
            # (2) Cập nhật Generator: Tối đa hóa log(D(G(z|c)|c)) (Non-saturating)
            # -----------------------------------------------------------
            netG.zero_grad()
            # Generator muốn Discriminator phân loại ảnh giả thành Thật (nhãn = 1.0)
            target_for_G = torch.full((batch_size,), 1.0, dtype=torch.float32, device=device)
            output_fake_for_G = netD(fake_images, conditions)
            loss_G = criterion(output_fake_for_G, target_for_G)
            loss_G.backward()
            D_G_z2 = output_fake_for_G.mean().item()
            optimizerG.step()

            # Tích lũy số liệu thống kê
            running_loss_D += loss_D.item()
            running_loss_G += loss_G.item()
            running_D_x += D_x
            running_D_G_z1 += D_G_z1

            # In log định kỳ
            if i % args.log_interval == 0 or i == total_batches:
                print(f"[Epoch {epoch:02d}/{args.epochs:02d}] "
                      f"[Batch {i:04d}/{total_batches:04d}] | "
                      f"Loss_D: {loss_D.item():.4f} | "
                      f"Loss_G: {loss_G.item():.4f} | "
                      f"D(x): {D_x:.4f} | "
                      f"D(G(z)): {D_G_z1:.4f} -> {D_G_z2:.4f}")

        # Thống kê trung bình sau mỗi epoch
        epoch_time = time.time() - epoch_start_time
        avg_loss_D = running_loss_D / total_batches
        avg_loss_G = running_loss_G / total_batches
        avg_D_x = running_D_x / total_batches
        avg_D_G_z = running_D_G_z1 / total_batches

        print(f"\n>>> KẾT THÚC EPOCH {epoch:02d} ({epoch_time:.2f}s) | "
              f"Avg Loss_D: {avg_loss_D:.4f} | Avg Loss_G: {avg_loss_G:.4f} | "
              f"Avg D(x): {avg_D_x:.4f} | Avg D(G(z)): {avg_D_G_z:.4f}")

        # 1. Sinh ảnh mẫu cố định để theo dõi
        sample_path = os.path.join(samples_dir, f"epoch_{epoch:03d}.png")
        generate_and_save_samples(netG, fixed_noise, fixed_conditions, sample_path, nrow=4)
        print(f"-> Đã lưu ảnh mẫu kiểm tra trực quan tại: {sample_path}")

        # 2. Lưu file trọng số Generator (theo yêu cầu đề bài: 'generator.pth' sau mỗi epoch)
        # Lưu vào thư mục gốc để app.py load trực tiếp
        payload = {"G": netG.state_dict(), "epoch": epoch, "args": vars(args)}
        root_model_path = args.save_name
        if os.path.dirname(root_model_path):
            os.makedirs(os.path.dirname(root_model_path), exist_ok=True)
        torch.save(payload, root_model_path)

        
        # Đồng thời lưu bản sao theo epoch trong thư mục checkpoints
        epoch_model_path = os.path.join(checkpoints_dir, f"generator_epoch_{epoch:03d}.pth")
        torch.save(payload, epoch_model_path)

        print(f"-> Đã lưu trọng số mô hình: '{root_model_path}' và '{epoch_model_path}'\n")

    total_training_time = time.time() - start_total_time
    print(f"========================================================")
    print(f"   HUẤN LUYỆN HOÀN TẤT TRONG {total_training_time / 60:.2f} PHÚT!")
    print(f"   File trọng số sẵn sàng cho Web: '{args.save_name}'")
    print(f"========================================================")


if __name__ == "__main__":
    train()
