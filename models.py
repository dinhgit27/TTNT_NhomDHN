"""
models.py - Kiến trúc mạng Conditional Deep Convolutional GAN (cDCGAN).

Tác giả: Senior AI Engineer & Full-stack Developer
Nhiệm vụ:
  1. Generator (Mạng sinh):
     - Nhận đầu vào là véc-tơ nhiễu z (100 chiều) ghép với véc-tơ nhãn điều kiện c (num_attrs chiều).
     - Chiếu và mở rộng dần không gian qua các tầng ConvTranspose2d, BatchNorm2d, ReLU.
     - Tầng cuối cùng áp dụng hàm kích hoạt Tanh để sinh ảnh 3x64x64 trong khoảng [-1, 1].
  2. Discriminator (Mạng phân biệt):
     - Nhận đầu vào là ảnh 3x64x64 ghép với các kênh điều kiện (condition channels).
     - Thu hẹp không gian dần qua các tầng Conv2d, BatchNorm2d, LeakyReLU(0.2).
     - Tầng cuối cùng áp dụng Sigmoid để trả về xác suất ảnh là Thật hay Giả trong ngữ cảnh nhãn điều kiện.
  3. weights_init:
     - Khởi tạo trọng số ngẫu nhiên theo phân phối chuẩn N(0, 0.02) theo bài báo DCGAN gốc.
"""

from typing import Tuple
import torch
import torch.nn as nn


def weights_init(m: nn.Module) -> None:
    """
    Khởi tạo trọng số mạng theo nguyên tắc DCGAN (Radford et al.):
    - Conv/ConvTranspose: Trọng số ~ Normal(0.0, 0.02).
    - BatchNorm: Trọng số ~ Normal(1.0, 0.02), bias = 0.
    """
    classname = m.__class__.__name__
    if "Conv" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif "BatchNorm" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class ConditionalGenerator(nn.Module):
    """
    Mạng Sinh có Điều kiện (Conditional Generator):
    Ghép véc-tơ nhiễu (noise) z (100) và điều kiện c (num_attrs) -> Chiếu lên không gian đặc trưng
    và giải chập liên tiếp để đạt kích thước ảnh (3, 64, 64).
    Hỗ trợ cả trường hợp num_attrs = 0 (Unconditional DCGAN).
    """
    def __init__(self, nz: int = 100, num_attrs: int = 6, ngf: int = 64, nc: int = 3):
        """
        Tham số:
            nz: Số chiều của véc-tơ nhiễu tiềm ẩn (latent noise vector, mặc định 100)
            num_attrs: Số lượng nhãn thuộc tính điều kiện (0 nếu là Unconditional GAN)
            ngf: Kích thước feature map cơ sở của Generator (mặc định 64)
            nc: Số kênh ảnh đầu ra (3 cho ảnh màu RGB)
        """
        super().__init__()
        self.nz = nz
        self.num_attrs = num_attrs
        in_channels = nz + num_attrs

        # Chuỗi các lớp giải chập (Transposed Convolution)
        self.net = nn.Sequential(
            # Input: (nz + num_attrs) x 1 x 1 -> Feature map: (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(in_channels, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf * 8) x 4 x 4 -> (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf * 4) x 8 x 8 -> (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf * 2) x 16 x 16 -> ngf x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # ngf x 32 x 32 -> nc x 64 x 64
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Đầu ra chuẩn hóa trong dải [-1, 1]
        )

    def forward(self, noise: torch.Tensor, conditions: torch.Tensor = None) -> torch.Tensor:
        """
        Lan truyền xuôi (Forward):
            noise: Tensor kích thước (B, nz) hoặc (B, nz, 1, 1)
            conditions: Tensor véc-tơ điều kiện kích thước (B, num_attrs) hoặc None
        Trả về:
            Ảnh tổng hợp: Tensor kích thước (B, 3, 64, 64)
        """
        if noise.dim() == 2:
            noise = noise.unsqueeze(2).unsqueeze(3)  # Chuyển (B, nz) -> (B, nz, 1, 1)

        if self.num_attrs > 0 and conditions is not None:
            if conditions.dim() == 2:
                conditions = conditions.unsqueeze(2).unsqueeze(3)  # Chuyển (B, num_attrs) -> (B, num_attrs, 1, 1)
            z_c = torch.cat([noise, conditions], dim=1)  # (B, nz + num_attrs, 1, 1)
        else:
            z_c = noise

        return self.net(z_c)


class ConditionalDiscriminator(nn.Module):
    """
    Mạng Phân biệt có Điều kiện (Conditional Discriminator):
    Nhận ảnh x (3, 64, 64) và mở rộng véc-tơ điều kiện c thành các kênh không gian (num_attrs, 64, 64).
    Ghép lại thành tensor (3 + num_attrs, 64, 64) và đi qua các tầng tích chập để đưa ra xác suất Thật/Giả.
    """
    def __init__(self, num_attrs: int = 6, ndf: int = 64, nc: int = 3):
        """
        Tham số:
            num_attrs: Số lượng nhãn thuộc tính điều kiện (mặc định 6)
            ndf: Kích thước feature map cơ sở của Discriminator (mặc định 64)
            nc: Số kênh ảnh đầu vào (3 cho ảnh màu RGB)
        """
        super().__init__()
        self.num_attrs = num_attrs
        in_channels = nc + num_attrs

        self.net = nn.Sequential(
            # Input: (nc + num_attrs) x 64 x 64 -> ndf x 32 x 32
            # Lưu ý: Không dùng BatchNorm ở lớp đầu tiên theo DCGAN guidelines
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf x 32 x 32 -> (ndf * 2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 2) x 16 x 16 -> (ndf * 4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 4) x 8 x 8 -> (ndf * 8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 8) x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # Đầu ra là xác suất trong khoảng [0, 1]
        )

    def forward(self, images: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền xuôi (Forward):
            images: Tensor ảnh (B, 3, 64, 64)
            conditions: Tensor véc-tơ điều kiện (B, num_attrs)
        Trả về:
            Xác suất Thật/Giả: Tensor 1D kích thước (B,)
        """
        batch_size, _, height, width = images.shape
        # Mở rộng (expand) véc-tơ điều kiện thành các ma trận 2D có cùng chiều cao/rộng với ảnh
        c_spatial = conditions.view(batch_size, self.num_attrs, 1, 1).expand(batch_size, self.num_attrs, height, width)

        # Ghép ảnh và kênh điều kiện theo chiều kênh (dim=1)
        x_c = torch.cat([images, c_spatial], dim=1)  # (B, 3 + num_attrs, 64, 64)
        return self.net(x_c).view(batch_size)


def build_cDCGAN(
    nz: int = 100,
    num_attrs: int = 6,
    ngf: int = 64,
    ndf: int = 64,
    nc: int = 3,
    device: torch.device = torch.device("cpu")
) -> Tuple[ConditionalGenerator, ConditionalDiscriminator]:
    """
    Hàm tạo và khởi tạo toàn bộ mô hình cDCGAN:
    1. Khởi tạo đối tượng Generator và Discriminator.
    2. Áp dụng chuẩn khởi tạo trọng số DCGAN (weights_init).
    3. Đẩy mô hình lên thiết bị (device).
    """
    netG = ConditionalGenerator(nz=nz, num_attrs=num_attrs, ngf=ngf, nc=nc).to(device)
    netD = ConditionalDiscriminator(num_attrs=num_attrs, ndf=ndf, nc=nc).to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    return netG, netD


if __name__ == "__main__":
    # Test kiểm tra nhanh kích thước tensor và số lượng tham số
    print("--- KIỂM TRA MÔ HÌNH cDCGAN (MODELS.PY) ---")
    device = torch.device("cpu")
    nz = 100
    num_attrs = 6
    batch_size = 4

    netG, netD = build_cDCGAN(nz=nz, num_attrs=num_attrs, device=device)

    # Tạo tensor giả lập
    fake_noise = torch.randn(batch_size, nz, device=device)
    fake_conditions = torch.randint(0, 2, (batch_size, num_attrs), dtype=torch.float32, device=device)

    # Kiểm tra Generator
    generated_imgs = netG(fake_noise, fake_conditions)
    print(f"Generator Input: noise {fake_noise.shape} + conditions {fake_conditions.shape}")
    print(f"Generator Output: {generated_imgs.shape} (Kỳ vọng: [{batch_size}, 3, 64, 64])")

    # Kiểm tra Discriminator
    pred_prob = netD(generated_imgs, fake_conditions)
    print(f"Discriminator Output: {pred_prob.shape} (Kỳ vọng: [{batch_size}])")
    print(f"Giá trị xác suất mẫu: {pred_prob.tolist()}")

    # Thống kê số lượng tham số
    params_G = sum(p.numel() for p in netG.parameters()) / 1e6
    params_D = sum(p.numel() for p in netD.parameters()) / 1e6
    print(f"Tổng số tham số: Generator = {params_G:.2f}M | Discriminator = {params_D:.2f}M")
    print("-> Toàn bộ cấu trúc cDCGAN hoạt động chuẩn xác!")
