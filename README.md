# Hệ Thống Sinh Ảnh Khuôn Mặt Người Bằng GANs (DCGAN & Conditional DCGAN)

Đồ án Học phần **Trí Tuệ Nhân Tạo** — Sinh ảnh khuôn mặt người bằng mạng đối nghịch tạo sinh (**GANs**) trên tập dữ liệu **CelebA (202,599 ảnh)** và đánh giá chất lượng bằng chỉ số **FID (Fréchet Inception Distance)**.

---

## Tính năng chính
1. **Tự Sinh Ảnh Khuôn Mặt Mới (Unconditional / Autonomous GAN Generation):**
   - Sinh tự động danh sách khuôn mặt mới từ không gian tiềm ẩn $z \sim \mathcal{N}(0, \psi^2 I)$ qua mạng Generator (`DCGAN` & `Conditional DCGAN`) kết hợp tinh chỉnh đa tầng.
   - Chấm điểm độ chân thực qua mạng `Discriminator` $D(G(z))$ và điều chỉnh hệ số `Truncation Trick` ($\psi$).
2. **Kết Hợp 2 Ảnh Sinh Ảnh Mới (Dual-Image Face Fusion & Latent Interpolation):**
   - Hỗ trợ chọn ảnh từ tập dữ liệu CelebA hoặc tải ảnh bất kỳ từ máy tính.
   - Phát hiện 5 điểm mốc khuôn mặt bằng mạng sâu `YuNet (FaceDetectorYN)`, căn chỉnh chuẩn hóa Affine 3 điểm + biến đổi đơn điệu cosin nâng 5 điểm mốc, đồng bộ sắc thái da `CIE-LAB` kèm cân bằng chiếu sáng nội tại 2D và pha trộn tháp `Laplacian Pyramid` đa tần số.
   - Sinh dải nội suy chuyển tiếp liên tục 7 bước ($\alpha \in [0.00, 1.00]$) bằng `SLERP`.
3. **Đánh Giá Chất Lượng Ảnh Bằng Chỉ Số FID (Fréchet Inception Distance):**
   - Tính toán trực tiếp khoảng cách Wasserstein-2 giữa phân phối đặc trưng ảnh thật CelebA $(\mu_r, \Sigma_r)$ và ảnh sinh bởi Generator $(\mu_g, \Sigma_g)$ cùng các chỉ số `KID`, `Inception Score`.
4. **Huấn Luyện Trên Kaggle GPU & Nạp Trọng Số Trực Tiếp (`.pth`):**
   - Notebook huấn luyện chuẩn chia tập `80/20` (`162,079` ảnh Train / `40,520` ảnh Val-Test): `notebooks/CelebA_DCGAN_cDCGAN_Kaggle_Training.ipynb`.
   - Tích hợp sẵn file trọng số đã huấn luyện `checkpoints/celeba_gan_bundle.pth`.

---

## Cấu trúc thư mục
```text
├── app.py                                 # Máy chủ Web Flask & REST API
├── requirements.txt                       # Các thư viện Python cần thiết
├── core/
│   ├── dataset.py                         # Quản lý dữ liệu CelebA & bộ nhớ đệm
│   ├── models.py                          # Kiến trúc DCGAN, cDCGAN, Encoder & bộ tổng hợp ảnh
│   ├── fid_metrics.py                     # Bộ tính toán chỉ số FID, KID, IS
│   └── trainer.py                         # Bộ huấn luyện tương tác & quản lý checkpoint
├── checkpoints/
│   ├── celeba_gan_bundle.pth              # Trọng số mạng GAN đã huấn luyện trên Kaggle GPU
│   └── face_detection_yunet_2023mar.onnx  # Mô hình phát hiện 5 điểm mốc khuôn mặt YuNet
├── data_cache/                            # Bộ mẫu CelebA & ngân hàng đặc trưng chuẩn hóa
├── notebooks/
│   └── CelebA_DCGAN_cDCGAN_Kaggle_Training.ipynb
├── templates/
│   └── index.html                         # Giao diện Web Studio
└── static/
    ├── css/style.css
    └── js/main.js
```

---

## Hướng dẫn cài đặt & chạy chương trình
```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Khởi chạy ứng dụng Web
python app.py
```
Sau đó mở trình duyệt truy cập: **`http://127.0.0.1:5000`**
