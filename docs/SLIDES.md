---
marp: true
theme: default
paginate: true
---

# Sinh ảnh khuôn mặt người bằng DCGAN
Nhóm DHN – Môn Trí tuệ nhân tạo

---

# 1. Bối cảnh & mục tiêu
- Generative AI → GAN là kiến trúc nòng cốt tạo dữ liệu tổng hợp
- Thách thức: mất cân bằng G–D, **mode collapse**
- Mục tiêu: sinh ảnh mặt người 64×64 mới, chân thực, từ nhiễu `z ∈ R^100`

---

# 2. Lý thuyết GAN
$$\min_G\max_D\ \mathbb{E}_{x}[\log D(x)]+\mathbb{E}_z[\log(1-D(G(z)))]$$
- D tối ưu: $D^*=\frac{p_{data}}{p_{data}+p_g}$ → cực tiểu JS-divergence
- Cân bằng Nash: $p_g=p_{data}$, $D\equiv 0.5$
- Thực hành: non-saturating loss cho G (BCE)

---

# 3. Kiến trúc DCGAN
- **G**: z → 5×ConvTranspose2d (512→256→128→64→3), BN+ReLU, Tanh
- **D**: 5×Conv2d (64→128→256→512→1), LeakyReLU(0.2), Sigmoid
- BCE Loss, Adam (lr 2e-4, β₁=0.5), init N(0,0.02)

---

# 4. Dữ liệu CelebA
- 202.599 ảnh Align & Cropped
- Center crop 148 → resize 64×64 → chuẩn hóa [-1, 1]
- Script `src/preprocess.py` (OpenCV, đa tiến trình)

---

# 5. Hệ thống & lộ trình
- Python, PyTorch, TorchVision, NumPy, OpenCV, Matplotlib
- Code trên VS Code → huấn luyện trên GPU Google Colab
- GĐ1 Dữ liệu → GĐ2 Mô hình → GĐ3 Huấn luyện → GĐ4 Đánh giá & báo cáo

---

# 6. Huấn luyện
*(chèn `training_curves.png`)*
- Loss G/D, D(x), D(G(z)) → cân bằng quanh 0.5?
- Siêu tham số đã tinh chỉnh: `[ĐIỀN]`

---

# 7. Kết quả định tính
*(chèn `samples/epoch_001/005/015/025.png`)*
- Ngũ quan, da, tóc, phông nền: `[ĐIỀN]`

---

# 8. Đánh giá định lượng – FID
- Inception-v3 pool3 (2048-d), $FID=\lVert\mu_r-\mu_g\rVert^2+\mathrm{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$
- FID theo epoch: `[ĐIỀN]` – tốt nhất `[ĐIỀN]`

---

# 9. Kiểm chứng mode collapse
- Diversity ratio (sinh/thật) = `[ĐIỀN]`
- Nội suy latent mượt *(chèn `interpolation.png`)*
- Ảnh sinh ≠ ảnh thật gần nhất *(chèn `nearest_neighbors.png`)*

---

# 10. Kết luận & hướng phát triển
- Kết quả: `[ĐIỀN]`
- Mở rộng: WGAN-GP, StyleGAN, cGAN theo thuộc tính, ảnh độ phân giải cao

---

# Cảm ơn! Q&A
