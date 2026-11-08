# BÁO CÁO: SINH ẢNH KHUÔN MẶT NGƯỜI BẰNG DCGAN

> Các mục có dấu `[ĐIỀN]` cần cập nhật số liệu/hình sau khi chạy `notebooks/colab_train.ipynb`.

## 1. Đặt vấn đề
Generative AI tạo dữ liệu tổng hợp; GAN là kiến trúc nòng cốt. Thách thức: huấn luyện mất cân bằng G–D và *mode collapse*. Mục tiêu: nghiên cứu mạng tích chập sâu cho GAN và sinh ảnh chân dung 64×64 hoàn toàn mới từ nhiễu `z ~ N(0, I)` trong không gian tiềm ẩn.

## 2. Cơ sở lý thuyết
### 2.1 GAN
Hai mạng đối kháng: Generator `G(z)` ánh xạ nhiễu sang ảnh; Discriminator `D(x)` ∈ [0,1] ước lượng xác suất ảnh là thật. Bài toán minimax:

$$\min_G \max_D V(D,G)=\mathbb{E}_{x\sim p_{data}}[\log D(x)]+\mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$$

Với G cố định, D tối ưu là $D^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}$; thay vào V thì bài toán tương đương cực tiểu hóa độ phân kỳ Jensen–Shannon giữa $p_{data}$ và $p_g$, đạt cực tiểu toàn cục khi $p_g=p_{data}$, lúc đó $D^*\equiv 1/2$ (cân bằng Nash).

Thực hành dùng *non-saturating loss*: G tối đa hóa $\log D(G(z))$ (thay vì cực tiểu $\log(1-D(G(z)))$) để tránh gradient triệt tiêu đầu huấn luyện. Cả hai đều là Binary Cross-Entropy với nhãn tương ứng.

### 2.2 Mạng tích chập sâu & DCGAN
- **Conv2d (stride 2)** thay pooling: D học giảm kích thước (64→32→16→8→4→1).
- **ConvTranspose2d** (tích chập chuyển vị): G học tăng kích thước (1→4→8→16→32→64). Kích thước ra: $H_{out}=(H_{in}-1)\cdot s-2p+k$ → với k=4, s=2, p=1 độ phân giải nhân đôi.
- BatchNorm (trừ lớp vào của D và lớp ra của G); ReLU trong G, LeakyReLU(0.2) trong D; Tanh đầu ra G (ảnh trong [-1,1]); khởi tạo N(0, 0.02); Adam(lr=2e-4, β₁=0.5); không dùng lớp fully-connected ẩn.

### 2.3 Khó khăn huấn luyện
| Hiện tượng | Biểu hiện | Biện pháp trong dự án |
|---|---|---|
| Mất cân bằng hội tụ | D(x)→1, D(G(z))→0, loss G tăng; G ngừng học | β₁=0.5, cùng lr; theo dõi D(x), D(G(z)); (tùy chọn) label smoothing `--real-label 0.9` |
| Mode collapse | G chỉ sinh vài kiểu mặt | Theo dõi `diversity` (ảnh sinh vs thật) + xem lưới `fixed_noise` theo epoch |
| Gradient triệt tiêu | loss G không giảm | Non-saturating loss, LeakyReLU |

### 2.4 FID
$$FID=\lVert\mu_r-\mu_g\rVert^2+\mathrm{Tr}\big(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\big)$$
$\mu,\Sigma$ là trung bình/hiệp phương sai của đặc trưng 2048 chiều (pool3) của Inception-v3 trên ảnh thật và ảnh sinh. FID thấp ⇒ phân phối giống nhau hơn. Lưu ý: dùng trọng số torchvision nên không so sánh trực tiếp với số FID công bố từ bản TensorFlow.

## 3. Dữ liệu & tiền xử lý (GĐ1)
CelebA Align & Cropped: 202.599 ảnh 178×218. Pipeline (`src/preprocess.py`, `src/dataset.py`): center-crop 148×148 → resize 64×64 (OpenCV, INTER_AREA) → lưu uint8 → chuẩn hóa $x/127.5-1$ ∈ [-1,1]; lật ngang ngẫu nhiên khi huấn luyện.

## 4. Thiết kế mô hình (GĐ2)
| | Generator | Discriminator |
|---|---|---|
| Vào | z: 100×1×1 | ảnh 3×64×64 |
| Lớp | 5 × ConvTranspose2d (512→256→128→64→3) | 5 × Conv2d (64→128→256→512→1) |
| Chuẩn hóa/kích hoạt | BN + ReLU, Tanh ra | BN (trừ lớp đầu) + LeakyReLU, Sigmoid ra |
| Tham số | `[ĐIỀN]` (chạy `python -m src.models`) | `[ĐIỀN]` |

Loss: `nn.BCELoss`; Optimizer: Adam (lr 2e-4, β=(0.5, 0.999)), batch 128.

## 5. Huấn luyện (GĐ3)
Môi trường: Google Colab GPU `[ĐIỀN loại GPU]`; `[ĐIỀN]` epoch, thời gian/epoch `[ĐIỀN]`.
- Hình: `training_curves.png` (loss G/D, D(x)/D(G(z)), FID, diversity).
- Phân tích hội tụ: `[ĐIỀN]` – D(x), D(G(z)) tiến về khoảng 0.5 hay lệch? Epoch nào FID thấp nhất? Các tinh chỉnh siêu tham số đã thử (lr, label smoothing, ngf/ndf, batch size) và tác động.

## 6. Kết quả & đánh giá (GĐ4)
### 6.1 Định lượng
| Epoch | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|
| FID | `[ĐIỀN]` | | | | |

FID tốt nhất: `[ĐIỀN]` tại epoch `[ĐIỀN]` (`eval_report.json`).

### 6.2 Định tính
- Tiến trình theo epoch: `samples/epoch_001.png` → `epoch_025.png` (chèn 4–5 ảnh).
- Nhận xét ngũ quan (mắt, mũi, miệng), kết cấu da, tóc, phông nền, lỗi thường gặp (méo kính, răng, tai) `[ĐIỀN]`.
- Đa dạng/mode collapse: `diversity_ratio = [ĐIỀN]` (≈1 là tốt) và nhận xét về đa dạng giới tính, tuổi, hướng mặt, biểu cảm, màu tóc.
- Nội suy latent (`interpolation.png`): chuyển đổi mượt ⇒ G học được không gian tiềm ẩn liên tục, không ghi nhớ.
- Nearest-neighbor (`nearest_neighbors.png`): ảnh sinh khác ảnh thật gần nhất ⇒ là khuôn mặt mới.

## 7. Kết luận & hướng phát triển
`[ĐIỀN]`. Hướng mở rộng: WGAN-GP / spectral norm, StyleGAN, ảnh 128×128+, điều kiện theo thuộc tính CelebA (cGAN), tính FID với tập lớn hơn.

## Tài liệu tham khảo
1. Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014.
2. Radford, Metz, Chintala, *Unsupervised Representation Learning with Deep Convolutional GANs*, ICLR 2016.
3. Liu et al., *Deep Learning Face Attributes in the Wild* (CelebA), ICCV 2015.
4. Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (FID), NeurIPS 2017.
5. Szegedy et al., *Rethinking the Inception Architecture for Computer Vision*, CVPR 2016.
6. Salimans et al., *Improved Techniques for Training GANs*, NeurIPS 2016.
