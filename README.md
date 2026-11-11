# cDCGAN Face Generator – Sinh Ảnh Khuôn Mặt Người Có Điều Kiện

> **Đồ án Trí tuệ Nhân tạo (TTNT) - Nhóm DHN**  
> Dự án xây dựng mô hình **Conditional Deep Convolutional GAN (cDCGAN)** trên tập dữ liệu **CelebA**, cho phép sinh ảnh khuôn mặt người có điều kiện (Giới tính, Nụ cười, Kính mắt, Màu tóc) kết hợp giao diện Web **Streamlit** tương tác thời gian thực.

---

## 📁 Cấu Trúc Dự Án

Dự án được thiết kế phân tầng rõ ràng theo chuẩn Senior AI Engineer và Full-stack Developer:

```text
TTNT_NhomDHN/
├── requirements.txt      # Danh sách thư viện: torch, torchvision, streamlit, pytorch-fid, ...
├── dataset.py           # Custom Dataset đọc CelebA, transforms và trích xuất véc-tơ thuộc tính
├── models.py            # Kiến trúc cDCGAN: ConditionalGenerator & ConditionalDiscriminator
├── train.py             # Vòng lặp huấn luyện GAN xen kẽ, tự động nhận diện CPU/CUDA/MPS
├── utils.py             # Tiện ích lưu ảnh, tạo mẫu fixed, và hàm tính chỉ số FID (pytorch-fid)
├── app.py               # Ứng dụng Web Streamlit trực quan hóa và tương tác sinh ảnh
├── notebooks/           # Notebook chạy huấn luyện trên Kaggle / Colab (GPU T4/P100)
│   ├── kaggle_cDCGAN_CelebA.ipynb
│   └── colab_train.ipynb
└── outputs/             # Thư mục lưu ảnh mẫu qua các epoch và checkpoints mô hình
```

---

## 🚀 Quy Trình Chuẩn Kỹ Sư (4 Bước)

### Bước 1: Dựng Khung & Kiểm Thử Tại Chỗ (Local Smoke Test)

1. **Khởi tạo môi trường ảo và cài đặt thư viện:**
   ```bash
   python -m venv .venv
   # Kích hoạt trên Windows:
   .venv\Scripts\activate
   # Hoặc trên Linux/macOS:
   source .venv/bin/activate

   # Cài đặt các thư viện cần thiết:
   pip install -r requirements.txt
   ```

2. **Kiểm tra kiến trúc mạng và luồng dữ liệu (Debug trên CPU):**
   ```bash
   # Kiểm tra Generator, Discriminator và kích thước tensor:
   python models.py

   # Kiểm tra Custom Dataset & Transforms:
   python dataset.py

   # Chạy thử 1 lượt train nhanh (1 epoch, 100 mẫu giả lập) để chắc chắn không có lỗi logic:
   python train.py --epochs 1 --limit 100 --batch-size 16 --save-name outputs/smoke_generator.pth
   ```

---

### Bước 2: Đưa Lên Kaggle Huấn Luyện (Training trên GPU)

Do mô hình GAN cần năng lực tính toán cao của card đồ họa, bạn nên đưa code lên **Kaggle Notebook**:
1. Đăng nhập [Kaggle](https://www.kaggle.com/) -> Tạo **New Notebook**.
2. Tại bảng điều khiển bên phải:
   - **Accelerator**: Chọn **GPU T4 x2** hoặc **GPU P100**.
   - **Data**: Nhấn *Add Data* -> Tìm kiếm dataset `celeba-dataset` (hoặc `jessicali9530/celeba-dataset`).
3. Tải lên notebook `notebooks/kaggle_cDCGAN_CelebA.ipynb` hoặc copy mã nguồn của các file `models.py`, `dataset.py`, `train.py` vào notebook.
4. Chạy huấn luyện khoảng 20-30 epochs (mất khoảng 1.5 - 2 giờ trên GPU).
5. Mô hình sẽ liên tục lưu các ảnh mẫu tại `outputs/samples/` và file trọng số `generator.pth`.

---

### Bước 3: Thu Hoạch Trọng Số (Weights)

Sau khi huấn luyện hoàn tất trên Kaggle:
1. Vào mục **Output** ở bảng bên phải của Kaggle Notebook.
2. Tải duy nhất file **`generator.pth`** về máy tính cá nhân.
3. Đặt file `generator.pth` trực tiếp vào thư mục gốc của dự án (`TTNT_NhomDHN/generator.pth`).

---

### Bước 4: Khởi Chạy Giao Diện Web

Dự án cung cấp **2 lựa chọn giao diện** tùy theo nhu cầu của bạn:

#### Lựa chọn 1: Giao Diện Web Studio Hiện Đại (Khuyến nghị số 1 ⭐⭐⭐⭐⭐)
Được xây dựng trên nền tảng **Flask REST API + TailwindCSS Glassmorphism** (Phong cách Midjourney / Apple Studio):
```bash
python web_app.py
```
* Mở trình duyệt tại: `http://localhost:5000`
* **Ưu điểm vượt trội:**
  * Giao diện kính mờ (Glassmorphism) đẳng cấp, tối giản, mượt mà và hiện đại.
  * Phản hồi tức thì, không bị tải lại toàn trang (No page reload).
  * Hiệu ứng xem ảnh phóng to toàn màn hình (Lightbox Modal) chỉ với 1 click.
  * Tích hợp tính năng **🎬 Xuất Video GIF Biến Hình Động** để chèn trực tiếp vào slide PowerPoint thuyết trình!

#### Lựa chọn 2: Giao Diện Streamlit Dashboard
```bash
python -m streamlit run app.py
```
* Mở trình duyệt tại: `http://localhost:8501`

---

## 📊 Đánh Giá Bằng Chỉ Số FID (Fréchet Inception Distance)

Sau khi huấn luyện, bạn có thể đánh giá khoảng cách phân phối giữa tập ảnh thật và tập ảnh sinh ra bằng hàm trong `utils.py`:

```python
from utils import calculate_fid_score

# Tính FID giữa thư mục ảnh thật và ảnh sinh ra:
fid = calculate_fid_score(
    real_images_dir="data/celeba/img_align_celeba",
    fake_images_dir="outputs/eval_fake_images",
    device="cuda"
)
print(f"Chỉ số FID đạt được: {fid:.2f}")
```
*(Chỉ số FID càng thấp thể hiện ảnh sinh ra càng sắc nét và đa dạng tự nhiên).*

---

## 👥 Thành Viên Thực Hiện (Nhóm DHN)
- Đồ án môn học: Trí tuệ Nhân tạo (TTNT).
- Kiến trúc mô hình: cDCGAN (Conditional Deep Convolutional Generative Adversarial Networks).
- Nền tảng: PyTorch, Torchvision, Streamlit, Kaggle GPU.
