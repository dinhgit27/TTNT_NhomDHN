# DCGAN – Sinh ảnh khuôn mặt người trên CelebA

Đồ án TTNT – Nhóm DHN. Mô hình Deep Convolutional GAN (PyTorch) sinh ảnh chân dung 64×64 từ véc-tơ nhiễu `z ∈ R^100`, huấn luyện trên CelebA (202.599 ảnh), đánh giá bằng FID và quan sát định tính / độ đa dạng (mode collapse).

## Cấu trúc

```
src/
  preprocess.py   # GĐ1: center crop 148 -> resize 64x64 -> .npy uint8 (đọc thẳng từ archive.zip)
  dataset.py      # Dataset, chuẩn hóa về [-1, 1]
  models.py       # GĐ2: Generator (ConvTranspose2d), Discriminator (Conv2d), weights_init
  train.py        # GĐ3: BCE + Adam, log CSV, ảnh theo epoch, checkpoint, FID định kỳ, resume
  fid.py          # GĐ4: module FID độc lập (Inception-v3, có CLI)
  evaluate.py     # GĐ4: FID, độ đa dạng, nội suy latent, ảnh thật gần nhất
  utils.py        # lưới ảnh, diversity score, slerp, vẽ đường cong huấn luyện
notebooks/colab_train.ipynb   # chạy toàn bộ pipeline trên GPU Colab
docs/BAO_CAO.md, docs/SLIDES.md   # báo cáo + slide (Marp) để điền số liệu sau khi train
```

## Chạy trên máy cục bộ (VS Code)

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt            # cài torch theo hướng dẫn pytorch.org nếu có GPU
python -m src.models                       # kiểm tra kích thước tensor
python -m src.preprocess --src archive.zip --dst data/processed/celeba_64.npy --limit 2000   # chạy thử
python -m src.train --data data/processed/celeba_64.npy --out outputs/smoke --epochs 1 --fid-every 0 --workers 0
```

## Huấn luyện đầy đủ (Colab)

Upload `src/`, `requirements.txt`, `archive.zip` lên `MyDrive/TTNT_NhomDHN/`, mở `notebooks/colab_train.ipynb`, chọn GPU và chạy lần lượt các cell. 25 epoch trên GPU Colab (T4) – thời gian thực tế phụ thuộc phiên Colab; có `--resume` nếu bị ngắt.

## Kết quả sinh ra (`outputs/run1/`)

| File | Ý nghĩa |
|---|---|
| `samples/epoch_XXX.png` | 64 ảnh từ cùng `fixed_noise` qua từng epoch (đánh giá định tính) |
| `metrics.csv`, `iter_losses.csv` | loss G/D, D(x), D(G(z)), độ đa dạng, FID |
| `training_curves.png` | đồ thị hội tụ |
| `best_fid.pt`, `last.pt`, `checkpoints/` | trọng số |
| `eval/` | kết quả `src.evaluate` (lưới ảnh, nội suy, nearest-neighbor, `eval_report.json`) |

## Tiêu chí nghiệm thu

- **FID** giảm và ổn định theo epoch (ghi số liệu thực tế vào báo cáo sau khi huấn luyện).
- **Đa dạng**: `diversity_ratio` (ảnh sinh / ảnh thật) gần 1; ảnh từ `fixed_noise` khác nhau về giới tính, tóc, góc mặt, biểu cảm, phông nền.
- **Không chép dữ liệu**: ảnh sinh khác ảnh thật gần nhất (`nearest_neighbors.png`).
