"""Giai đoạn 3 - Vòng lặp huấn luyện DCGAN.

Mỗi iteration:
  1) Cập nhật D: tối đa hóa log D(x) + log(1 - D(G(z)))   (BCE với nhãn thật=1, giả=0)
  2) Cập nhật G: tối đa hóa log D(G(z))                   (non-saturating loss, nhãn giả→1)
Cả hai dùng Adam(lr=2e-4, beta1=0.5) để huấn luyện ổn định, tiến tới cân bằng Nash
(D(x) ≈ D(G(z)) ≈ 0.5).

Sau mỗi epoch: lưu lưới ảnh từ ``fixed_noise`` (đánh giá định tính), checkpoint, ghi
CSV và đo độ đa dạng; mỗi ``--fid-every`` epoch tính thêm FID.

Ví dụ::

    python -m src.train --data data/processed/celeba_64.npy --out outputs/run1 --epochs 25
    python -m src.train --data data/processed/celeba_64.npy --out outputs/run1 --epochs 25 --resume
    python -m src.train --data data/processed/celeba_64.npy --out outputs/smoke --epochs 1 --limit 2000 --fid-every 0  # chạy thử
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CelebA64
from .fid import InceptionFeatures, fid_of_generator, real_stats
from .models import build_models
from .utils import diversity_score, plot_training, save_grid

METRIC_FIELDS = ["epoch", "loss_d", "loss_g", "d_x", "d_gz", "diversity", "diversity_real", "fid", "time_s"]


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/processed/celeba_64.npy")
    p.add_argument("--out", default="outputs/run1")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--nz", type=int, default=100, help="số chiều không gian tiềm ẩn")
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--real-label", type=float, default=1.0, help="<1 (vd 0.9) để bật one-sided label smoothing")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="chỉ dùng N ảnh đầu (chạy thử)")
    p.add_argument("--fid-every", type=int, default=5, help="tính FID mỗi N epoch (0 = tắt)")
    p.add_argument("--fid-samples", type=int, default=5000)
    p.add_argument("--fid-real", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="tiếp tục từ outputs/<run>/last.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_csv(path: str, fields: list[str], rows: list[dict]) -> None:
    new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    a = get_args()
    set_seed(a.seed)
    dev = torch.device(a.device)
    os.makedirs(os.path.join(a.out, "samples"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "checkpoints"), exist_ok=True)
    torch.backends.cudnn.benchmark = True

    # ---- dữ liệu ----
    ds = CelebA64(a.data, flip=True, limit=a.limit)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                    drop_last=True, pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    print(f"Dữ liệu: {len(ds)} ảnh | {len(dl)} iteration/epoch | thiết bị: {dev}")

    # ---- mô hình, loss, optimizer ----
    G, D = build_models(a.nz, a.ngf, a.ndf, 3, dev)
    criterion = nn.BCELoss()
    optD = torch.optim.Adam(D.parameters(), lr=a.lr, betas=(a.beta1, a.beta2))
    optG = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(a.beta1, a.beta2))
    fixed_noise = torch.randn(64, a.nz, 1, 1, device=dev, generator=torch.Generator(dev).manual_seed(a.seed))

    start_epoch, best_fid, global_iter = 1, float("inf"), 0
    last_path = os.path.join(a.out, "last.pt")
    if a.resume and os.path.isfile(last_path):
        ck = torch.load(last_path, map_location=dev)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        optG.load_state_dict(ck["optG"]); optD.load_state_dict(ck["optD"])
        start_epoch, best_fid, global_iter = ck["epoch"] + 1, ck["best_fid"], ck["global_iter"]
        fixed_noise = ck["fixed_noise"].to(dev)
        print(f"Tiếp tục từ epoch {start_epoch}")
    else:
        for f in ("metrics.csv", "iter_losses.csv"):
            if os.path.isfile(os.path.join(a.out, f)):
                os.remove(os.path.join(a.out, f))
    with open(os.path.join(a.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(a), f, indent=2)

    # ---- FID & độ đa dạng của ảnh thật (mốc so sánh) ----
    inception, mu_r, sigma_r = None, None, None
    if a.fid_every > 0:
        inception = InceptionFeatures(dev)
        cache = os.path.splitext(a.data)[0] + f"_fidstats_{a.fid_real}.npz"
        mu_r, sigma_r = real_stats(a.data, inception, a.fid_real, cache=cache)
    div_real = diversity_score(ds.sample(64, seed=a.seed))
    print(f"Độ đa dạng ảnh thật (mốc): {div_real:.4f}")

    # ---- huấn luyện ----
    for epoch in range(start_epoch, a.epochs + 1):
        G.train(); D.train()
        t0 = time.time()
        sums = dict(loss_d=0.0, loss_g=0.0, d_x=0.0, d_gz=0.0)
        iter_rows = []
        for real in tqdm(dl, desc=f"epoch {epoch}/{a.epochs}", leave=False):
            real = real.to(dev, non_blocking=True)
            b = real.size(0)
            lab_real = torch.full((b,), a.real_label, device=dev)
            lab_fake = torch.zeros(b, device=dev)

            # (1) Cập nhật Discriminator
            D.zero_grad(set_to_none=True)
            out_real = D(real)
            loss_real = criterion(out_real, lab_real)
            z = torch.randn(b, a.nz, 1, 1, device=dev)
            fake = G(z)
            out_fake = D(fake.detach())
            loss_fake = criterion(out_fake, lab_fake)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optD.step()

            # (2) Cập nhật Generator: đánh lừa D (nhãn giả coi như thật)
            G.zero_grad(set_to_none=True)
            out_g = D(fake)
            loss_g = criterion(out_g, torch.ones(b, device=dev))
            loss_g.backward()
            optG.step()

            global_iter += 1
            ld, lg = loss_d.item(), loss_g.item()
            sums["loss_d"] += ld; sums["loss_g"] += lg
            sums["d_x"] += out_real.mean().item(); sums["d_gz"] += out_fake.mean().item()
            iter_rows.append({"iter": global_iter, "loss_d": ld, "loss_g": lg})

        n = len(dl)
        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise)
        save_grid(samples, os.path.join(a.out, "samples", f"epoch_{epoch:03d}.png"))
        div = diversity_score(samples)

        fid = float("nan")
        if inception is not None and (epoch % a.fid_every == 0 or epoch == a.epochs):
            fid = fid_of_generator(G, mu_r, sigma_r, inception, a.fid_samples, dev)
            if fid < best_fid:
                best_fid = fid
                torch.save({"G": G.state_dict(), "args": vars(a), "epoch": epoch, "fid": fid},
                           os.path.join(a.out, "best_fid.pt"))

        row = {"epoch": epoch, "loss_d": sums["loss_d"] / n, "loss_g": sums["loss_g"] / n,
               "d_x": sums["d_x"] / n, "d_gz": sums["d_gz"] / n, "diversity": div,
               "diversity_real": div_real, "fid": fid, "time_s": time.time() - t0}
        append_csv(os.path.join(a.out, "metrics.csv"), METRIC_FIELDS, [row])
        append_csv(os.path.join(a.out, "iter_losses.csv"), ["iter", "loss_d", "loss_g"], iter_rows)
        print(f"[{epoch:03d}/{a.epochs}] loss_D={row['loss_d']:.3f} loss_G={row['loss_g']:.3f} "
              f"D(x)={row['d_x']:.3f} D(G(z))={row['d_gz']:.3f} div={div:.3f} fid={fid:.2f} "
              f"({row['time_s']:.0f}s)")

        ck = {"G": G.state_dict(), "D": D.state_dict(), "optG": optG.state_dict(), "optD": optD.state_dict(),
              "epoch": epoch, "best_fid": best_fid, "global_iter": global_iter,
              "fixed_noise": fixed_noise.cpu(), "args": vars(a)}
        torch.save(ck, last_path)
        if epoch % 5 == 0 or epoch == a.epochs:
            torch.save({"G": G.state_dict(), "args": vars(a), "epoch": epoch},
                       os.path.join(a.out, "checkpoints", f"G_epoch_{epoch:03d}.pt"))
        plot_training(a.out)

    print("Hoàn tất huấn luyện. Kết quả ở:", a.out)


if __name__ == "__main__":
    main()
