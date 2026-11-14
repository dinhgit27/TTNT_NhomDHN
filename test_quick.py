"""
test_quick.py - Script kiem tra nhanh toan bo he thong truoc khi demo / bao cao.

Cach chay:
    python test_quick.py
"""

import os
import sys
import time
import torch

from models import ConditionalGenerator
from utils import (
    ensure_utf8_stdio,
    extract_checkpoint_info,
    save_image_grid,
    slerp,
)


def run_quick_check() -> bool:
    ensure_utf8_stdio()
    print("========================================================")
    print("   AI FACE GENERATOR STUDIO - HE THONG KIEM TRA NHANH")
    print("   Do an Tri tue Nhan tao - Nhom DHN")
    print("========================================================\n")

    # 1. Kiem tra phan cung
    t0 = time.time()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU Host"
    print(f"[1/4] Thiet bi tinh toan: {device.type.upper()} ({dev_name})")

    # 2. Nap trong so mo hinh
    ckpt_path = "generator.pth"
    if not os.path.isfile(ckpt_path):
        print(f"[-] Khong tim thay file {ckpt_path}!")
        return False

    ck = torch.load(ckpt_path, map_location=device)
    state_dict, num_attrs, meta = extract_checkpoint_info(ck)
    nz = int(meta.get("nz", 100))
    ngf = int(meta.get("ngf", 64))

    netG = ConditionalGenerator(nz=nz, num_attrs=num_attrs, ngf=ngf, nc=3).to(device)
    netG.load_state_dict(state_dict, strict=True)
    netG.eval()
    params_m = sum(p.numel() for p in netG.parameters()) / 1e6
    epoch_val = meta.get("epoch", "N/A")
    fid_val = round(float(meta["fid"]), 2) if meta.get("fid") is not None else "N/A"
    print(f"[2/4] Nap thanh cong '{ckpt_path}' | Epoch: {epoch_val} | FID: {fid_val} | Tham so: {params_m:.2f}M")

    # 3. Sinh thu nghiem 4 khuon mat
    with torch.no_grad():
        z = netG.sample_noise(4, device=device) * 0.75
        imgs = netG(z)
        assert imgs.shape == (4, 3, 64, 64), f"Kich thuoc anh sai: {imgs.shape}"

    os.makedirs("outputs", exist_ok=True)
    out_img = os.path.join("outputs", "quick_check.png")
    save_image_grid(imgs.cpu(), out_img, nrow=2)
    print(f"[3/4] Sinh thanh cong 4 anh mau -> da luu vao '{out_img}'")

    # 4. Kiem tra noi suy khong gian tiem an (Slerp)
    with torch.no_grad():
        z0 = torch.randn(1, nz, device=device)
        z1 = torch.randn(1, nz, device=device)
        z_mid = slerp(z0, z1, 0.5)
        img_mid = netG(z_mid)
        assert img_mid.shape == (1, 3, 64, 64)
    print(f"[4/4] Noi suy Slerp latent space hoat dong chuan xac!")

    elapsed = time.time() - t0
    print("\n--------------------------------------------------------")
    print(f"[OK] TOAN BO HE THONG SAN SANG CHO DEMO ({elapsed:.2f}s)!")
    print("--------------------------------------------------------\n")
    return True


if __name__ == "__main__":
    success = run_quick_check()
    sys.exit(0 if success else 1)
