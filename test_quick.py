"""
test_quick.py - Script kiem tra nhanh toan bo he thong sinh anh khuon mat GANs.

Cach chay:
    python test_quick.py
"""

import io
import os
import sys
import time
import base64
import torch
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from core.models import HybridFaceSynthesizer
from core.dataset import CelebAZipDataset


def run_quick_check() -> bool:
    print("==================================================================")
    print("   AI FACE GENERATOR STUDIO (DCGAN & cDCGAN) - KIEM TRA HE THONG ")
    print("   Do an Tri tue Nhan tao - Nhom DHN                              ")
    print("==================================================================\n")

    t0 = time.time()

    # 1. Kiem tra phan cung
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU Host"
    print(f"[1/5] Phan cung tinh toan: {device.type.upper()} ({dev_name})")

    # 2. Kiem tra tap tin checkpoints
    bundle_path = os.path.join("checkpoints", "celeba_gan_bundle.pth")
    yunet_path = os.path.join("checkpoints", "face_detection_yunet_2023mar.onnx")

    if not os.path.isfile(bundle_path):
        print(f"[-] Khong tim thay {bundle_path}!")
        return False
    if not os.path.isfile(yunet_path):
        print(f"[-] Khong tim thay {yunet_path}!")
        return False

    ck = torch.load(bundle_path, map_location="cpu")
    meta = ck.get("meta", {})
    epochs = meta.get("epochs_trained", "N/A")
    fid_dcgan = meta.get("best_fid_dcgan", "N/A")
    fid_cdcgan = meta.get("best_fid_cdcgan", "N/A")
    print(f"[2/5] Trong so GANs Bundle: Epochs={epochs} | Best FID DCGAN={fid_dcgan} | cDCGAN={fid_cdcgan}")
    print(f"      Mo hinh YuNet Landmark ONNX: OK")

    # 3. Khoi tao Synthesizer
    dataset = CelebAZipDataset(zip_path="archive.zip", cache_dir="data_cache", max_indexed=100)
    synthesizer = HybridFaceSynthesizer(dataset=dataset, checkpoints_dir="checkpoints")
    print(f"[3/5] Khoi tao HybridFaceSynthesizer thanh cong")

    # 4. Sinh thu nghiem anh tu cDCGAN va DCGAN
    print(f"[4/5] Sinh thu nghiem 2 anh cDCGAN (Smiling=1) va 2 anh DCGAN...")
    faces_c = synthesizer.generate_faces(
        num_images=2,
        model_type="cdcgan",
        desired_attrs={"Smiling": 1},
        truncation_psi=0.75,
        seed=101,
        output_size=160,
    )
    faces_d = synthesizer.generate_faces(
        num_images=2,
        model_type="dcgan",
        truncation_psi=0.75,
        seed=202,
        output_size=160,
    )
    assert len(faces_c) == 2 and len(faces_d) == 2, "Loi so luong anh sinh ra!"
    print(f"      Sinh cDCGAN: OK (D_score: {faces_c[0].get('discriminator_score'):.3f})")
    print(f"      Sinh DCGAN:  OK (D_score: {faces_d[0].get('discriminator_score'):.3f})")

    # 5. Kiem tra thuat toan lai ghep 2 khuon mat (YuNet + Laplacian Pyramid + Slerp)
    print(f"[5/5] Kiem tra tinh nang lai ghep 2 khuon mat (Face Fusion)...")
    img_a = faces_c[0]["image"]
    img_b = faces_d[0]["image"]

    fusion_res = synthesizer.blend_two_faces(img_a, img_b, alpha=0.5, num_steps=5, output_size=160)
    assert "main_image" in fusion_res, "Ket qua blend thieu main_image"
    assert "interpolation_strip" in fusion_res, "Thieu dai noi suy transition"
    print(f"      Lai ghep 2 anh va noi suy dai chuyen tiep: HOAN HAO")

    elapsed = time.time() - t0
    print("\n------------------------------------------------------------------")
    print(f" [OK] HE THONG DA DUOC KIEM TRA HOAN TAT VA SAN SANG HOAT DONG!   ")
    print(f" Thoi gian kiem tra: {elapsed:.2f}s                                ")
    print(f" Lenh chay ung dung: python app.py -> http://127.0.0.1:5000       ")
    print("------------------------------------------------------------------\n")
    return True


if __name__ == "__main__":
    ok = run_quick_check()
    sys.exit(0 if ok else 1)
