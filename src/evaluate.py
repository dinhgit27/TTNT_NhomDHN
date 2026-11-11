"""Giai đoạn 4 - Module đánh giá mô hình đã huấn luyện (hỗ trợ cả DCGAN và cDCGAN).

Sinh ra trong thư mục ``--out``:
  * ``generated_grid.png``    : lưới 64 khuôn mặt mới từ nhiễu ngẫu nhiên
  * ``real_grid.png``         : lưới 64 ảnh thật để đối chiếu
  * ``interpolation.png``     : nội suy (slerp) trong latent space -> không gian tiềm ẩn liên tục
  * ``nearest_neighbors.png`` : ảnh sinh (hàng trên) và ảnh thật gần nhất (hàng dưới) -> kiểm tra
                                 mô hình không "chép" lại ảnh huấn luyện
  * ``generated/``            : N ảnh sinh riêng lẻ (tùy chọn ``--save-images``)
  * ``eval_report.json``      : FID + chỉ số đa dạng (ảnh sinh vs ảnh thật)

Ví dụ::

    python -m src.evaluate --ckpt generator.pth --data data/processed/celeba_64.npy --out outputs/eval
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from .dataset import CelebA64, to_unit_range
from .fid import InceptionFeatures, frechet_distance, generator_images, real_stats, stats_from_activations, get_activations
from .models import ConditionalGenerator
from .utils import diversity_score, ensure_utf8_stdio, extract_checkpoint_info, save_grid, slerp


def load_generator(ckpt_path: str, device) -> tuple[ConditionalGenerator, int | None]:
    ck = torch.load(ckpt_path, map_location=device)
    state_dict, num_attrs, meta = extract_checkpoint_info(ck)
    G = ConditionalGenerator(
        nz=int(meta.get("nz", 100)),
        num_attrs=num_attrs,
        ngf=int(meta.get("ngf", 64)),
        nc=3,
    ).to(device)
    G.load_state_dict(state_dict, strict=True)
    G.eval()
    return G, meta.get("epoch")


def _sample_from_generator(G: ConditionalGenerator, n: int, device) -> torch.Tensor:
    z = G.sample_noise(n, device)
    if G.num_attrs > 0:
        cond = torch.randint(0, 2, (n, G.num_attrs), dtype=torch.float32, device=device)
        return G(z, cond)
    return G(z)


@torch.no_grad()
def interpolation_grid(G: ConditionalGenerator, device, rows: int = 8, steps: int = 10, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device).manual_seed(seed)
    out = []
    for _ in range(rows):
        z0 = torch.randn(1, G.nz, 1, 1, device=device, generator=g)
        z1 = torch.randn(1, G.nz, 1, 1, device=device, generator=g)
        zs = torch.cat([slerp(z0[0], z1[0], t / (steps - 1)).unsqueeze(0) for t in range(steps)])
        if G.num_attrs > 0:
            c = torch.randint(0, 2, (1, G.num_attrs), dtype=torch.float32, device=device, generator=g).repeat(steps, 1)
            out.append(G(zs, c).cpu())
        else:
            out.append(G(zs).cpu())
    return torch.cat(out)


@torch.no_grad()
def nearest_real(fake: torch.Tensor, real_pool: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(fake.flatten(1), real_pool.flatten(1))
    return real_pool[d.argmin(dim=1)]


def main() -> None:
    ensure_utf8_stdio()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="generator.pth / best_fid.pt / last.pt / checkpoints/G_epoch_XXX.pt")
    p.add_argument("--data", default="data/processed/celeba_64.npy")
    p.add_argument("--out", default="outputs/eval")
    p.add_argument("--n-fid", type=int, default=10000, help="số ảnh sinh dùng tính FID")
    p.add_argument("--n-real", type=int, default=10000, help="số ảnh thật dùng tính FID")
    p.add_argument("--save-images", type=int, default=0, help="lưu N ảnh sinh riêng lẻ ra generated/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    G, epoch_val = load_generator(a.ckpt, dev)
    ds = CelebA64(a.data, flip=False)

    # --- định tính ---
    with torch.no_grad():
        fake64 = _sample_from_generator(G, 64, dev).cpu()
    save_grid(fake64, os.path.join(a.out, "generated_grid.png"))
    save_grid(ds.sample(64, seed=a.seed), os.path.join(a.out, "real_grid.png"))
    interp = interpolation_grid(G, dev, seed=a.seed)
    save_grid(interp, os.path.join(a.out, "interpolation.png"), nrow=10)
    pool = ds.sample(20000, seed=a.seed + 1)
    nn_real = nearest_real(fake64[:10], pool)
    save_grid(torch.cat([fake64[:10], nn_real]), os.path.join(a.out, "nearest_neighbors.png"), nrow=10)

    # --- chống mode collapse: độ đa dạng ---
    with torch.no_grad():
        fake_div = _sample_from_generator(G, 500, dev).cpu()
    div_fake = diversity_score(fake_div)
    div_real = diversity_score(ds.sample(500, seed=a.seed))

    # --- định lượng: FID ---
    inception = InceptionFeatures(dev)
    mu_r, s_r = real_stats(a.data, inception, a.n_real,
                           cache=os.path.splitext(a.data)[0] + f"_fidstats_{a.n_real}.npz")
    act = get_activations(generator_images(G, a.n_fid, dev), inception)
    mu_g, s_g = stats_from_activations(act)
    fid = frechet_distance(mu_r, s_r, mu_g, s_g)

    if a.save_images > 0:
        gdir = os.path.join(a.out, "generated")
        os.makedirs(gdir, exist_ok=True)
        i = 0
        for batch in generator_images(G, a.save_images, dev):
            for img in to_unit_range(batch):
                arr = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                Image.fromarray(arr).save(os.path.join(gdir, f"{i:05d}.png"))
                i += 1

    report = {
        "checkpoint": a.ckpt,
        "epoch": epoch_val,
        "num_attrs": G.num_attrs,
        "fid": fid,
        "n_fid_fake": a.n_fid,
        "n_fid_real": min(a.n_real, len(ds)),
        "diversity_fake": div_fake,
        "diversity_real": div_real,
        "diversity_ratio": (div_fake / div_real) if div_real > 0 else 0.0,
    }
    with open(os.path.join(a.out, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("Tỉ lệ đa dạng ≈ 1 → không mode collapse; ≪ 1 → G sinh ảnh rập khuôn.")


if __name__ == "__main__":
    main()
