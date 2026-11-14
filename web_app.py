"""
web_app.py - Ứng dụng Web Full-Stack Modern UI cho Mô hình DCGAN / cDCGAN.

Tác giả: Senior AI Engineer & Full-stack Developer
Kiến trúc: Flask REST API + Modern Tailwind Glassmorphism Frontend.
Ưu điểm vượt trội so với Streamlit:
  - Thiết kế UI/UX theo phong cách AI Studio cao cấp (Dark Glassmorphism, Gradient Glow, Animations).
  - Phản hồi tức thì qua REST API (không bị giật hay reload lại toàn trang như Streamlit).
  - Hỗ trợ xem ảnh Lightbox toàn màn hình, tải ảnh PNG và xuất video GIF biến hình.
"""

import base64
import io
import os
import random
import time
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request, send_file
import torch

from dataset import SELECTED_ATTRS
from models import ConditionalGenerator
from utils import (
    ensure_utf8_stdio,
    extract_checkpoint_info,
    generate_morph_gif_bytes,
    slerp,
    tensor_to_pil,
)

ensure_utf8_stdio()

app = Flask(__name__, template_folder="templates", static_folder="static")

NZ = 100
CANDIDATE_MODEL_PATHS = [
    "generator.pth",
    "best_fid.pt",
    "last.pt",
    os.path.join("outputs", "run1", "best_fid.pt"),
    os.path.join("outputs", "run1", "last.pt"),
]

# Tự động detect phần cứng
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
)
print(f"[*] Hardware device: {device}")

# Nạp mô hình
netG: Optional[ConditionalGenerator] = None
model_meta: Dict[str, Any] = {}


def init_model() -> None:
    """Nạp trọng số Generator đầu tiên tìm thấy hoặc khởi tạo bản demo."""
    global netG, model_meta
    chosen_path = next((p for p in CANDIDATE_MODEL_PATHS if os.path.isfile(p)), None)

    if chosen_path:
        try:
            ck = torch.load(chosen_path, map_location=device)
            state_dict, num_attrs, meta = extract_checkpoint_info(ck, nz=NZ)
            nz_val = int(meta.get("nz", NZ))
            ngf_val = int(meta.get("ngf", 64))

            netG = ConditionalGenerator(nz=nz_val, num_attrs=num_attrs, ngf=ngf_val, nc=3).to(device)
            netG.load_state_dict(state_dict, strict=True)
            netG.eval()

            fid_val = round(float(meta["fid"]), 2) if meta.get("fid") is not None else "N/A"
            epoch_val = int(meta["epoch"]) if meta.get("epoch") is not None else "N/A"
            params_m = sum(p.numel() for p in netG.parameters()) / 1e6

            model_meta = {
                "name": os.path.basename(chosen_path),
                "fid": fid_val,
                "epoch": epoch_val,
                "params": f"{params_m:.2f}M",
                "num_attrs": num_attrs,
                "attrs": SELECTED_ATTRS[:num_attrs],
                "type": "cDCGAN (Conditional)" if num_attrs > 0 else "DCGAN (CelebA Benchmark)",
                "status": "ready",
            }
            print(f"[+] Loaded model '{chosen_path}' (FID: {model_meta['fid']}, Epoch: {model_meta['epoch']}, attrs: {num_attrs})")
        except Exception as e:
            print(f"[-] Load model error: {e}")
            fallback_demo()
    else:
        fallback_demo()


def fallback_demo() -> None:
    """Khởi tạo Generator ngẫu nhiên khi chưa có file trọng số hợp lệ."""
    global netG, model_meta
    netG = ConditionalGenerator(nz=NZ, num_attrs=0, ngf=64, nc=3).to(device)
    netG.eval()
    model_meta = {
        "name": "Demo Generator (Khởi tạo ngẫu nhiên)",
        "fid": "N/A",
        "epoch": 0,
        "params": "3.58M",
        "num_attrs": 0,
        "attrs": [],
        "type": "DCGAN Demo",
        "status": "demo",
    }


init_model()


def _parse_seed(raw_seed: Any, default_random: bool = True) -> int:
    if raw_seed is not None and str(raw_seed).strip() != "":
        try:
            return int(raw_seed)
        except (TypeError, ValueError):
            pass
    return random.randint(1, 999999) if default_random else 120


def _to_attr_flag(val: Any) -> float:
    if val is None:
        return 0.0
    try:
        return 1.0 if float(val) > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _parse_conditions(data: Dict[str, Any]) -> Optional[torch.Tensor]:
    if netG is None or netG.num_attrs <= 0:
        return None
    raw_attrs = data.get("attrs")
    attr_names = SELECTED_ATTRS[: netG.num_attrs]
    if isinstance(raw_attrs, dict):
        vec = [_to_attr_flag(raw_attrs.get(name, 0)) for name in attr_names]
    elif isinstance(raw_attrs, (list, tuple)):
        vec = [_to_attr_flag(raw_attrs[i]) if i < len(raw_attrs) else 0.0 for i in range(netG.num_attrs)]
    else:
        vec = [0.0] * netG.num_attrs
    return torch.tensor([vec], dtype=torch.float32, device=device)


def _encode_pil_data_url(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_b64}"


@app.route("/")
def index():
    return render_template("index.html", meta=model_meta)


@app.route("/api/model_info")
def get_model_info():
    return jsonify(model_meta)


@app.route("/api/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "device": str(device),
        "model_loaded": netG is not None,
        "model_name": model_meta.get("name", "none"),
        "type": model_meta.get("type", "unknown"),
    })


@app.route("/api/generate", methods=["POST"])
def generate_images():
    """Sinh n khuôn mặt từ vector nhiễu (và điều kiện nếu mô hình là cDCGAN)."""
    data = request.get_json(silent=True) or {}
    try:
        count = int(data.get("count", 4))
        variance = float(data.get("variance", 0.75))
        upscale_size = int(data.get("size", 256))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Tham số đầu vào không hợp lệ."}), 400

    if count < 1 or count > 16:
        return jsonify({"success": False, "error": "count phải nằm trong khoảng 1..16."}), 400
    upscale_size = max(64, min(512, upscale_size))
    variance = max(0.05, min(2.0, variance))
    base_seed = _parse_seed(data.get("seed"))
    cond = _parse_conditions(data)
    t0 = time.time()

    results: List[Dict[str, Any]] = []
    with torch.no_grad():
        for i in range(count):
            cur_seed = base_seed + i
            torch.manual_seed(cur_seed)
            z = torch.randn(1, netG.nz, device=device) * variance
            img_tensor = netG(z, cond)[0] if cond is not None else netG(z)[0]
            pil_img = tensor_to_pil(img_tensor, upscale_size=upscale_size)
            results.append({
                "seed": cur_seed,
                "data_url": _encode_pil_data_url(pil_img),
            })

    return jsonify({"success": True, "base_seed": base_seed, "images": results, "elapsed_ms": round((time.time() - t0) * 1000, 1)})


@app.route("/api/morph", methods=["POST"])
def morph_face():
    """Nội suy giữa Face A và Face B tại một giá trị alpha cụ thể."""
    data = request.get_json(silent=True) or {}
    try:
        seed_a = int(data.get("seed_a", 120))
        seed_b = int(data.get("seed_b", 350))
        alpha = max(0.0, min(1.0, float(data.get("alpha", 0.5))))
        variance = max(0.05, min(2.0, float(data.get("variance", 0.75))))
        upscale_size = max(64, min(512, int(data.get("size", 256))))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Tham số morph không hợp lệ."}), 400

    cond = _parse_conditions(data)
    t0 = time.time()
    torch.manual_seed(seed_a)
    z_a = torch.randn(1, netG.nz, device=device) * variance
    torch.manual_seed(seed_b)
    z_b = torch.randn(1, netG.nz, device=device) * variance
    z_mid = slerp(z_a, z_b, alpha)

    with torch.no_grad():
        img_tensor = netG(z_mid, cond)[0] if cond is not None else netG(z_mid)[0]
        pil_img = tensor_to_pil(img_tensor, upscale_size=upscale_size)

    return jsonify({
        "success": True,
        "alpha": alpha,
        "data_url": _encode_pil_data_url(pil_img),
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    })


@app.route("/api/create_gif", methods=["POST"])
def create_gif():
    """Tạo video GIF biến hình hoàn chỉnh để tải về máy."""
    data = request.get_json(silent=True) or {}
    try:
        seed_a = int(data.get("seed_a", 120))
        seed_b = int(data.get("seed_b", 350))
        steps = int(data.get("steps", 16))
        variance = max(0.05, min(2.0, float(data.get("variance", 0.75))))
        upscale_size = max(64, min(512, int(data.get("size", 256))))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Tham số create_gif không hợp lệ."}), 400

    if steps < 1 or steps > 60:
        return jsonify({"success": False, "error": "steps phải nằm trong khoảng 1..60."}), 400

    cond = _parse_conditions(data)
    t0 = time.time()
    torch.manual_seed(seed_a)
    z_a = torch.randn(1, netG.nz, device=device) * variance
    torch.manual_seed(seed_b)
    z_b = torch.randn(1, netG.nz, device=device) * variance

    gif_bytes = generate_morph_gif_bytes(
        netG,
        z_a,
        z_b,
        conditions=cond,
        steps=steps,
        duration=80,
        upscale_size=upscale_size,
    )
    buf = io.BytesIO(gif_bytes)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="image/gif",
        as_attachment=True,
        download_name=f"morphing_face_{seed_a}_to_{seed_b}.gif",
    )


if __name__ == "__main__":
    print("\n=======================================================")
    print("   AI FACE GENERATOR STUDIO - MODERN WEB SERVER")
    print("   Truy cap Web tai: http://localhost:5000")
    print("=======================================================\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
