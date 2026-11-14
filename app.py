import os
import sys
import io
import base64
import random
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from PIL import Image, ImageDraw
from flask import Flask, render_template, request, jsonify

from core.dataset import CelebAZipDataset, SELECTED_ATTRIBUTES, ATTR_KEYS, ATTR_LABELS
from core.models import HybridFaceSynthesizer
from core.fid_metrics import FIDEvaluator
from core.trainer import GANTrainerEngine

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # Cho phép upload ảnh tối đa 25MB

# Khởi tạo Dataset, Mô hình GAN, Bộ tính FID và Bộ huấn luyện
print("==================================================================")
print(" KHỞI ĐỘNG HỆ THỐNG SINH ẢNH KHUÔN MẶT BẰNG GANs (CELEBA DATASET) ")
print("==================================================================")
dataset = CelebAZipDataset(zip_path="archive.zip", cache_dir="data_cache", max_indexed=300)
dataset.ensure_cached_samples(count=200)

synthesizer = HybridFaceSynthesizer(dataset=dataset, checkpoints_dir="checkpoints")
trainer = GANTrainerEngine(dataset=dataset, synthesizer=synthesizer)
trainer.ensure_initial_pretrained_weights(num_Bootstrap_images=48)
fid_evaluator = FIDEvaluator(dataset=dataset, synthesizer=synthesizer)


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def base64_to_pil(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    if max(w, h) > 768:
        scale = 768.0 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def build_image_grid(images: list, cols: int = 4, pad: int = 8, bg_color=(15, 23, 42)) -> Image.Image:
    if not images:
        return Image.new("RGB", (256, 256), bg_color)
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid_w = cols * w + (cols + 1) * pad
    grid_h = rows * h + (rows + 1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), bg_color)
    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        grid.paste(im, (x, y))
    return grid


@app.route("/")
def index():
    return render_template("index.html", attributes=SELECTED_ATTRIBUTES)


@app.route("/api/dataset/summary", methods=["GET"])
def api_dataset_summary():
    summary = dataset.get_dataset_summary()
    return jsonify({"ok": True, "summary": summary})


@app.route("/api/dataset/samples", methods=["GET"])
def api_dataset_samples():
    limit = int(request.args.get("limit", 18))
    attr_filter = request.args.get("attr", "")
    randomize = request.args.get("random", "1") == "1"

    desired = {}
    if attr_filter and attr_filter in ATTR_KEYS:
        desired[attr_filter] = 1

    matched = dataset.query_by_attributes(desired, limit=limit, randomize=randomize)
    items = []
    for img_id, _ in matched:
        pil_img = dataset.get_image_pil(img_id, size=(144, 144))
        rec_attrs = dataset.records.get(img_id, {}).get("attrs", {})
        active_tags = [ATTR_LABELS[k].split(" (")[0] for k in ATTR_KEYS if rec_attrs.get(k, 0) == 1]
        items.append(
            {
                "image_id": img_id,
                "image_b64": pil_to_base64(pil_img, fmt="JPEG"),
                "attributes": active_tags[:4],
                "attr_map": rec_attrs,
            }
        )
    return jsonify({"ok": True, "samples": items})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True) or {}
    model_type = data.get("model_type", "cdcgan")
    num_images = int(data.get("num_images", 8))
    num_images = max(1, min(16, num_images))
    truncation_psi = float(data.get("truncation_psi", 0.75))
    seed = data.get("seed", None)
    if seed is not None and str(seed).strip() != "" and int(seed) >= 0:
        seed = int(seed)
    else:
        seed = random.randint(1000, 999999)

    raw_attrs = data.get("attributes", {})
    desired_attrs = {}
    for k in ATTR_KEYS:
        v = raw_attrs.get(k, None)
        if v in (0, 1, "0", "1"):
            desired_attrs[k] = int(v)

    generated = synthesizer.generate_faces(
        num_images=num_images,
        model_type=model_type,
        desired_attrs=desired_attrs,
        truncation_psi=truncation_psi,
        seed=seed,
        output_size=160,
    )

    pil_list = [item["image"] for item in generated]
    cols = 4 if num_images >= 4 else num_images
    grid_img = build_image_grid(pil_list, cols=cols, pad=10)

    response_items = []
    for item in generated:
        response_items.append(
            {
                "index": item["index"],
                "seed": item["seed"],
                "image_b64": pil_to_base64(item["image"]),
                "latent_norm": item["latent_norm"],
                "discriminator_score": item["discriminator_score"],
                "attr_match_score": item["attr_match_score"],
                "attributes": item["attributes"],
                "latent_preview": item["latent_preview"],
            }
        )

    return jsonify(
        {
            "ok": True,
            "seed_used": seed,
            "model_type": model_type,
            "truncation_psi": truncation_psi,
            "grid_image_b64": pil_to_base64(grid_img),
            "results": response_items,
        }
    )


@app.route("/api/blend", methods=["POST"])
def api_blend():
    data = request.get_json(force=True) or {}
    source_a_type = data.get("source_a_type", "celeba")
    source_b_type = data.get("source_b_type", "celeba")
    alpha = float(data.get("alpha", 0.5))
    color_harmony = float(data.get("color_harmony", 0.65))
    num_steps = int(data.get("num_steps", 7))
    output_size = 192

    # Đọc Ảnh A + Landmarks A
    lm_a = None
    if source_a_type == "upload" and data.get("image_b64_a"):
        img_a = base64_to_pil(data["image_b64_a"])
        label_a = "Ảnh tải lên A"
    else:
        id_a = data.get("image_id_a") or "000001.jpg"
        if id_a not in dataset.records:
            id_a = dataset.image_ids[0]
        img_a = dataset.get_image_pil(id_a, size=(output_size, output_size))
        lm_a = dataset.get_landmarks_cropped(id_a, target_size=output_size)
        label_a = f"CelebA #{id_a}"

    # Đọc Ảnh B + Landmarks B
    lm_b = None
    if source_b_type == "upload" and data.get("image_b64_b"):
        img_b = base64_to_pil(data["image_b64_b"])
        label_b = "Ảnh tải lên B"
    else:
        id_b = data.get("image_id_b") or "000002.jpg"
        if id_b not in dataset.records:
            id_b = dataset.image_ids[1]
        img_b = dataset.get_image_pil(id_b, size=(output_size, output_size))
        lm_b = dataset.get_landmarks_cropped(id_b, target_size=output_size)
        label_b = f"CelebA #{id_b}"

    blend_res = synthesizer.blend_two_faces(
        img_a_pil=img_a,
        img_b_pil=img_b,
        alpha=alpha,
        lm_a_5=lm_a,
        lm_b_5=lm_b,
        color_harmony=color_harmony,
        num_steps=num_steps,
        output_size=output_size,
    )

    strip_pil_list = [step["image"] for step in blend_res["interpolation_strip"]]
    strip_banner = build_image_grid(strip_pil_list, cols=len(strip_pil_list), pad=8)

    strip_serialized = [
        {
            "alpha": step["alpha"],
            "pct_a": 100 - step["pct_b"],
            "pct_b": step["pct_b"],
            "image_b64": pil_to_base64(step["image"]),
        }
        for step in blend_res["interpolation_strip"]
    ]

    attr_comparison = []
    for k in ATTR_KEYS:
        attr_comparison.append(
            {
                "key": k,
                "label": ATTR_LABELS[k].split(" (")[0],
                "prob_a": round(blend_res["attrs_a"][k] * 100, 1),
                "prob_b": round(blend_res["attrs_b"][k] * 100, 1),
                "prob_blended": round(blend_res["attrs_blended"][k] * 100, 1),
            }
        )

    return jsonify(
        {
            "ok": True,
            "label_a": label_a,
            "label_b": label_b,
            "alpha": blend_res["alpha"],
            "pct_a": int(round((1.0 - blend_res["alpha"]) * 100)),
            "pct_b": int(round(blend_res["alpha"] * 100)),
            "image_a_b64": pil_to_base64(blend_res["img_a_aligned"]),
            "image_b_b64": pil_to_base64(blend_res["img_b_aligned"]),
            "blended_image_b64": pil_to_base64(blend_res["main_image"]),
            "strip_banner_b64": pil_to_base64(strip_banner),
            "interpolation_strip": strip_serialized,
            "latent_cosine_similarity": blend_res["latent_cosine_similarity"],
            "latent_l2_distance": blend_res["latent_l2_distance"],
            "latent_z_blended": blend_res["latent_z_blended"],
            "attribute_comparison": attr_comparison,
        }
    )


@app.route("/api/evaluate_fid", methods=["POST"])
def api_evaluate_fid():
    data = request.get_json(force=True) or {}
    model_type = data.get("model_type", "cdcgan")
    sample_size = int(data.get("sample_size", 64))
    truncation_psi = float(data.get("truncation_psi", 0.75))
    seed = int(data.get("seed", 2026))

    eval_res = fid_evaluator.evaluate_model(
        model_type=model_type,
        sample_size=sample_size,
        truncation_psi=truncation_psi,
        seed=seed,
    )

    real_b64 = [pil_to_base64(im, fmt="JPEG") for im in eval_res.pop("real_samples_preview")]
    fake_b64 = [pil_to_base64(im, fmt="PNG") for im in eval_res.pop("fake_samples_preview")]
    eval_res["real_samples_b64"] = real_b64
    eval_res["fake_samples_b64"] = fake_b64

    return jsonify({"ok": True, "evaluation": eval_res})


@app.route("/api/train/status", methods=["GET"])
def api_train_status():
    return jsonify(
        {
            "ok": True,
            "status": trainer.status,
            "history": trainer.history[-40:],
            "checkpoint_meta": synthesizer.training_meta,
        }
    )


@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    data = request.get_json(force=True) or {}
    model_type = data.get("model_type", "cdcgan")
    epochs = int(data.get("epochs", 5))
    batch_size = int(data.get("batch_size", 16))
    lr = float(data.get("lr", 0.0002))
    num_samples = int(data.get("num_samples", 64))

    res = trainer.start_interactive_training(
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_samples=num_samples,
    )
    return jsonify(res)


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    res = trainer.stop_training()
    return jsonify(res)


@app.route("/api/checkpoint/reload", methods=["POST"])
def api_checkpoint_reload():
    synthesizer._load_or_init_checkpoints()
    return jsonify(
        {
            "ok": True,
            "message": "Đã nạp lại file trọng số checkpoints/celeba_gan_bundle.pth thành công!",
            "meta": synthesizer.training_meta,
        }
    )


@app.route("/api/checkpoint/upload", methods=["POST"])
def api_checkpoint_upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "message": "Chưa chọn file .pth!"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "message": "Tên file không hợp lệ!"}), 400

    ckpt_path = os.path.join(synthesizer.checkpoints_dir, "celeba_gan_bundle.pth")
    f.save(ckpt_path)
    synthesizer._load_or_init_checkpoints()
    return jsonify(
        {
            "ok": True,
            "message": f"Đã nạp thành công trọng số Kaggle ({f.filename}) vào hệ thống Web!",
            "meta": synthesizer.training_meta,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
