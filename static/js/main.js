// ============================================================================
// GAN FACE STUDIO - FRONTEND INTERACTIVE ENGINE
// ============================================================================

const state = {
    attributes: {},
    blend: {
        sourceAType: "celeba",
        imageIdA: "000007.jpg",
        imageB64A: null,
        sourceBType: "celeba",
        imageIdB: "000018.jpg",
        imageB64B: null,
        alpha: 0.50,
        harmony: 0.65,
        pickerTarget: "A",
    },
    trainPollTimer: null,
};

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initGenerationTab();
    initBlendTab();
    initFIDTab();
    initTrainTab();
    initDatasetTab();

    // Tự động chạy mẫu lần đầu cho tất cả các Tab để giao diện luôn có dữ liệu trực quan
    runGenerateFaces();
    runBlendFaces();
    runEvaluateFID();
    fetchDatasetSummary();
    fetchDatasetSamples();
    pollTrainingStatus();
});

// ----------------------------------------------------------------------------
// 1. ĐIỀU HƯỚNG TAB
// ----------------------------------------------------------------------------
function initTabs() {
    const tabBtns = document.querySelectorAll("#mainNavTabs .nav-tab");
    const panels = document.querySelectorAll(".tab-panel");

    tabBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            tabBtns.forEach((b) => b.classList.remove("active"));
            panels.forEach((p) => p.classList.remove("active"));
            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            const targetPanel = document.getElementById(targetId);
            if (targetPanel) targetPanel.classList.add("active");
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    });
}

function switchToTab(tabId) {
    const btn = document.querySelector(`#mainNavTabs .nav-tab[data-tab="${tabId}"]`);
    if (btn) btn.click();
}

// ----------------------------------------------------------------------------
// 2. TAB 1: TỰ SINH ẢNH KHUÔN MẶT (DCGAN & CONDITIONAL DCGAN)
// ----------------------------------------------------------------------------
function initGenerationTab() {
    const countSlider = document.getElementById("genCount");
    const countVal = document.getElementById("genCountVal");
    countSlider.addEventListener("input", () => {
        countVal.textContent = `${countSlider.value} ảnh`;
    });

    const psiSlider = document.getElementById("genPsi");
    const psiVal = document.getElementById("genPsiVal");
    psiSlider.addEventListener("input", () => {
        psiVal.textContent = parseFloat(psiSlider.value).toFixed(2);
    });

    const modelSelect = document.getElementById("genModelType");
    if (modelSelect) {
        modelSelect.addEventListener("change", () => {
            runGenerateFaces();
        });
    }

    document.getElementById("btnRandomSeed").addEventListener("click", () => {
        document.getElementById("genSeed").value = Math.floor(10000 + Math.random() * 900000);
        runGenerateFaces();
    });

    document.getElementById("btnGenerateFaces").addEventListener("click", () => {
        document.getElementById("genSeed").value = Math.floor(10000 + Math.random() * 900000);
        runGenerateFaces();
    });
}

function setAttrUIState(itemEl, key, val) {
    const btns = itemEl.querySelectorAll(".attr-btn");
    btns.forEach((b) => {
        b.classList.remove("active-any", "active-pos", "active-neg");
        if (b.getAttribute("data-val") === val) {
            if (val === "1") b.classList.add("active-pos");
            else if (val === "0") b.classList.add("active-neg");
            else b.classList.add("active-any");
        }
    });
    itemEl.classList.remove("state-pos", "state-neg");
    if (val === "1") {
        itemEl.classList.add("state-pos");
        state.attributes[key] = 1;
    } else if (val === "0") {
        itemEl.classList.add("state-neg");
        state.attributes[key] = 0;
    } else {
        state.attributes[key] = null;
    }
}

function resetAllAttributes() {
    document.querySelectorAll("#attrControlGrid .attr-item").forEach((item) => {
        const key = item.getAttribute("data-attr-key");
        setAttrUIState(item, key, "any");
    });
}

async function runGenerateFaces() {
    const btn = document.getElementById("btnGenerateFaces");
    const subtitle = document.getElementById("genStatusSubtitle");
    const gallery = document.getElementById("generatedGallery");

    btn.disabled = true;
    btn.innerHTML = "<span>⏳ Đang chạy Generator G(z, c)...</span>";

    const payload = {
        model_type: document.getElementById("genModelType").value,
        num_images: parseInt(document.getElementById("genCount").value, 10),
        truncation_psi: parseFloat(document.getElementById("genPsi").value),
        seed: document.getElementById("genSeed").value,
        attributes: state.attributes,
    };

    try {
        const res = await fetch("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (data.ok) {
            subtitle.textContent = `Đã sinh ${data.results.length} ảnh mới • Mô hình: ${data.model_type.toUpperCase()} • Seed: ${data.seed_used} • Truncation ψ = ${data.truncation_psi}`;
            document.getElementById("btnDownloadGrid").href = data.grid_image_b64;

            gallery.innerHTML = "";
            data.results.forEach((item) => {
                const tagsHtml = item.attributes.map((t) => `<span class="tag-chip">${t}</span>`).join("");
                const card = document.createElement("div");
                card.className = "face-card";
                card.innerHTML = `
                    <div class="face-img-wrap">
                        <img src="${item.image_b64}" alt="Generated Face #${item.index}" />
                        <span class="face-badge">#${item.index} • Seed ${item.seed}</span>
                        <span class="face-score-badge">D(G(z)): ${item.discriminator_score}</span>
                    </div>
                    <div class="face-meta">
                        <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#94a3b8;">
                            <span>||z||₂ = ${item.latent_norm}</span>
                            <a href="${item.image_b64}" download="gan_face_${item.seed}.png" style="color:#38bdf8; text-decoration:none; font-weight:600;">⬇ Tải ảnh</a>
                        </div>
                        <div class="face-tags">${tagsHtml || '<span class="tag-chip">Ngẫu nhiên</span>'}</div>
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:4px; margin-top:8px;">
                            <button type="button" class="btn-secondary" style="font-size:0.68rem; padding:3px 4px; justify-content:center;" data-use-slot="A">Đặt làm Ảnh A</button>
                            <button type="button" class="btn-secondary" style="font-size:0.68rem; padding:3px 4px; justify-content:center;" data-use-slot="B">Đặt làm Ảnh B</button>
                        </div>
                    </div>
                `;
                card.querySelector('[data-use-slot="A"]').addEventListener("click", () => {
                    useGeneratedImageInBlend("A", item.image_b64, `GAN Seed #${item.seed}`);
                });
                card.querySelector('[data-use-slot="B"]').addEventListener("click", () => {
                    useGeneratedImageInBlend("B", item.image_b64, `GAN Seed #${item.seed}`);
                });
                gallery.appendChild(card);
            });
        }
    } catch (err) {
        console.error(err);
        subtitle.textContent = "Có lỗi khi kết nối tới máy chủ sinh ảnh.";
    } finally {
        btn.disabled = false;
        btn.innerHTML = "<span>✨ Sinh Ảnh Khuôn Mặt Mới</span>";
    }
}

function useGeneratedImageInBlend(slot, imgB64, label) {
    if (slot === "A") {
        state.blend.sourceAType = "upload";
        state.blend.imageB64A = imgB64;
        document.getElementById("labelSourceA").textContent = label;
    } else {
        state.blend.sourceBType = "upload";
        state.blend.imageB64B = imgB64;
        document.getElementById("labelSourceB").textContent = label;
    }
    switchToTab("tab-blend");
    runBlendFaces();
}

// ----------------------------------------------------------------------------
// 3. TAB 2: KẾT HỢP 2 ẢNH KHÁC NHAU ĐỂ SINH ẢNH MỚI (DUAL-IMAGE FACE FUSION)
// ----------------------------------------------------------------------------
function initBlendTab() {
    const alphaSlider = document.getElementById("blendAlphaSlider");
    const harmonySlider = document.getElementById("blendHarmonySlider");

    alphaSlider.addEventListener("input", () => {
        const a = parseFloat(alphaSlider.value);
        state.blend.alpha = a;
        const pctB = Math.round(a * 100);
        const pctA = 100 - pctB;
        document.getElementById("blendAlphaVal").textContent = `${pctA}% A — ${pctB}% B (α = ${a.toFixed(2)})`;
        document.getElementById("weightLabelA").textContent = `${pctA}%`;
        document.getElementById("weightLabelB").textContent = `${pctB}%`;
    });

    alphaSlider.addEventListener("change", () => {
        runBlendFaces();
    });

    harmonySlider.addEventListener("input", () => {
        state.blend.harmony = parseFloat(harmonySlider.value);
        document.getElementById("blendHarmonyVal").textContent = state.blend.harmony.toFixed(2);
    });

    harmonySlider.addEventListener("change", () => {
        runBlendFaces();
    });

    document.getElementById("btnRunBlend").addEventListener("click", () => {
        runBlendFaces();
    });

    const frontalIds = [
        "000007.jpg", "000013.jpg", "000015.jpg", "000018.jpg", "000023.jpg",
        "000026.jpg", "000031.jpg", "000038.jpg", "000043.jpg", "000045.jpg",
        "000051.jpg", "000056.jpg", "000063.jpg", "000068.jpg", "000072.jpg",
        "000079.jpg", "000085.jpg", "000092.jpg", "000099.jpg", "000106.jpg"
    ];

    document.getElementById("btnRandomPair").addEventListener("click", () => {
        const idx1 = Math.floor(Math.random() * frontalIds.length);
        let idx2 = Math.floor(Math.random() * frontalIds.length);
        if (idx2 === idx1) idx2 = (idx1 + 1) % frontalIds.length;
        state.blend.sourceAType = "celeba";
        state.blend.imageIdA = frontalIds[idx1];
        state.blend.sourceBType = "celeba";
        state.blend.imageIdB = frontalIds[idx2];
        runBlendFaces();
    });

    // Upload Image A & B from user's computer
    document.getElementById("uploadInputA").addEventListener("change", (e) => {
        handleFileUpload(e, "A");
    });
    document.getElementById("uploadInputB").addEventListener("change", (e) => {
        handleFileUpload(e, "B");
    });

    // Modal Picker controls
    document.getElementById("btnRefreshPicker").addEventListener("click", () => {
        loadPickerSamples();
    });
    document.getElementById("pickerAttrFilter").addEventListener("change", () => {
        loadPickerSamples();
    });
}

function handleFileUpload(event, slot) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
        const b64 = ev.target.result;
        if (slot === "A") {
            state.blend.sourceAType = "upload";
            state.blend.imageB64A = b64;
            document.getElementById("labelSourceA").textContent = `Upload: ${file.name}`;
        } else {
            state.blend.sourceBType = "upload";
            state.blend.imageB64B = b64;
            document.getElementById("labelSourceB").textContent = `Upload: ${file.name}`;
        }
        runBlendFaces();
    };
    reader.readAsDataURL(file);
}

async function runBlendFaces() {
    const btn = document.getElementById("btnRunBlend");
    btn.disabled = true;
    btn.innerHTML = "<span>⏳ Đang mã hóa &amp; lai ghép khuôn mặt...</span>";

    const payload = {
        source_a_type: state.blend.sourceAType,
        image_id_a: state.blend.imageIdA,
        image_b64_a: state.blend.imageB64A,
        source_b_type: state.blend.sourceBType,
        image_id_b: state.blend.imageIdB,
        image_b64_b: state.blend.imageB64B,
        alpha: state.blend.alpha,
        color_harmony: state.blend.harmony,
        num_steps: 7,
    };

    try {
        const res = await fetch("/api/blend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (data.ok) {
            document.getElementById("labelSourceA").textContent = data.label_a;
            document.getElementById("labelSourceB").textContent = data.label_b;
            document.getElementById("previewImgA").src = data.image_a_b64;
            document.getElementById("previewImgB").src = data.image_b_b64;
            document.getElementById("previewImgBlended").src = data.blended_image_b64;

            document.getElementById("btnDownloadBlended").href = data.blended_image_b64;
            document.getElementById("btnDownloadStrip").href = data.strip_banner_b64;

            document.getElementById("metricCosineSim").textContent = data.latent_cosine_similarity;
            document.getElementById("metricL2Dist").textContent = data.latent_l2_distance;
            document.getElementById("latentVectorPreview").textContent =
                `[${data.latent_z_blended.join(", ")}, ...]`;

            // Render 7-step Morphing Strip
            const stripEl = document.getElementById("morphStripContainer");
            stripEl.innerHTML = "";
            data.interpolation_strip.forEach((step) => {
                const isCurrent = Math.abs(step.alpha - data.alpha) < 0.09;
                const div = document.createElement("div");
                div.className = `morph-step-card ${isCurrent ? "active-step" : ""}`;
                div.innerHTML = `
                    <img src="${step.image_b64}" alt="Alpha ${step.alpha}" />
                    <div class="morph-step-label">α = ${step.alpha.toFixed(2)}</div>
                    <div style="font-size:0.65rem; color:#94a3b8;">${step.pct_a}% A / ${step.pct_b}% B</div>
                `;
                div.addEventListener("click", () => {
                    document.getElementById("previewImgBlended").src = step.image_b64;
                    document.getElementById("btnDownloadBlended").href = step.image_b64;
                    document.querySelectorAll(".morph-step-card").forEach((c) => c.classList.remove("active-step"));
                    div.classList.add("active-step");
                });
                stripEl.appendChild(div);
            });

            // Render Attribute Comparison Table
            const tbody = document.querySelector("#blendAttrTable tbody");
            tbody.innerHTML = "";
            data.attribute_comparison.forEach((row) => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td style="font-weight:600; color:#e2e8f0;">${row.label}</td>
                    <td style="color:#38bdf8;">${row.prob_a}%</td>
                    <td style="color:#a855f7;">${row.prob_b}%</td>
                    <td><strong style="color:#10b981;">${row.prob_blended}%</strong></td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (err) {
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.innerHTML = "<span>🧬 Tổng Hợp Khuôn Mặt Kết Hợp Mới</span>";
    }
}

function openCelebAPicker(slot) {
    state.blend.pickerTarget = slot;
    document.getElementById("pickerModalTitle").textContent = `Chọn Ảnh Nguồn ${slot} từ Tập Dữ Liệu CelebA (202,599 ảnh)`;
    document.getElementById("celebaPickerModal").classList.add("open");
    loadPickerSamples();
}

function closeCelebAPicker() {
    document.getElementById("celebaPickerModal").classList.remove("open");
}

async function loadPickerSamples() {
    const grid = document.getElementById("pickerGridContainer");
    const attr = document.getElementById("pickerAttrFilter").value;
    grid.innerHTML = "<div style='color:#94a3b8; padding:1rem;'>Đang tải mẫu ảnh từ archive.zip...</div>";
    try {
        const res = await fetch(`/api/dataset/samples?limit=24&random=1&attr=${encodeURIComponent(attr)}`);
        const data = await res.json();
        if (data.ok) {
            grid.innerHTML = "";
            data.samples.forEach((item) => {
                const div = document.createElement("div");
                div.className = "picker-item";
                div.innerHTML = `
                    <img src="${item.image_b64}" alt="${item.image_id}" />
                    <span>#${item.image_id}</span>
                `;
                div.addEventListener("click", () => {
                    if (state.blend.pickerTarget === "A") {
                        state.blend.sourceAType = "celeba";
                        state.blend.imageIdA = item.image_id;
                    } else {
                        state.blend.sourceBType = "celeba";
                        state.blend.imageIdB = item.image_id;
                    }
                    closeCelebAPicker();
                    runBlendFaces();
                });
                grid.appendChild(div);
            });
        }
    } catch (e) {
        console.error(e);
    }
}

// ----------------------------------------------------------------------------
// 4. TAB 3: ĐÁNH GIÁ CHẤT LƯỢNG ẢNH BẰNG CHỈ SỐ FID
// ----------------------------------------------------------------------------
function initFIDTab() {
    const sampleSlider = document.getElementById("fidSampleSize");
    const sampleVal = document.getElementById("fidSampleSizeVal");
    sampleSlider.addEventListener("input", () => {
        sampleVal.textContent = `${sampleSlider.value} mẫu`;
    });

    const psiSlider = document.getElementById("fidPsi");
    const psiVal = document.getElementById("fidPsiVal");
    psiSlider.addEventListener("input", () => {
        psiVal.textContent = parseFloat(psiSlider.value).toFixed(2);
    });

    document.getElementById("btnComputeFID").addEventListener("click", () => {
        runEvaluateFID();
    });
}

async function runEvaluateFID() {
    const btn = document.getElementById("btnComputeFID");
    btn.disabled = true;
    btn.innerHTML = "<span>⏳ Đang trích xuất đặc trưng &amp; tính ma trận hiệp phương sai...</span>";

    const payload = {
        model_type: document.getElementById("fidModelSelect").value,
        sample_size: parseInt(document.getElementById("fidSampleSize").value, 10),
        truncation_psi: parseFloat(document.getElementById("fidPsi").value),
        seed: Math.floor(1000 + Math.random() * 9000),
    };

    try {
        const res = await fetch("/api/evaluate_fid", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (data.ok) {
            const ev = data.evaluation;
            document.getElementById("fidMainScore").textContent = ev.fid_score.toFixed(2);
            document.getElementById("fidRatingText").textContent = `Chất lượng: ${ev.quality_rating} (${ev.elapsed_seconds}s)`;
            document.getElementById("fidMeanPart").textContent = ev.mean_component.toFixed(2);
            document.getElementById("fidCovPart").textContent = ev.covariance_component.toFixed(2);
            document.getElementById("fidIsScore").textContent = `${ev.inception_score} / ${ev.kid_score}`;

            // Render Comparison Table
            const tbody = document.querySelector("#fidComparisonTable tbody");
            tbody.innerHTML = "";
            ev.comparison_table.forEach((r, idx) => {
                const tr = document.createElement("tr");
                const highlight = idx === 2 ? "style='color:#10b981; font-weight:700;'" : "";
                tr.innerHTML = `
                    <td ${highlight}>${r.model_name}</td>
                    <td style="color:#94a3b8;">${r.architecture}</td>
                    <td ${highlight}>${r.fid}</td>
                    <td>${r.is_score}</td>
                    <td>${r.kid}</td>
                    <td>${r.params_m}</td>
                    <td>${r.attribute_control}</td>
                `;
                tbody.appendChild(tr);
            });

            drawFIDConvergenceChart(ev.fid_progression);
            drawFIDBarChart(ev.comparison_table);
        }
    } catch (e) {
        console.error(e);
    } finally {
        btn.disabled = false;
        btn.innerHTML = "<span>📊 Chạy Đánh Giá FID Trên Tập CelebA</span>";
    }
}

function drawFIDConvergenceChart(progression) {
    const canvas = document.getElementById("fidCurveCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 42, padR = 18, padT = 22, padB = 30;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    const maxY = 130;

    // Grid lines
    ctx.strokeStyle = "rgba(148, 163, 184, 0.15)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padT + (plotH * i) / 4;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(w - padR, y);
        ctx.stroke();

        const val = Math.round(maxY * (1 - i / 4));
        ctx.fillStyle = "#94a3b8";
        ctx.font = "11px Inter";
        ctx.fillText(val, 8, y + 4);
    }

    const drawSeries = (key, color) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        progression.forEach((pt, i) => {
            const x = padL + (plotW * i) / (progression.length - 1);
            const y = padT + plotH * (1 - pt[key] / maxY);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        progression.forEach((pt, i) => {
            const x = padL + (plotW * i) / (progression.length - 1);
            const y = padT + plotH * (1 - pt[key] / maxY);
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, 3.5, 0, Math.PI * 2);
            ctx.fill();

            if (key === "cdcgan") {
                ctx.fillStyle = "#94a3b8";
                ctx.font = "10px Inter";
                ctx.fillText(`Ep${pt.epoch}`, x - 12, h - 8);
            }
        });
    };

    drawSeries("dcgan", "#f59e0b");
    drawSeries("cdcgan", "#10b981");

    // Legend
    ctx.fillStyle = "#f59e0b";
    ctx.fillRect(padL + 10, 6, 10, 10);
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "11px Inter";
    ctx.fillText("DCGAN Cơ bản", padL + 25, 15);

    ctx.fillStyle = "#10b981";
    ctx.fillRect(padL + 130, 6, 10, 10);
    ctx.fillStyle = "#cbd5e1";
    ctx.fillText("Conditional DCGAN (Đề xuất)", padL + 145, 15);
}

function drawFIDBarChart(rows) {
    const canvas = document.getElementById("fidBarCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 42, padR = 16, padT = 22, padB = 34;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    const maxVal = Math.max(...rows.map((r) => r.fid), 50) * 1.15;

    const colors = ["#f59e0b", "#38bdf8", "#10b981", "#a855f7"];
    const shortLabels = ["DCGAN", "cDCGAN", "cDCGAN+Fusion", "WGAN-GP"];
    const barW = Math.min(48, plotW / (rows.length * 1.8));
    const stepX = plotW / rows.length;

    rows.forEach((r, i) => {
        const x = padL + i * stepX + (stepX - barW) / 2;
        const barH = (r.fid / maxVal) * plotH;
        const y = padT + plotH - barH;

        ctx.fillStyle = colors[i % colors.length];
        ctx.beginPath();
        ctx.roundRect(x, y, barW, barH, 5);
        ctx.fill();

        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 11px JetBrains Mono";
        ctx.fillText(r.fid, x + 4, y - 5);

        ctx.fillStyle = "#94a3b8";
        ctx.font = "10px Inter";
        ctx.fillText(shortLabels[i] || r.model_name.slice(0, 8), x - 4, h - 10);
    });
}

// ----------------------------------------------------------------------------
// 5. TAB 4: HUẤN LUYỆN MÔ HÌNH TRỰC TIẾP (TRAINING MONITOR)
// ----------------------------------------------------------------------------
function initTrainTab() {
    const epSlider = document.getElementById("trainEpochs");
    epSlider.addEventListener("input", () => {
        document.getElementById("trainEpochsVal").textContent = `${epSlider.value} epochs`;
    });

    const batchSlider = document.getElementById("trainBatch");
    batchSlider.addEventListener("input", () => {
        document.getElementById("trainBatchVal").textContent = batchSlider.value;
    });

    const samplesSlider = document.getElementById("trainSamples");
    samplesSlider.addEventListener("input", () => {
        document.getElementById("trainSamplesVal").textContent = `${samplesSlider.value} ảnh`;
    });

    document.getElementById("btnStartTrain").addEventListener("click", async () => {
        const payload = {
            model_type: document.getElementById("trainModelType").value,
            epochs: parseInt(epSlider.value, 10),
            batch_size: parseInt(batchSlider.value, 10),
            num_samples: parseInt(samplesSlider.value, 10),
            lr: 0.0002,
        };
        await fetch("/api/train/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        startTrainingPoller();
    });

    document.getElementById("btnStopTrain").addEventListener("click", async () => {
        await fetch("/api/train/stop", { method: "POST" });
    });

    const pthInput = document.getElementById("uploadKagglePth");
    if (pthInput) {
        pthInput.addEventListener("change", async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append("file", file);
            document.getElementById("trainStatusMessage").textContent = "Đang nạp file trọng số từ Kaggle...";
            const res = await fetch("/api/checkpoint/upload", { method: "POST", body: formData });
            const data = await res.json();
            document.getElementById("trainStatusMessage").textContent = data.message;
            pollTrainingStatus();
            runGenerateFaces();
            runBlendFaces();
        });
    }

    const btnReload = document.getElementById("btnReloadCkpt");
    if (btnReload) {
        btnReload.addEventListener("click", async () => {
            const res = await fetch("/api/checkpoint/reload", { method: "POST" });
            const data = await res.json();
            document.getElementById("trainStatusMessage").textContent = data.message;
            pollTrainingStatus();
            runGenerateFaces();
            runBlendFaces();
        });
    }
}

function startTrainingPoller() {
    if (state.trainPollTimer) clearInterval(state.trainPollTimer);
    state.trainPollTimer = setInterval(pollTrainingStatus, 450);
}

async function pollTrainingStatus() {
    try {
        const res = await fetch("/api/train/status");
        const data = await res.json();
        if (!data.ok) return;

        const st = data.status;
        document.getElementById("trainStatusMessage").textContent = st.message;
        document.getElementById("liveLossD").textContent = st.loss_d;
        document.getElementById("liveLossG").textContent = st.loss_g;
        document.getElementById("liveDx").textContent = st.d_x;
        document.getElementById("liveDgz").textContent = st.d_g_z;

        const pct = st.state === "idle" ? 100 : st.progress_pct;
        document.getElementById("trainProgressBar").style.width = `${pct}%`;
        document.getElementById("trainPctLabel").textContent = `${pct}%`;

        if (st.state === "running") {
            document.getElementById("trainEpochLabel").textContent =
                `Đang chạy Epoch ${st.current_epoch}/${st.total_epochs} (Step ${st.current_step}/${st.total_steps}) • ${st.elapsed_sec}s`;
        } else {
            document.getElementById("trainEpochLabel").textContent =
                `Trạng thái: ${st.state.toUpperCase()} • Tổng số Epochs tích lũy: ${data.checkpoint_meta.epochs_trained}`;
            if (state.trainPollTimer) {
                clearInterval(state.trainPollTimer);
                state.trainPollTimer = null;
            }
        }

        drawLossCurveCanvas(data.history || []);
    } catch (e) {
        console.error(e);
    }
}

function drawLossCurveCanvas(history) {
    const canvas = document.getElementById("lossCurveCanvas");
    if (!canvas || history.length === 0) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 45, padR = 20, padT = 24, padB = 28;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    const maxLoss = Math.max(...history.map((p) => Math.max(p.loss_g, p.loss_d)), 3.0) * 1.1;

    ctx.strokeStyle = "rgba(148, 163, 184, 0.14)";
    for (let i = 0; i <= 4; i++) {
        const y = padT + (plotH * i) / 4;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(w - padR, y);
        ctx.stroke();

        const v = (maxLoss * (1 - i / 4)).toFixed(1);
        ctx.fillStyle = "#94a3b8";
        ctx.font = "11px Inter";
        ctx.fillText(v, 10, y + 4);
    }

    const plotLine = (key, color) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        history.forEach((pt, idx) => {
            const x = padL + (plotW * idx) / Math.max(1, history.length - 1);
            const y = padT + plotH * (1 - pt[key] / maxLoss);
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    };

    plotLine("loss_g", "#38bdf8");
    plotLine("loss_d", "#f43f5e");

    ctx.fillStyle = "#38bdf8";
    ctx.fillRect(padL + 10, 6, 12, 10);
    ctx.fillStyle = "#e2e8f0";
    ctx.font = "11px Inter";
    ctx.fillText("Generator Loss (L_G)", padL + 28, 15);

    ctx.fillStyle = "#f43f5e";
    ctx.fillRect(padL + 175, 6, 12, 10);
    ctx.fillStyle = "#e2e8f0";
    ctx.fillText("Discriminator Loss (L_D)", padL + 193, 15);
}

// ----------------------------------------------------------------------------
// 6. TAB 5: KHÁM PHÁ TẬP DỮ LIỆU CELEBA (ARCHIVE.ZIP)
// ----------------------------------------------------------------------------
function initDatasetTab() {
    document.getElementById("btnReloadDsSamples").addEventListener("click", () => {
        fetchDatasetSamples();
    });
    document.getElementById("dsFilterAttr").addEventListener("change", () => {
        fetchDatasetSamples();
    });
}

async function fetchDatasetSummary() {
    try {
        const res = await fetch("/api/dataset/summary");
        const data = await res.json();
        if (data.ok) {
            const s = data.summary;
            document.getElementById("dsTotalImages").textContent = s.total_images.toLocaleString();
            document.getElementById("dsTotalAttrs").textContent = `${s.num_attributes} nhãn`;

            const tbody = document.querySelector("#dsAttrTable tbody");
            tbody.innerHTML = "";
            s.selected_attributes.forEach((attr) => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td style="font-family:'JetBrains Mono', monospace; color:#38bdf8;">${attr.key}</td>
                    <td>${attr.label}</td>
                    <td>${attr.positive_count.toLocaleString()}</td>
                    <td>
                        <div style="display:flex; align-items:center; gap:8px;">
                            <div style="width:70px; height:7px; background:rgba(255,255,255,0.1); border-radius:4px; overflow:hidden;">
                                <div style="width:${Math.min(100, attr.percentage)}%; height:100%; background:#6366f1;"></div>
                            </div>
                            <span>${attr.percentage}%</span>
                        </div>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (e) {
        console.error(e);
    }
}

async function fetchDatasetSamples() {
    const gallery = document.getElementById("dsSampleGallery");
    const attr = document.getElementById("dsFilterAttr").value;
    try {
        const res = await fetch(`/api/dataset/samples?limit=12&random=1&attr=${encodeURIComponent(attr)}`);
        const data = await res.json();
        if (data.ok) {
            gallery.innerHTML = "";
            data.samples.forEach((item) => {
                const tags = item.attributes.map((t) => `<span class="tag-chip">${t}</span>`).join("");
                const card = document.createElement("div");
                card.className = "face-card";
                card.innerHTML = `
                    <div class="face-img-wrap">
                        <img src="${item.image_b64}" alt="${item.image_id}" />
                        <span class="face-badge">CelebA #${item.image_id}</span>
                    </div>
                    <div class="face-meta">
                        <div class="face-tags">${tags}</div>
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:4px; margin-top:8px;">
                            <button type="button" class="btn-secondary" style="font-size:0.68rem; padding:3px 4px; justify-content:center;" data-slot="A">Đưa vào Ảnh A</button>
                            <button type="button" class="btn-secondary" style="font-size:0.68rem; padding:3px 4px; justify-content:center;" data-slot="B">Đưa vào Ảnh B</button>
                        </div>
                    </div>
                `;
                card.querySelector('[data-slot="A"]').addEventListener("click", () => {
                    state.blend.sourceAType = "celeba";
                    state.blend.imageIdA = item.image_id;
                    switchToTab("tab-blend");
                    runBlendFaces();
                });
                card.querySelector('[data-slot="B"]').addEventListener("click", () => {
                    state.blend.sourceBType = "celeba";
                    state.blend.imageIdB = item.image_id;
                    switchToTab("tab-blend");
                    runBlendFaces();
                });
                gallery.appendChild(card);
            });
        }
    } catch (e) {
        console.error(e);
    }
}
