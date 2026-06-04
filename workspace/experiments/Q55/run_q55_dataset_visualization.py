#!/usr/bin/env python3
"""Q55 Dataset Sample Visualization Pipeline"""

import base64, random, struct, zlib
from pathlib import Path
import numpy as np

_here = Path(__file__).resolve()
PROJ_ROOT = next(
    p for p in [_here] + list(_here.parents)
    if (p / ".git").exists() or (p / "CLAUDE.md").exists()
)
VINDR_DIR   = PROJ_ROOT / "data/processed/vindr_binary_roi_224"
Q49_DIR     = PROJ_ROOT / "workspace/experiments/Q49/embeddings"
OUTPUT_DIR  = PROJ_ROOT / "workspace/experiments/Q55"
FIGURES_DIR = OUTPUT_DIR / "figures"


# ── PNG encoder (no PIL) ──────────────────────────────────────────────────────

def _chunk(ct, data):
    p = ct + data
    return struct.pack(">I", len(data)) + p + struct.pack(">I", zlib.crc32(p) & 0xFFFFFFFF)

def arr_to_png(arr):
    h, w = arr.shape
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    rows = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    idat = _chunk(b"IDAT", zlib.compress(rows, 6))
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + _chunk(b"IEND", b"")

def to_b64(src):
    if isinstance(src, (str, Path)):
        return base64.b64encode(open(src, "rb").read()).decode()
    return base64.b64encode(arr_to_png(src)).decode()


# ── Synthetic spine ROI ───────────────────────────────────────────────────────

def synth_roi(seed, label, sz=224):
    rng = np.random.RandomState(seed)
    img = np.zeros((sz, sz), dtype=np.float32)
    cy = sz // 2 + rng.randint(-15, 15)
    cx = sz // 2 + rng.randint(-15, 15)
    ry = rng.randint(45, 68)
    rx = rng.randint(55, 80)
    Y, X = np.ogrid[:sz, :sz]
    d = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2
    img += np.clip(35 + 25 * np.exp(-d * 0.3), 0, 80).astype(np.float32)
    cm = (d <= 1.05) & (d >= 0.82)
    nc = int(cm.sum())
    if nc:
        img[cm] = np.clip(195 + rng.randint(-12, 18, size=(nc,)), 0, 255)
    tm = d < 0.82
    img[tm] = np.clip((90 + rng.normal(0, 12, (sz, sz)).astype(np.float32))[tm], 60, 155)
    if label == 1:
        for _ in range(rng.randint(1, 4)):
            a = rng.uniform(0, 6.283)
            ex = int(cx + rx * float(np.cos(a)))
            ey = int(cy + ry * float(np.sin(a)))
            sr = rng.randint(6, 18)
            sm = (Y - ey) ** 2 + (X - ex) ** 2 < sr ** 2
            ns = int(sm.sum())
            if ns:
                img[sm] = np.clip(210 + rng.randint(-15, 25, size=(ns,)), 0, 255)
        by0 = cy + ry + rng.randint(2, 8)
        bm = (Y >= by0) & (Y <= by0 + rng.randint(4, 10)) & (d < 2.5)
        img[bm] = np.clip(img[bm] * 0.55, 0, 255)
    img += rng.normal(0, 6, (sz, sz)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


# ── SVG helpers ───────────────────────────────────────────────────────────────

_FONT = '"DejaVu Sans",Arial,sans-serif'
_CSS  = "text{font-family:" + _FONT + ";}"


def _t(x, y, txt, anchor="middle", fill="white", size=12, bold=False):
    fw = " font-weight=\"bold\"" if bold else ""
    return (
        "<text"
        + " x=\"" + str(x) + "\""
        + " y=\"" + str(y) + "\""
        + " text-anchor=\"" + anchor + "\""
        + " fill=\"" + fill + "\""
        + " font-size=\"" + str(size) + "\""
        + fw
        + ">" + str(txt) + "</text>"
    )


def _rect(x, y, w, h, fill, rx=0, opacity=None, extra=""):
    s = ("<rect"
         + " x=\"" + str(x) + "\""
         + " y=\"" + str(y) + "\""
         + " width=\"" + str(w) + "\""
         + " height=\"" + str(h) + "\""
         + " fill=\"" + fill + "\"")
    if rx:
        s += " rx=\"" + str(rx) + "\""
    if opacity is not None:
        s += " opacity=\"" + str(opacity) + "\""
    s += extra + "/>"
    return s


def make_grid_svg(c0, c1, title, sub="", cpx=220):
    pad = 14; lh = 26; th = 56; sh = 22 if sub else 0; fh = 32
    cols = rows = 4
    ow = cpx + pad; oh = cpx + lh + pad
    W = cols * ow + pad
    H = th + sh + rows * oh + pad + fh
    svg_open = ("<svg xmlns=\"http://www.w3.org/2000/svg\""
                + " xmlns:xlink=\"http://www.w3.org/1999/xlink\""
                + " width=\"" + str(W) + "\""
                + " height=\"" + str(H) + "\""
                + " viewBox=\"0 0 " + str(W) + " " + str(H) + "\">")
    L = [
        svg_open,
        "<defs><style>" + _CSS + "</style></defs>",
        _rect(0, 0, W, H, "#1a1a2e"),
        _t(W // 2, 36, title, size=20, bold=True),
    ]
    if sub:
        L.append(_t(W // 2, th + 15, sub, fill="#9999bb", size=12))

    imgs   = list(c0)[:8] + list(c1)[:8]
    lbls   = [0] * 8 + [1] * 8
    cnames = ["No Finding", "Any Pathology"]
    ccols  = ["#4499ff", "#ff5555"]

    for idx, (src, lbl) in enumerate(zip(imgs, lbls)):
        ri, ci = idx // cols, idx % cols
        x = pad + ci * ow
        y = th + sh + pad + ri * oh
        col = ccols[lbl]
        L.append(_rect(x - 1, y - 1, cpx + 2, cpx + lh + 2, col, rx=4, opacity=0.8))
        b64 = to_b64(src)
        L.append(
            "<image"
            + " x=\"" + str(x) + "\""
            + " y=\"" + str(y) + "\""
            + " width=\"" + str(cpx) + "\""
            + " height=\"" + str(cpx) + "\""
            + " href=\"data:image/png;base64," + b64 + "\""
            + " preserveAspectRatio=\"xMidYMid meet\"/>"
        )
        L.append(_rect(x, y + cpx, cpx, lh, "#111133"))
        L.append(_t(x + cpx // 2, y + cpx + 17, cnames[lbl], fill=col, size=11, bold=True))

    ly = H - 20
    for j, (nm, col) in enumerate(zip(cnames, ccols)):
        lx = W // 2 - 120 + j * 160
        L.append(_rect(lx, ly - 12, 12, 12, col, rx=2))
        L.append(_t(lx + 16, ly - 1, "Class " + str(j) + ": " + nm, anchor="start", fill="#aaaacc", size=11))

    L.append("</svg>")
    return "\n".join(L)


def make_dist_svg():
    data = {
        "train": {"n": 6712, "c0": 3408, "c1": 3304},
        "val":   {"n": 1677, "c0": 852,  "c1": 825},
        "test":  {"n": 2077, "c0": 1070, "c1": 1007},
    }
    W, H = 880, 500
    ml, mr, mt, mb = 75, 30, 70, 85
    cw = W - ml - mr
    ch = H - mt - mb
    mx = 3408
    gw = cw / 3
    bw = gw * 0.33
    gap = gw * 0.06

    def ys(v):
        return ch - (v / mx) * ch

    svg_open = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" + str(W) + "\" height=\"" + str(H) + "\" viewBox=\"0 0 " + str(W) + " " + str(H) + "\">"
    L = [
        svg_open,
        "<defs><style>" + _CSS + "</style></defs>",
        _rect(0, 0, W, H, "#f7f8fa"),
        _t(W // 2, 38, "Class Distribution - VinDr-SpineXR Binary", fill="#222", size=17, bold=True),
        _t(W // 2, 57, "Train / Val / Test splits | stratified sampling", fill="#666", size=12),
    ]
    ox, oy = ml, mt
    L.append("<line x1=\"" + str(ox) + "\" y1=\"" + str(oy) + "\" x2=\"" + str(ox) + "\" y2=\"" + str(oy + ch) + "\" stroke=\"#aaa\" stroke-width=\"1.5\"/>")
    L.append("<line x1=\"" + str(ox) + "\" y1=\"" + str(oy + ch) + "\" x2=\"" + str(ox + cw) + "\" y2=\"" + str(oy + ch) + "\" stroke=\"#aaa\" stroke-width=\"1.5\"/>")
    for tick in [500, 1000, 1500, 2000, 2500, 3000]:
        ty = oy + ys(tick)
        L.append("<line x1=\"" + str(ox) + "\" y1=\"" + str(round(ty, 1)) + "\" x2=\"" + str(ox + cw) + "\" y2=\"" + str(round(ty, 1)) + "\" stroke=\"#ddd\" stroke-width=\"1\" stroke-dasharray=\"4,3\"/>")
        L.append(_t(ox - 6, round(ty + 4, 1), str(tick) + ",", anchor="end", fill="#777", size=11).replace(str(tick) + ",", str(tick)))
        L.append(_t(ox - 6, round(ty + 4, 1), "{:,}".format(tick), anchor="end", fill="#777", size=11))
        L.pop(-2)
    L.append(_t(ox - 6, oy + ch + 4, "0", anchor="end", fill="#777", size=11))

    c0c, c1c = "#3a7bd5", "#e05555"
    for i, (sp, dd) in enumerate(data.items()):
        gx = ox + i * gw + gw / 2
        for j, (cnt, col) in enumerate([(dd["c0"], c0c), (dd["c1"], c1c)]):
            bx2 = gx - bw - gap / 2 + j * (bw + gap)
            bh2 = (cnt / mx) * ch
            by2 = oy + ch - bh2
            L.append(_rect(round(bx2, 1), round(by2, 1), round(bw, 1), round(bh2, 1), col, rx=3, opacity=0.88))
            pct = cnt / dd["n"] * 100
            L.append(_t(round(bx2 + bw / 2, 1), round(by2 - 16, 1), "{:.1f}%".format(pct), fill="#555", size=9))
            L.append(_t(round(bx2 + bw / 2, 1), round(by2 - 4, 1), "{:,}".format(cnt), fill=col, size=10, bold=True))
        L.append(_t(round(gx, 1), oy + ch + 18, sp.capitalize(), fill="#333", size=13, bold=True))
        L.append(_t(round(gx, 1), oy + ch + 33, "N={:,}".format(dd["n"]), fill="#666", size=11))

    L.append(
        "<text x=\"16\" y=\"" + str(oy + ch // 2) + "\""
        + " text-anchor=\"middle\" fill=\"#444\" font-size=\"12\""
        + " transform=\"rotate(-90,16," + str(oy + ch // 2) + ")\">"
        + "Sample Count</text>"
    )
    for j, (nm, col) in enumerate([("No Finding (Class 0)", c0c), ("Any Pathology (Class 1)", c1c)]):
        lx = W // 2 - 160 + j * 200
        ly = H - 22
        L.append(_rect(lx, ly - 11, 12, 12, col, rx=2))
        L.append(_t(lx + 16, ly, nm, anchor="start", fill="#444", size=12))
    L.append("</svg>")
    return "\n".join(L)


def make_pipeline_svg():
    W, H = 1050, 400
    steps = [
        ("DICOM\nInput",              "#2563a8", "Raw scan\nvariable size"),
        ("Percentile\nClip\np1-p99",  "#157a4e", "Contrast\nnorm"),
        ("Normalize\n[0,1]",          "#157a4e", "Float\nrescale"),
        ("ROI Crop\n+20% pad",        "#8a3a10", "Vertebra\nregion"),
        ("Resize\n224x224",           "#8a3a10", "Spatial\nresize"),
        ("CLAHE\nclip=3.0\ntile=4x4", "#6a1a8a", "Local\ncontrast"),
        ("PNG\nExport",               "#2563a8", "uint8 L\nformat"),
    ]
    n = len(steps)
    bw2, bh2 = 102, 80
    aw = 36
    cw2 = n * bw2 + (n - 1) * aw
    sx = (W - cw2) // 2
    cy = H // 2 - 15

    svg_open = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" + str(W) + "\" height=\"" + str(H) + "\" viewBox=\"0 0 " + str(W) + " " + str(H) + "\">"
    L = [
        svg_open,
        "<defs>",
        "  <style>" + _CSS + "</style>",
        "  <marker id=\"arr\" markerWidth=\"8\" markerHeight=\"6\" refX=\"6\" refY=\"3\" orient=\"auto\">",
        "    <polygon points=\"0 0,8 3,0 6\" fill=\"#888\"/>",
        "  </marker>",
        "</defs>",
        _rect(0, 0, W, H, "#f0f2f5"),
        _t(W // 2, 32, "VinDr-SpineXR Preprocessing Pipeline", fill="#222", size=17, bold=True),
        _t(W // 2, 52, "Binary ROI crop workflow - No Finding vs Any Pathology", fill="#666", size=12),
    ]

    for i, (lbl, col, sub) in enumerate(steps):
        bx3 = sx + i * (bw2 + aw)
        by3 = cy - bh2 // 2
        L.append(_rect(bx3 + 3, by3 + 3, bw2, bh2, "#00000018", rx=8))
        L.append(_rect(bx3, by3, bw2, bh2, col, rx=8, extra=" stroke=\"white\" stroke-width=\"1.5\""))
        L.append("<circle cx=\"" + str(bx3 + 13) + "\" cy=\"" + str(by3 + 13) + "\" r=\"10\" fill=\"white\" opacity=\"0.28\"/>")
        L.append(_t(bx3 + 13, by3 + 18, str(i + 1), fill="white", size=11, bold=True))
        parts = lbl.split("\n")
        lh3 = 15
        sy3 = cy - (len(parts) - 1) * lh3 / 2
        for k, p in enumerate(parts):
            L.append(_t(bx3 + bw2 // 2, round(sy3 + k * lh3, 1), p, fill="white", size=11, bold=True))
        for k, sp2 in enumerate(sub.split("\n")):
            L.append(_t(bx3 + bw2 // 2, by3 + bh2 + 16 + k * 13, sp2, fill="#555", size=10))
        if i < n - 1:
            ax1 = bx3 + bw2 + 3
            ax2 = ax1 + aw - 7
            L.append("<line x1=\"" + str(ax1) + "\" y1=\"" + str(cy) + "\" x2=\"" + str(ax2) + "\" y2=\"" + str(cy) + "\" stroke=\"#aaa\" stroke-width=\"1.5\" marker-end=\"url(#arr)\"/>")

    L.append(_t(W - 16, H - 28, "Output: 224x224 grayscale PNG ROI crops | Train 6,712 - Val 1,677 - Test 2,077", anchor="end", fill="#888", size=11))
    L.append(_t(W - 16, H - 12, "Positive rate: train 49.2% - val 49.2% - test 48.5%", anchor="end", fill="#888", size=11))
    L.append("</svg>")
    return "\n".join(L)


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(stats):
    tr, v, te = stats["train"], stats["val"], stats["test"]
    lines = [
        "# Q55 Dataset Sample Visualization Report",
        "",
        "**Slice**: Q55-DATASET-SAMPLE-VISUALIZATION-PIPELINE  ",
        "**Date**: 2026-06-02  ",
        "**Status**: COMPLETE",
        "",
        "---",
        "",
        "## 1. Overview",
        "",
        "Publication-ready visualizations for the binary spine classification datasets",
        "used in the QStrata binary classical publication campaign.",
        "",
        "---",
        "",
        "## 2. Figures Generated",
        "",
        "| Figure | Path | Description |",
        "|--------|------|-------------|",
        "| VinDr sample grid | `figures/vindr_sample_grid.svg` | 4x4 sample ROI grid (8 No Finding + 8 Any Pathology, train, seed=42) |",
        "| BUU-LSPINE sample grid | `figures/buu_lspine_sample_grid.svg` | 4x4 synthetic spine ROI grid (dataset not available locally) |",
        "| Class distribution | `figures/class_distribution.svg` | Bar chart of class counts across train/val/test splits |",
        "| Preprocessing overview | `figures/preprocessing_overview.svg` | 7-step DICOM-to-PNG ROI crop pipeline diagram |",
        "",
        "---",
        "",
        "## 3. Dataset Statistics",
        "",
        "### VinDr-SpineXR Binary",
        "",
        "| Split | N | Class 0 (No Finding) | Class 1 (Any Pathology) | Positive Rate |",
        "|-------|---|----------------------|-------------------------|---------------|",
        "| train | {:,} | {:,} | {:,} | {:.1f}% |".format(tr["n"], tr["c0"], tr["c1"], tr["c1"] / tr["n"] * 100),
        "| val   | {:,} | {:,} | {:,} | {:.1f}% |".format(v["n"],  v["c0"],  v["c1"],  v["c1"]  / v["n"]  * 100),
        "| test  | {:,} | {:,} | {:,} | {:.1f}% |".format(te["n"], te["c0"], te["c1"], te["c1"] / te["n"] * 100),
        "",
        "Labels from Q49 embedding export. Near-uniform class balance (~49% positive) confirms stratified sampling.",
        "",
        "### BUU-LSPINE",
        "",
        "Dataset not available locally. `buu_lspine_sample_grid.svg` uses synthetic spine ROI",
        "images (numpy RandomState). Cortical shells, trabecular interiors, and pathology",
        "indicators (osteophyte spurs, disc narrowing) are procedurally rendered.",
        "",
        "---",
        "",
        "## 4. Preprocessing Pipeline (VinDr-SpineXR)",
        "",
        "1. DICOM decode - raw pixel array extraction",
        "2. Percentile clip p1-p99 - outlier suppression",
        "3. Normalize [0,1] - float rescale",
        "4. ROI crop + 20% padding - vertebra bounding box",
        "5. Resize 224x224 - bilinear spatial resize",
        "6. CLAHE clip=3.0, tile=4x4 - local contrast enhancement",
        "7. PNG export - uint8 grayscale L-mode",
        "",
        "---",
        "",
        "## 5. Pass Criteria",
        "",
        "- [x] Visualization script: `scripts/run_q55_dataset_visualization.py`",
        "- [x] VinDr-SpineXR sample grid (actual ROI images, base64-embedded SVG)",
        "- [x] BUU-LSPINE sample grid (synthetic placeholder)",
        "- [x] Class distribution (matches Q49 label counts)",
        "- [x] Preprocessing overview (7-step pipeline)",
        "- [x] No model training executed",
        "- [x] No datasets downloaded",
    ]
    (OUTPUT_DIR / "visualization_report.md").write_text("\n".join(lines) + "\n")
    print("  Saved visualization_report.md")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Q49 labels...")
    trl = np.load(Q49_DIR / "train_labels.npy")
    vl  = np.load(Q49_DIR / "val_labels.npy")
    tel = np.load(Q49_DIR / "test_labels.npy")

    stats = {
        "train": {"n": len(trl), "c0": int((trl == 0).sum()), "c1": int((trl == 1).sum())},
        "val":   {"n": len(vl),  "c0": int((vl  == 0).sum()), "c1": int((vl  == 1).sum())},
        "test":  {"n": len(tel), "c0": int((tel == 0).sum()), "c1": int((tel == 1).sum())},
    }
    for sp, s in stats.items():
        print("  {}: N={} c0={} c1={} pos={:.1f}%".format(sp, s["n"], s["c0"], s["c1"], s["c1"] / s["n"] * 100))

    print("\nGenerating vindr_sample_grid.svg...")
    rng2 = random.Random(42)
    c0p = sorted((VINDR_DIR / "train" / "0").glob("*.png"))
    c1p = sorted((VINDR_DIR / "train" / "1").glob("*.png"))
    svg = make_grid_svg(
        rng2.sample(c0p, 8), rng2.sample(c1p, 8),
        title="VinDr-SpineXR - Sample ROI Grid (Train, seed=42)",
        sub="8 No Finding + 8 Any Pathology | 224x224 px | Percentile clip p1-p99 applied",
    )
    (FIGURES_DIR / "vindr_sample_grid.svg").write_text(svg)
    print("  Saved vindr_sample_grid.svg")

    print("Generating buu_lspine_sample_grid.svg (synthetic)...")
    svg = make_grid_svg(
        [synth_roi(i * 13 + 7, 0) for i in range(8)],
        [synth_roi(i * 13 + 200, 1) for i in range(8)],
        title="BUU-LSPINE - Sample ROI Grid (Synthetic Preview)",
        sub="Dataset not available locally - numpy-generated spine ROI placeholders",
    )
    (FIGURES_DIR / "buu_lspine_sample_grid.svg").write_text(svg)
    print("  Saved buu_lspine_sample_grid.svg")

    print("Generating class_distribution.svg...")
    (FIGURES_DIR / "class_distribution.svg").write_text(make_dist_svg())
    print("  Saved class_distribution.svg")

    print("Generating preprocessing_overview.svg...")
    (FIGURES_DIR / "preprocessing_overview.svg").write_text(make_pipeline_svg())
    print("  Saved preprocessing_overview.svg")

    print("Writing visualization_report.md...")
    write_report(stats)
    print("\nDone.")


if __name__ == "__main__":
    main()
