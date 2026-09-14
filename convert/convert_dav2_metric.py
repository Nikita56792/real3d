#!/usr/bin/env python
"""
Convert Depth Anything V2 *Metric* Small (Hypersim indoor / VKITTI outdoor fine-tunes) to a Core ML
.mlpackage with a fixed 16:9-friendly input, using the same ANE-friendly DINOv2-S layout as
convert_dav2.py / ane_dav2.py (the encoder is identical; only the DPT head weights and the final
activation differ: metric = sigmoid(x) * max_depth, i.e. DEPTH IN METERS, larger = farther).

I/O contract (same as the relative packages):
  input  'image': 32BGRA image (W x H), raw 0..255 — ImageNet normalization is inside the graph
  output 'depth': GRAYSCALE_FLOAT16 image (W x H), metric depth in meters (larger = farther)

Usage:
  python convert_dav2_metric.py --domain indoor  --width 644 --height 364 \
      --out /path/DAv2MetricIndoor_644x364.mlpackage --photo street.png --photo indoor.png
  python convert_dav2_metric.py --domain outdoor ...
"""
import argparse
import os
import shutil
import sys
import time
from collections import Counter

import numpy as np
import torch
import coremltools as ct
from PIL import Image
from transformers import AutoModelForDepthEstimation

from ane_dav2 import DepthAnythingV2ANE
from convert_dav2 import load_photo_bgr, normalized

HF_IDS = {
    "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",    # Hypersim, max_depth 20
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",  # VKITTI,  max_depth 80
}


def depth_stats(a: np.ndarray) -> str:
    a = a.astype(np.float64)
    return (f"min {a.min():.2f}  p5 {np.percentile(a, 5):.2f}  median {np.median(a):.2f}  "
            f"p95 {np.percentile(a, 95):.2f}  max {a.max():.2f} m")


def probe(a: np.ndarray, points):
    """points: list of (label, x_frac, y_frac); returns 'label=value' strings using a 5x5 median."""
    H, W = a.shape
    out = []
    for label, xf, yf in points:
        x, y = int(xf * (W - 1)), int(yf * (H - 1))
        patch = a[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        out.append(f"{label}={np.median(patch):.2f}m")
    return "  ".join(out)


def verify_one(photo: str, wrapper, mlmodel_path: str, W: int, H: int, base: str, dump_dir: str, tag: str):
    pil_rgb, t_bgr = load_photo_bgr(photo, W, H)
    with torch.no_grad():
        pt = wrapper(t_bgr)[0, 0].numpy().astype(np.float32)
    ml = ct.models.MLModel(mlmodel_path, compute_units=ct.ComputeUnit.ALL)
    pred = ml.predict({"image": pil_rgb})["depth"]
    if isinstance(pred, Image.Image):
        cm = np.asarray(pred).astype(np.float32)
    else:
        cm = np.asarray(pred).astype(np.float32).reshape(H, W)
    finite = bool(np.isfinite(cm).all())
    corr = float(np.corrcoef(normalized(pt).ravel(), normalized(cm).ravel())[0, 1])
    abs_rel = np.abs(cm - pt) / np.maximum(pt, 1e-3)
    med_abs_rel = float(np.median(abs_rel))
    mean_abs_rel = float(abs_rel.mean())
    max_abs = float(np.abs(cm - pt).max())
    print(f"  [{tag}] {os.path.basename(photo)}")
    print(f"     pytorch : {depth_stats(pt)}")
    print(f"     coreml  : {depth_stats(cm)}   finite={finite}")
    print(f"     corr={corr:.5f}  median abs-rel err={med_abs_rel:.5f}  mean abs-rel err={mean_abs_rel:.5f}  "
          f"max |diff|={max_abs:.3f} m")
    status = "PASS" if (corr > 0.99 and finite and med_abs_rel < 0.02) else "FAIL"
    print(f"     numeric check: {status}")

    def to_u8(a):  # white = nearer (invert metric depth for display)
        inv = 1.0 / np.maximum(a, 1e-3)
        lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
        return (np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)

    os.makedirs(dump_dir, exist_ok=True)
    side = np.concatenate([np.asarray(pil_rgb), np.stack([to_u8(pt)] * 3, -1), np.stack([to_u8(cm)] * 3, -1)], axis=1)
    cmp_path = os.path.join(dump_dir, f"{base}_{tag}_compare.png")
    Image.fromarray(side).save(cmp_path)
    np.save(os.path.join(dump_dir, f"{base}_{tag}_coreml_depth_m.npy"), cm)
    return dict(tag=tag, photo=photo, corr=corr, med_abs_rel=med_abs_rel, mean_abs_rel=mean_abs_rel,
                max_abs=max_abs, status=status, pt=pt, cm=cm, cmp=cmp_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(HF_IDS), required=True)
    ap.add_argument("--width", type=int, default=644)
    ap.add_argument("--height", type=int, default=364)
    ap.add_argument("--out", required=True)
    ap.add_argument("--photo", action="append", default=[], help="test image(s); may be given multiple times")
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--tensor-output", action="store_true")
    args = ap.parse_args()

    W, H = args.width, args.height
    assert W % 14 == 0 and H % 14 == 0, "resolution must be a multiple of 14"
    hf_id = HF_IDS[args.domain]
    print(f"=== DAv2-Metric-{args.domain}-Small -> Core ML {W}x{H} (grid {W // 14}x{H // 14} = {(W // 14) * (H // 14)} patches)")
    print("  source:", hf_id)

    torch.manual_seed(0)
    model = AutoModelForDepthEstimation.from_pretrained(hf_id, attn_implementation="eager").eval()
    cfg = model.config
    assert cfg.depth_estimation_type == "metric", cfg.depth_estimation_type
    max_depth = float(cfg.max_depth)
    print(f"  depth_estimation_type={cfg.depth_estimation_type}  max_depth={max_depth}  "
          f"head.activation2={model.head.activation2.__class__.__name__}")
    assert isinstance(model.head.activation2, torch.nn.Sigmoid)

    wrapper = DepthAnythingV2ANE(model, H, W).eval()
    example = torch.rand(1, 3, H, W) * 255.0
    with torch.no_grad():
        ref = wrapper(example)
    print(f"  eager out {tuple(ref.shape)} range [{ref.min():.3f}, {ref.max():.3f}] (<= max_depth {max_depth})")
    assert float(ref.max()) <= max_depth + 1e-3 and float(ref.min()) >= 0.0

    print("  tracing ...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, check_trace=False)
        tr_out = traced(example)
    print("  trace-vs-eager max abs diff:", float((tr_out - ref).abs().max()))

    def convert(outputs):
        return ct.convert(
            traced, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS14,
            inputs=[ct.ImageType(name="image", shape=(1, 3, H, W), color_layout=ct.colorlayout.BGR, scale=1.0)],
            outputs=outputs, compute_units=ct.ComputeUnit.ALL)

    t0 = time.time()
    if args.tensor_output:
        mlmodel = convert([ct.TensorType(name="depth", dtype=np.float16)]); out_kind = "TensorType fp16"
    else:
        try:
            mlmodel = convert([ct.ImageType(name="depth", color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)])
            out_kind = "ImageType GRAYSCALE_FLOAT16"
        except Exception as e:  # noqa: BLE001
            print("  !! GRAYSCALE_FLOAT16 image output failed, falling back to TensorType fp16:", repr(e))
            mlmodel = convert([ct.TensorType(name="depth", dtype=np.float16)]); out_kind = "TensorType fp16 (fallback)"
    print(f"  converted in {time.time() - t0:.1f}s, output = {out_kind}")

    ops = Counter(op.op_type for op in mlmodel._mil_program.functions["main"].operations if op.op_type != "const")
    print("  MIL ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    assert ops.get("sigmoid", 0) >= 1, "sigmoid missing from graph -- metric head not traced?"

    dom = "Indoor (Hypersim)" if args.domain == "indoor" else "Outdoor (VKITTI)"
    mlmodel.author = f"Depth Anything V2 Metric {dom} Small (HF {hf_id}), converted with coremltools"
    mlmodel.short_description = (f"Depth Anything V2 Metric {dom} Small, {W}x{H} BGRA input; output is METRIC DEPTH "
                                 f"IN METERS (larger = farther, sigmoid*{max_depth:g}) as {W}x{H} float16 grayscale. "
                                 f"ImageNet normalization in-graph.")
    mlmodel.version = "1.0"
    mlmodel.input_description["image"] = f"{W}x{H} BGRA image, raw 0..255 (normalization inside the model)"
    mlmodel.output_description["depth"] = f"Metric depth in meters (0..{max_depth:g}), float16, larger = farther"
    mlmodel.user_defined_metadata["polarity"] = "depth"
    mlmodel.user_defined_metadata["units"] = "meters"
    mlmodel.user_defined_metadata["max_depth"] = f"{max_depth:g}"
    mlmodel.user_defined_metadata["domain"] = args.domain

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    mlmodel.save(args.out)
    print("  saved", args.out)

    if not args.photo:
        return
    base = os.path.splitext(os.path.basename(args.out))[0]
    dump_dir = args.dump_dir or os.path.dirname(args.out)
    results = []
    for i, photo in enumerate(args.photo):
        results.append(verify_one(photo, wrapper, args.out, W, H, base, dump_dir, tag=f"photo{i}"))
    overall = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"
    with open(os.path.join(dump_dir, f"{base}_verify.txt"), "w") as f:
        f.write(f"{base}\nsource={hf_id}\nmax_depth={max_depth:g}\noutput={out_kind}\n"
                f"polarity=metric depth in meters (larger = farther)\nstatus={overall}\n")
        for r in results:
            f.write(f"[{r['tag']}] {r['photo']}\n  corr={r['corr']:.5f} median_abs_rel={r['med_abs_rel']:.5f} "
                    f"mean_abs_rel={r['mean_abs_rel']:.5f} max_abs_diff_m={r['max_abs']:.3f} status={r['status']}\n"
                    f"  coreml range: {depth_stats(r['cm'])}\n")
    print(f"  overall: {overall}")
    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
