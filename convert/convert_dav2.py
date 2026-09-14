#!/usr/bin/env python
"""
Convert Depth Anything V2 Small (HF: depth-anything/Depth-Anything-V2-Small-hf) to a Core ML
.mlpackage with a fixed 16:9-friendly input resolution.

I/O contract mirrors Apple's DepthAnythingV2SmallF16.mlpackage:
  input  'image': 32BGRA image  (W x H), raw 0..255 — ImageNet normalization is inside the graph
  output 'depth': GRAYSCALE_FLOAT16 image (W x H), relative inverse depth (larger = nearer)

Usage:
  python convert_dav2.py --width 518 --height 294 --out /path/DAv2Small_518x294.mlpackage
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from PIL import Image
from transformers import AutoModelForDepthEstimation

from ane_dav2 import DepthAnythingV2ANE

HF_ID = "depth-anything/Depth-Anything-V2-Small-hf"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DAv2Wrapper(nn.Module):
    """Takes a [1,3,H,W] float tensor in BGR channel order with values in [0,255],
    normalizes with ImageNet stats in-graph and returns [1,1,H,W] relative inverse depth."""

    def __init__(self, model: nn.Module, height: int, width: int):
        super().__init__()
        self.model = model
        ps = model.config.patch_size
        self.ph, self.pw = height // ps, width // ps  # Python ints -> baked into the trace
        # Stats are stored in BGR order so the input can be consumed as-is from a 32BGRA buffer.
        mean = torch.tensor(IMAGENET_MEAN[::-1], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD[::-1], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x / 255.0 - self.mean) / self.std           # normalize (BGR order)
        x = torch.cat([x[:, 2:3], x[:, 1:2], x[:, 0:1]], dim=1)  # BGR -> RGB for the backbone
        # Drive backbone / neck / head directly with Python-int patch dims. The stock
        # DepthAnythingForDepthEstimation.forward derives them from pixel_values.shape, which
        # traces as size tensors and trips coremltools' `int` op converter.
        feats = self.model.backbone(x).feature_maps
        feats = self.model.neck(feats, self.ph, self.pw)
        depth = self.model.head(feats, self.ph, self.pw)  # [1,H,W]
        return depth.unsqueeze(1)                          # [1,1,H,W]


def bake_position_embeddings(model: nn.Module, height: int, width: int) -> None:
    """Replace the 37x37 pre-trained pos-embeddings with a fixed, bicubic-interpolated
    (H/14)x(W/14) grid so the trace has no dynamic interpolation (and no bicubic op)."""
    emb = model.backbone.embeddings
    ps = emb.patch_size if hasattr(emb, "patch_size") else model.config.patch_size
    gh, gw = height // ps, width // ps
    dim = emb.position_embeddings.shape[-1]
    with torch.no_grad():
        dummy = torch.zeros(1, 1 + gh * gw, dim)
        new_pe = emb.interpolate_pos_encoding(dummy, height, width).detach().clone()
    assert new_pe.shape == (1, 1 + gh * gw, dim), new_pe.shape
    emb.position_embeddings = nn.Parameter(new_pe, requires_grad=False)
    # Short-circuit the dynamic path: the grid is now exactly the target grid.
    emb.interpolate_pos_encoding = lambda embeddings, h, w: emb.position_embeddings
    print(f"  baked pos-embeddings -> grid {gh}x{gw} ({gh * gw} patches), shape {tuple(new_pe.shape)}")


def load_photo_bgr(path: str, width: int, height: int):
    """Returns (pil_rgb_resized, torch_bgr_[1,3,H,W]_0..255)."""
    img = Image.open(path).convert("RGB").resize((width, height), Image.BICUBIC)
    rgb = np.asarray(img).astype(np.float32)          # H,W,3 RGB
    bgr = rgb[..., ::-1].copy()
    t = torch.from_numpy(bgr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return img, t


def normalized(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--photo", default=None, help="test image for numeric verification")
    ap.add_argument("--dump-dir", default=None, help="where to write comparison PNGs")
    ap.add_argument("--tensor-output", action="store_true", help="force TensorType fp16 output")
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--layout", default="ane", choices=["ane", "std"],
                    help="ane: [B,C,1,S] conv1x1/einsum layout (ml-ane-transformers style, like Apple's "
                         "package); std: straight trace of the HF model ([B,S,C] linears)")
    args = ap.parse_args()

    W, H = args.width, args.height
    assert W % 14 == 0 and H % 14 == 0, "resolution must be a multiple of 14"
    print(f"=== DAv2-Small -> Core ML  {W}x{H}  (grid {W // 14}x{H // 14} = {(W // 14) * (H // 14)} patches)")

    torch.manual_seed(0)
    model = AutoModelForDepthEstimation.from_pretrained(HF_ID, attn_implementation=args.attn)
    model.eval()
    if args.layout == "ane":
        wrapper = DepthAnythingV2ANE(model, H, W).eval()
        print(f"  layout: ANE ([B,C,1,S], conv1x1 + per-head einsum), grid {wrapper.gh}x{wrapper.gw}")
    else:
        bake_position_embeddings(model, H, W)
        wrapper = DAv2Wrapper(model, H, W).eval()
        print("  layout: std (HF [B,S,C] linears)")

    example = torch.rand(1, 3, H, W) * 255.0
    with torch.no_grad():
        ref = wrapper(example)
    print("  eager output shape", tuple(ref.shape))

    print("  tracing ...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, check_trace=False)
        tr_out = traced(example)
    print("  trace-vs-eager max abs diff:", float((tr_out - ref).abs().max()))

    outputs = ([ct.TensorType(name="depth", dtype=np.float16)] if args.tensor_output
               else [ct.ImageType(name="depth", color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)])
    print("  converting ...")
    t0 = time.time()
    try:
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS14,
            inputs=[ct.ImageType(name="image", shape=(1, 3, H, W),
                                 color_layout=ct.colorlayout.BGR, scale=1.0)],
            outputs=outputs,
            compute_units=ct.ComputeUnit.ALL,
        )
        out_kind = "TensorType fp16" if args.tensor_output else "ImageType GRAYSCALE_FLOAT16"
    except Exception as e:  # noqa: BLE001
        if args.tensor_output:
            raise
        print("  !! GRAYSCALE_FLOAT16 image output failed, falling back to TensorType fp16:", repr(e))
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS14,
            inputs=[ct.ImageType(name="image", shape=(1, 3, H, W),
                                 color_layout=ct.colorlayout.BGR, scale=1.0)],
            outputs=[ct.TensorType(name="depth", dtype=np.float16)],
            compute_units=ct.ComputeUnit.ALL,
        )
        out_kind = "TensorType fp16 (fallback)"
    print(f"  converted in {time.time() - t0:.1f}s, output = {out_kind}")

    mlmodel.author = "Depth Anything V2 Small (HF depth-anything/Depth-Anything-V2-Small-hf), converted with coremltools"
    mlmodel.short_description = (f"Depth Anything V2 Small, {W}x{H} BGRA input, relative inverse depth "
                                 f"(larger = nearer) as {W}x{H} float16 grayscale. ImageNet normalization in-graph.")
    mlmodel.version = "1.0"
    mlmodel.input_description["image"] = f"{W}x{H} BGRA image, raw 0..255 (normalization inside the model)"
    mlmodel.output_description["depth"] = "Relative inverse depth, float16, larger = nearer"

    if os.path.exists(args.out):
        import shutil
        shutil.rmtree(args.out)
    mlmodel.save(args.out)
    print("  saved", args.out)
    from collections import Counter
    hist = Counter(op.op_type for op in mlmodel._mil_program.functions["main"].operations if op.op_type != "const")
    print("  MIL ops:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))

    # ---- numeric verification ----
    if args.photo:
        print("  verifying against PyTorch on", args.photo)
        pil_rgb, t_bgr = load_photo_bgr(args.photo, W, H)
        with torch.no_grad():
            pt = wrapper(t_bgr)[0, 0].numpy()
        ml = ct.models.MLModel(args.out, compute_units=ct.ComputeUnit.ALL)
        pred = ml.predict({"image": pil_rgb})["depth"]
        if isinstance(pred, Image.Image):
            cm = np.asarray(pred).astype(np.float32)
        else:
            cm = np.asarray(pred).astype(np.float32).reshape(H, W)
        corr = float(np.corrcoef(normalized(pt).ravel(), normalized(cm).ravel())[0, 1])
        rel_err = float(np.abs(pt - cm).mean() / (np.abs(pt).mean() + 1e-9))
        print(f"  pytorch range [{pt.min():.3f}, {pt.max():.3f}]  coreml range [{cm.min():.3f}, {cm.max():.3f}]")
        print(f"  correlation (per-image normalized) = {corr:.5f}   mean rel err = {rel_err:.4f}")
        status = "PASS" if corr > 0.99 else "FAIL"
        print(f"  numeric check: {status}")
        if args.dump_dir:
            os.makedirs(args.dump_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(args.out))[0]

            def to_u8(a):
                lo, hi = np.percentile(a, 2), np.percentile(a, 98)
                return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)

            Image.fromarray(to_u8(pt)).save(os.path.join(args.dump_dir, f"{base}_pytorch.png"))
            Image.fromarray(to_u8(cm)).save(os.path.join(args.dump_dir, f"{base}_coreml_py.png"))
            side = np.concatenate([np.asarray(pil_rgb), np.stack([to_u8(pt)] * 3, -1),
                                   np.stack([to_u8(cm)] * 3, -1)], axis=1)
            Image.fromarray(side).save(os.path.join(args.dump_dir, f"{base}_compare.png"))
            print("  wrote comparison PNGs to", args.dump_dir)
        with open(os.path.join(args.dump_dir or os.path.dirname(args.out), f"{base}_verify.txt"), "w") as f:
            f.write(f"{base}\ncorr={corr:.5f}\nrel_err={rel_err:.5f}\nstatus={status}\noutput={out_kind}\n")
        if status != "PASS":
            sys.exit(2)


if __name__ == "__main__":
    main()
