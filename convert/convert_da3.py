#!/usr/bin/env python
"""
Convert Depth Anything 3 Small (depth-anything/DA3-SMALL, monocular depth branch only) to a Core ML
.mlpackage with a fixed, non-square (16:9-friendly) input.

Adapted from john-rocky/CoreML-Models `conversion_scripts/convert_depth_anything_v3.py`
(coremltools `int` patch, meshgrid-free UV grid, frozen pos-embed / RoPE, in-place cam-token fix),
generalised to W != H and to the same I/O contract as the DAv2 packages in this folder:
  input  'image': 32BGRA image (W x H), raw 0..255, ImageNet normalisation in-graph
  output 'depth': GRAYSCALE_FLOAT16 image (W x H)  -- NOTE: DA3 outputs *depth* (larger = farther)

Needs the DA3 repo at ./DepthAnythingV3 (auto-cloned if missing).

Usage:
  python convert_da3.py --width 644 --height 364 --out /path/DA3Small_644x364.mlpackage --photo photo.png
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from coremltools.converters.mil.frontend.torch import ops as _ct_ops
from coremltools.converters.mil import Builder as mb
from PIL import Image

HERE = Path(__file__).resolve().parent
DA3_REPO = HERE / "DepthAnythingV3"
if not (DA3_REPO / "src").exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ByteDance-Seed/Depth-Anything-3",
                    str(DA3_REPO)], check=True)
sys.path.insert(0, str(DA3_REPO / "src"))

from depth_anything_3.cfg import create_object, load_config  # noqa: E402
from depth_anything_3.registry import MODEL_REGISTRY  # noqa: E402
from depth_anything_3.model.da3 import DepthAnything3Net  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.torch import load_file as load_safetensors  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --- coremltools 9.0: `int` op on multi-dim shape casts (community patch) -------------------------
def _patched_int(context, node):
    inputs = _ct_ops._get_inputs(context, node)
    x = inputs[0]
    if x.val is not None:
        val = x.val
        if isinstance(val, np.ndarray):
            val = int(val.item()) if val.ndim == 0 else int(val.flat[0])
        else:
            val = int(val)
        res = mb.const(val=np.int32(val), name=node.name)
    else:
        res = mb.cast(x=x, dtype="int32", name=node.name)
    context.add(res)


_ct_ops._TORCH_OPS_REGISTRY.register_func(_patched_int, torch_alias=["int"], override=True)


# --- trace-friendly UV grid (torch.meshgrid trips coremltools) ------------------------------------
def _coreml_safe_create_uv_grid(width, height, aspect_ratio=None, dtype=None, device=None):
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag_factor = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)
    uu = x_coords.view(1, width).expand(height, width)
    vv = y_coords.view(height, 1).expand(height, width)
    return torch.stack((uu, vv), dim=-1)


def patch_head_utils():
    from depth_anything_3.model.utils import head_utils as _hu
    from depth_anything_3.model import dualdpt as _dualdpt
    _hu.create_uv_grid = _coreml_safe_create_uv_grid
    _dualdpt.create_uv_grid = _coreml_safe_create_uv_grid


# --- freeze pos-embed (bicubic, fp32) and RoPE positions for the fixed grid -----------------------
def freeze_pos_embed(backbone, height: int, width: int) -> None:
    dummy = torch.zeros(1, 3, height, width)
    tokens = backbone.patch_embed(dummy)
    cls = backbone.cls_token.expand(tokens.shape[0], -1, -1)
    x = torch.cat([cls, tokens], dim=1)
    with torch.no_grad():
        # prepare_tokens_with_masks unpacks (B,S,c,h,w) as `w, h` and calls (x, w, h) -> pass (H, W)
        pos = backbone.interpolate_pos_encoding(x, height, width)
    assert pos.shape[1] == x.shape[1], (pos.shape, x.shape)
    backbone.register_buffer("_frozen_pos_embed", pos.detach().clone(), persistent=False)
    backbone.interpolate_pos_encoding = types.MethodType(lambda self, x, w, h: self._frozen_pos_embed, backbone)
    print(f"  frozen pos-embed {tuple(pos.shape)} for grid {height // backbone.patch_size}x{width // backbone.patch_size}")


def freeze_rope_positions(backbone, height: int, width: int) -> None:
    if backbone.rope is None:
        return
    with torch.no_grad():
        pos, pos_nodiff = backbone._prepare_rope(1, 1, height, width, torch.device("cpu"))  # eager: fine
    backbone.register_buffer("_frozen_rope_pos", pos.detach().clone(), persistent=False)
    backbone.register_buffer("_frozen_rope_pos_nodiff", pos_nodiff.detach().clone(), persistent=False)
    backbone._prepare_rope = types.MethodType(
        lambda self, B, S, H, W, device: (self._frozen_rope_pos, self._frozen_rope_pos_nodiff), backbone)
    print(f"  frozen RoPE positions {tuple(pos.shape)}")


# --- S=1 backbone forward without the in-place camera-token write --------------------------------
def patch_backbone_forward(backbone):
    def _patched(self, x, n=1, export_feat_layers=[], **kwargs):
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len, aux_output = [], len(self.blocks), []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)
        local_x = x
        for i, blk in enumerate(self.blocks):
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos, l_pos = pos_nodiff, pos
            if self.alt_start != -1 and i == self.alt_start:
                cam_token = self.camera_token[:, :1].expand(1, -1, -1).unsqueeze(2)  # (1, 1, 1, C)
                x = torch.cat([cam_token, x[:, :, 1:]], dim=2)
            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(x, blk, "global", pos=g_pos, attn_mask=None)
            else:
                x = self.process_attention(x, blk, "local", pos=l_pos)
                local_x = x
            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(x)
        return output, aux_output

    backbone._get_intermediate_layers_not_chunked = types.MethodType(_patched, backbone)


class MonoDepthWrapper(nn.Module):
    """[1,3,H,W] BGR 0..255 -> [1,1,H,W] depth (larger = farther)."""

    def __init__(self, net: DepthAnything3Net, height: int, width: int):
        super().__init__()
        self.net = net
        self.h, self.w = height, width
        mean = torch.tensor(IMAGENET_MEAN[::-1]).view(1, 3, 1, 1) * 255.0       # BGR order
        inv_std = 1.0 / (torch.tensor(IMAGENET_STD[::-1]).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("mean255", mean, persistent=False)
        self.register_buffer("inv_std255", inv_std, persistent=False)
        # Fold BGR->RGB into the patch-embedding conv's input channels (exact).
        proj = net.backbone.pretrained.patch_embed.proj
        with torch.no_grad():
            proj.weight.copy_(proj.weight[:, [2, 1, 0]].clone())

    def forward(self, image: torch.Tensor):
        x = (image - self.mean255) * self.inv_std255      # == (x/255 - mean)/std
        x = x.unsqueeze(1)                                # (1, S=1, 3, H, W)
        feats, _ = self.net.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy="first")
        out = self.net.head(feats, self.h, self.w, patch_start_idx=0)
        return out.depth.squeeze(1).unsqueeze(1)          # (1, 1, H, W)


def load_net(repo_id="depth-anything/DA3-SMALL", config_name="da3-small") -> DepthAnything3Net:
    net = create_object(load_config(MODEL_REGISTRY[config_name]))
    net.eval()
    ckpt = hf_hub_download(repo_id, filename="model.safetensors")
    state = load_safetensors(ckpt)
    if any(k.startswith("model.") for k in state):
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"  loaded {ckpt}  missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("   missing (first 5):", missing[:5])
    return net


def normalized(a):
    a = a.astype(np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=644)
    ap.add_argument("--height", type=int, default=364)
    ap.add_argument("--out", required=True)
    ap.add_argument("--photo", default=None)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--tensor-output", action="store_true")
    args = ap.parse_args()
    W, H = args.width, args.height
    assert W % 14 == 0 and H % 14 == 0
    print(f"=== DA3-Small -> Core ML {W}x{H} (grid {W // 14}x{H // 14} = {(W // 14) * (H // 14)} patches)")

    net = load_net()
    backbone = net.backbone.pretrained
    patch_head_utils()
    freeze_pos_embed(backbone, H, W)
    freeze_rope_positions(backbone, H, W)
    patch_backbone_forward(backbone)

    # Reference from the *unmodified* forward path (RGB, normalised) before the BGR fold, for a sanity check.
    ref_rgb_depth = None
    if args.photo:
        img = Image.open(args.photo).convert("RGB").resize((W, H), Image.BICUBIC)
        rgb = torch.from_numpy(np.asarray(img).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        xn = ((rgb - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)).unsqueeze(1)
        with torch.no_grad():
            feats, _ = net.backbone(xn, cam_token=None, export_feat_layers=[], ref_view_strategy="first")
            ref_rgb_depth = net.head(feats, H, W, patch_start_idx=0).depth[0, 0].numpy()

    wrapper = MonoDepthWrapper(net, H, W).eval()
    example = torch.rand(1, 3, H, W) * 255.0
    with torch.no_grad():
        ref = wrapper(example)
        print(f"  eager out {tuple(ref.shape)} range [{ref.min():.3f}, {ref.max():.3f}]")
        traced = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
        tr = traced(example)
    print("  trace-vs-eager max abs diff:", float((tr - ref).abs().max()))

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
            print("  !! image output failed, falling back to tensor:", repr(e))
            mlmodel = convert([ct.TensorType(name="depth", dtype=np.float16)]); out_kind = "TensorType fp16 (fallback)"
    print(f"  converted in {time.time() - t0:.1f}s, output = {out_kind}")
    mlmodel.author = "Depth Anything 3 Small (depth-anything/DA3-SMALL), monocular depth branch, coremltools"
    mlmodel.short_description = (f"Depth Anything 3 Small, {W}x{H} BGRA input; output is DEPTH (larger = farther) "
                                 f"as {W}x{H} float16 grayscale. ImageNet normalisation in-graph.")
    mlmodel.version = "1.0"
    mlmodel.input_description["image"] = f"{W}x{H} BGRA image, raw 0..255"
    mlmodel.output_description["depth"] = "Depth (larger = farther), float16"
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    mlmodel.save(args.out)
    print("  saved", args.out)
    from collections import Counter
    hist = Counter(op.op_type for op in mlmodel._mil_program.functions["main"].operations if op.op_type != "const")
    print("  MIL ops:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))

    if args.photo:
        img = Image.open(args.photo).convert("RGB").resize((W, H), Image.BICUBIC)
        bgr = np.asarray(img).astype(np.float32)[..., ::-1].copy()
        t = torch.from_numpy(bgr).permute(2, 0, 1).unsqueeze(0).contiguous()
        with torch.no_grad():
            pt = wrapper(t)[0, 0].numpy()
        c0 = float(np.corrcoef(normalized(pt).ravel(), normalized(ref_rgb_depth).ravel())[0, 1])
        print(f"  wrapper(BGR) vs stock DA3(RGB) correlation = {c0:.5f}  max abs diff = {np.abs(pt - ref_rgb_depth).max():.4f}")
        ml = ct.models.MLModel(args.out, compute_units=ct.ComputeUnit.ALL)
        pred = ml.predict({"image": img})["depth"]
        cm = np.asarray(pred).astype(np.float32).reshape(H, W)
        corr = float(np.corrcoef(normalized(pt).ravel(), normalized(cm).ravel())[0, 1])
        rel = float(np.abs(pt - cm).mean() / (np.abs(pt).mean() + 1e-9))
        print(f"  pytorch range [{pt.min():.3f}, {pt.max():.3f}]  coreml range [{cm.min():.3f}, {cm.max():.3f}]  "
              f"finite={np.isfinite(cm).all()}")
        print(f"  correlation (per-image normalized) = {corr:.5f}   mean rel err = {rel:.4f}")
        status = "PASS" if corr > 0.99 and np.isfinite(cm).all() else "FAIL"
        print(f"  numeric check: {status}")
        base = os.path.splitext(os.path.basename(args.out))[0]
        dump_dir = args.dump_dir or os.path.dirname(args.out)
        os.makedirs(dump_dir, exist_ok=True)

        def to_u8(a):  # white = nearer for a depth map: invert
            inv = 1.0 / np.maximum(a, 1e-6)
            lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
            return (np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)

        side = np.concatenate([np.asarray(img), np.stack([to_u8(pt)] * 3, -1), np.stack([to_u8(cm)] * 3, -1)], axis=1)
        Image.fromarray(side).save(os.path.join(dump_dir, f"{base}_compare.png"))
        with open(os.path.join(dump_dir, f"{base}_verify.txt"), "w") as f:
            f.write(f"{base}\ncorr={corr:.5f}\nrel_err={rel:.5f}\nstatus={status}\noutput={out_kind}\n"
                    f"polarity=depth (larger = farther)\n")
        print("  wrote", os.path.join(dump_dir, f"{base}_compare.png"), "(depth shown inverted: white = nearer)")
        if status != "PASS":
            sys.exit(2)


if __name__ == "__main__":
    main()
