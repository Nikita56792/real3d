#!/usr/bin/env python
"""
Depth Anything V2 Small -> Core ML, with a choice of encoder layout / deployment target and the option to
cut the network into a STAGE-A model ("enc", meant for the GPU) and a STAGE-B model ("head", meant for the
ANE) for pipeline parallelism: both stages run concurrently on consecutive frames.

Cut point  --cut K  (1..12, default 12): stage A runs the image preprocessing + DINOv2-S blocks 1..K,
stage B runs blocks K+1..12 (if any) + the DPT neck/head.

Parts
  full : image (BGRA WxH) -> depth (GRAYSCALE_FLOAT16 WxH)                          [same contract as bundled]
  enc  : image (BGRA WxH) -> featN (TensorType float16 [1, 384, H/14, W/14]) for each DPT tap N*3 <= K
                             + tokens (TensorType float16 [1, 384, 1, 1 + (H/14)*(W/14)]) if K < 12
  head : the outputs of `enc` with the same --cut -> depth (GRAYSCALE_FLOAT16 WxH)

featN (N=1..4) = DINOv2-S hidden state after block 3N, passed through the backbone's final LayerNorm, CLS token
dropped, reshaped to NCHW [1, 384, gh, gw] (row-major patch grid) -- exactly what the HF DPT neck consumes.
tokens = raw residual stream after block K (no LayerNorm), channels-first [1, 384, 1, S], S = 1 + gh*gw, CLS
token at index 0 followed by the row-major patch grid.

Layouts (--layout for stage A / full, --head-layout for the blocks inside stage B)
  ane  : ml-ane-transformers layout from ane_dav2.py ([B,C,1,S], conv1x1, per-head einsum)
  sdpa : [B,S,C] nn.Linear + torch.nn.functional.scaled_dot_product_attention WITHOUT an explicit `scale`
         (so coremltools lowers it to the fused MIL `scaled_dot_product_attention` op when the deployment
         target is >= macOS15 / iOS18); layer-scale folded into the out-proj / fc2 weights.

Usage
  python dav2_split.py --check-pt                                         (rewritten torch graphs vs HF)
  python dav2_split.py --part full --layout sdpa --target macOS15 --out ../models/custom/X.mlpackage
  python dav2_split.py --part enc  --cut 11 --layout sdpa --target macOS15 --out ../models/custom/X_enc.mlpackage
  python dav2_split.py --part head --cut 11 --head-layout ane --target macOS15 --out ../models/custom/X_head.mlpackage
  python dav2_split.py --cut 11 --verify-chain X_enc.mlpackage X_head.mlpackage [--units CPU_AND_GPU CPU_AND_NE]
  python dav2_split.py --verify-full X.mlpackage [--units CPU_AND_GPU]
"""
import argparse
import os
import shutil
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from PIL import Image
from transformers import AutoModelForDepthEstimation

from ane_dav2 import IMAGENET_MEAN, IMAGENET_STD, LayerANE, LayerNormANE

HF_ID = "depth-anything/Depth-Anything-V2-Small-hf"
HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = [os.path.join(HERE, "test_street.png"), os.path.join(HERE, "test_indoor.png")]
NUM_BLOCKS = 12
TAPS = [2, 5, 8, 11]          # 0-based block indices feeding the DPT neck (HF out_indices 3,6,9,12)


# ----------------------------------------------------------------------------------------------------------
# DINOv2 blocks
# ----------------------------------------------------------------------------------------------------------
def _linear_scaled(lin: nn.Linear, scale: torch.Tensor | None = None) -> nn.Linear:
    out = nn.Linear(lin.in_features, lin.out_features, bias=True)
    w = lin.weight.detach().clone()
    b = lin.bias.detach().clone() if lin.bias is not None else torch.zeros(lin.out_features)
    if scale is not None:
        w = w * scale.detach()[:, None]
        b = b * scale.detach()
    out.weight.data.copy_(w)
    out.bias.data.copy_(b)
    return out


class LayerSDPA(nn.Module):
    """DINOv2 block in [1, S, C] layout with fused-SDPA-friendly attention."""

    def __init__(self, hf_layer, num_heads: int, seq: int):
        super().__init__()
        a = hf_layer.attention
        self.heads = num_heads
        self.dim = a.attention.query.weight.shape[0]
        self.hd = self.dim // num_heads
        self.seq = seq
        self.norm1 = hf_layer.norm1
        self.q = _linear_scaled(a.attention.query)
        self.k = _linear_scaled(a.attention.key)
        self.v = _linear_scaled(a.attention.value)
        self.o = _linear_scaled(a.output.dense, hf_layer.layer_scale1.lambda1)
        self.norm2 = hf_layer.norm2
        self.fc1 = _linear_scaled(hf_layer.mlp.fc1)
        self.fc2 = _linear_scaled(hf_layer.mlp.fc2, hf_layer.layer_scale2.lambda1)

    def forward(self, x):                                           # x: [1, S, C]
        h = self.norm1(x)
        q = self.q(h).reshape(1, self.seq, self.heads, self.hd).transpose(1, 2)
        k = self.k(h).reshape(1, self.seq, self.heads, self.hd).transpose(1, 2)
        v = self.v(h).reshape(1, self.seq, self.heads, self.hd).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)                  # default scale = 1/sqrt(64), no `scale=`
        h = h.transpose(1, 2).reshape(1, self.seq, self.dim)
        x = x + self.o(h)
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


class AttentionANEChunked(nn.Module):
    """ane_dav2.AttentionANE with the query axis processed in `chunks` pieces (softmax is over keys, so this is
    exactly equivalent) -- smaller [B, S, 1, S/chunks] attention tensors for the ANE."""

    def __init__(self, base, chunks: int):
        super().__init__()
        self.q, self.k, self.v, self.out = base.q, base.k, base.v, base.out
        self.dim_per_head, self.chunks = base.dim_per_head, chunks

    def forward(self, x):                                           # x: [B, C, 1, S]
        q, k, v = self.q(x), self.k(x), self.v(x)
        S = x.shape[3]
        step = -(-S // self.chunks)
        mh_q = q.split(self.dim_per_head, dim=1)
        mh_k = k.transpose(1, 3).split(self.dim_per_head, dim=3)
        mh_v = v.split(self.dim_per_head, dim=1)
        outs = []
        for qi, ki, vi in zip(mh_q, mh_k, mh_v):
            parts = []
            for qc in qi.split(step, dim=3):
                w = torch.einsum("bchq,bkhc->bkhq", qc, ki).softmax(dim=1)
                parts.append(torch.einsum("bkhq,bchk->bchq", w, vi))
            outs.append(torch.cat(parts, dim=3) if len(parts) > 1 else parts[0])
        return self.out(torch.cat(outs, dim=1))


class MLPChunked(nn.Module):
    """Token-wise MLP applied to `chunks` pieces of the sequence axis of [B, C, 1, S] (exactly equivalent)."""

    def __init__(self, mlp, chunks: int):
        super().__init__()
        self.mlp, self.chunks = mlp, chunks

    def forward(self, x):
        step = -(-x.shape[3] // self.chunks)
        return torch.cat([self.mlp(c) for c in x.split(step, dim=3)], dim=3)


ANE_QCHUNKS = 1
ANE_MLPCHUNKS = 1


class Blocks(nn.Module):
    """Runs DINOv2 blocks [start, stop) in the given layout on a channels-first [1, C, 1, S] token tensor.
    Returns (tokens_out [1,C,1,S], list of NCHW DPT features for taps inside the range)."""

    def __init__(self, hf, height, width, layout, start, stop):
        super().__init__()
        bcfg, bb = hf.config.backbone_config, hf.backbone
        self.layout, self.start, self.stop = layout, start, stop
        self.gh, self.gw = height // 14, width // 14
        self.C = bcfg.hidden_size
        self.S = 1 + self.gh * self.gw
        if layout == "ane":
            self.layers = nn.ModuleList(LayerANE(bb.encoder.layer[i], bcfg.num_attention_heads) for i in range(start, stop))
            if ANE_QCHUNKS > 1:
                for l in self.layers:
                    l.attn = AttentionANEChunked(l.attn, ANE_QCHUNKS)
            if ANE_MLPCHUNKS > 1:
                for l in self.layers:
                    l.mlp = MLPChunked(l.mlp, ANE_MLPCHUNKS)
            self.final_norm = LayerNormANE(bb.layernorm)
        else:
            self.layers = nn.ModuleList(LayerSDPA(bb.encoder.layer[i], bcfg.num_attention_heads, self.S)
                                        for i in range(start, stop))
            self.final_norm = bb.layernorm

    def forward(self, x):                                            # x: [1, C, 1, S]
        feats = []
        if self.layout == "sdpa":
            x = x.reshape(1, self.C, self.S).transpose(1, 2)          # [1, S, C]
        for j, layer in enumerate(self.layers):
            x = layer(x)
            if self.start + j in TAPS:
                if self.layout == "ane":
                    f = self.final_norm(x)[:, :, :, 1:]
                else:
                    f = self.final_norm(x)[:, 1:, :].transpose(1, 2)
                feats.append(f.reshape(1, self.C, self.gh, self.gw))
        if self.layout == "sdpa":
            x = x.transpose(1, 2).reshape(1, self.C, 1, self.S)
        return x, feats


class Embed(nn.Module):
    """[1,3,H,W] BGR 0..255 -> [1, C, 1, S] tokens (CLS first, + positional embeddings)."""

    def __init__(self, hf, height, width):
        super().__init__()
        bb = hf.backbone
        self.C = hf.config.backbone_config.hidden_size
        self.gh, self.gw = height // 14, width // 14
        n = self.gh * self.gw
        self.register_buffer("mean255", torch.tensor(IMAGENET_MEAN[::-1]).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("inv_std255", 1.0 / (torch.tensor(IMAGENET_STD[::-1]).view(1, 3, 1, 1) * 255.0))
        pe = bb.embeddings.patch_embeddings.projection
        self.patch_embed = nn.Conv2d(3, self.C, kernel_size=14, stride=14, bias=True)
        self.patch_embed.weight.data.copy_(pe.weight.detach()[:, [2, 1, 0]])      # BGR input folded in
        self.patch_embed.bias.data.copy_(pe.bias.detach())
        with torch.no_grad():
            pos = bb.embeddings.interpolate_pos_encoding(torch.zeros(1, 1 + n, self.C), height, width)
        self.register_buffer("pos", pos.transpose(1, 2).reshape(1, self.C, 1, 1 + n).contiguous())
        self.register_buffer("cls", bb.embeddings.cls_token.detach().reshape(1, self.C, 1, 1).contiguous())

    def forward(self, x):
        x = self.patch_embed((x - self.mean255) * self.inv_std255)
        x = x.reshape(1, self.C, 1, self.gh * self.gw)
        return torch.cat([self.cls, x], dim=3) + self.pos


class StageA(nn.Module):
    def __init__(self, hf, height, width, layout, cut):
        super().__init__()
        self.cut = cut
        self.embed = Embed(hf, height, width)
        self.blocks = Blocks(hf, height, width, layout, 0, cut)

    def forward(self, img):
        x, feats = self.blocks(self.embed(img))
        return tuple(feats) + ((x,) if self.cut < NUM_BLOCKS else ())


class DPT(nn.Module):
    def __init__(self, hf, height, width):
        super().__init__()
        self.gh, self.gw = height // 14, width // 14
        self.neck, self.head = hf.neck, hf.head

    def forward(self, feats):
        maps = [self.neck.convs[i](self.neck.reassemble_stage.layers[i](f)) for i, f in enumerate(feats)]
        return self.head(self.neck.fusion_stage(maps), self.gh, self.gw).unsqueeze(1)   # [1, 1, H, W]


class StageB(nn.Module):
    def __init__(self, hf, height, width, head_layout, cut):
        super().__init__()
        self.cut = cut
        self.n_in_feats = sum(1 for t in TAPS if t < cut)
        self.blocks = Blocks(hf, height, width, head_layout, cut, NUM_BLOCKS) if cut < NUM_BLOCKS else None
        self.dpt = DPT(hf, height, width)

    def forward(self, *inputs):
        feats = list(inputs[:self.n_in_feats])
        if self.blocks is not None:
            _, more = self.blocks(inputs[self.n_in_feats])
            feats += more
        return self.dpt(feats)


class Full(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b

    def forward(self, x):
        return self.b(*self.a(x))


class HFReference(nn.Module):
    """Stock HF model driven with BGR 0..255 input (the numeric reference)."""

    def __init__(self, hf):
        super().__init__()
        self.m = hf
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, bgr):
        x = (bgr[:, [2, 1, 0]] / 255.0 - self.mean) / self.std
        return self.m(pixel_values=x).predicted_depth.unsqueeze(1)


def load_hf():
    return AutoModelForDepthEstimation.from_pretrained(HF_ID, attn_implementation="eager").eval()


def io_names(cut):
    feats = [f"feat{i + 1}" for i, t in enumerate(TAPS) if t < cut]
    return feats + (["tokens"] if cut < NUM_BLOCKS else [])


def photo_bgr(path, W, H):
    img = Image.open(path).convert("RGB").resize((W, H), Image.BICUBIC)
    bgr = np.asarray(img).astype(np.float32)[..., ::-1].copy()
    return img, torch.from_numpy(bgr).permute(2, 0, 1).unsqueeze(0).contiguous()


def corr(a, b):
    return float(np.corrcoef(a.astype(np.float64).ravel(), b.astype(np.float64).ravel())[0, 1])


def as_depth(pred, W, H):
    if isinstance(pred, Image.Image):
        return np.asarray(pred).astype(np.float32)
    return np.asarray(pred).astype(np.float32).reshape(H, W)


# ----------------------------------------------------------------------------------------------------------
def convert(args):
    W, H, K = args.width, args.height, args.cut
    gh, gw, S = H // 14, W // 14, 1 + (H // 14) * (W // 14)
    hf = load_hf()
    target = getattr(ct.target, args.target)
    img_in = ct.ImageType(name="image", shape=(1, 3, H, W), color_layout=ct.colorlayout.BGR, scale=1.0)
    depth_out = ct.ImageType(name="depth", color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)
    names = io_names(K)
    shapes = {n: ((1, 384, 1, S) if n == "tokens" else (1, 384, gh, gw)) for n in names}
    img_example = (torch.rand(1, 3, H, W) * 255,)
    if args.part == "full":
        module = Full(StageA(hf, H, W, args.layout, NUM_BLOCKS), StageB(hf, H, W, args.head_layout, NUM_BLOCKS))
        example, inputs, outputs = img_example, [img_in], [depth_out]
    elif args.part == "enc":
        module = StageA(hf, H, W, args.layout, K)
        example, inputs = img_example, [img_in]
        outputs = [ct.TensorType(name=n, dtype=np.float16) for n in names]
    else:
        module = StageB(hf, H, W, args.head_layout, K)
        example = tuple(torch.randn(*shapes[n]) for n in names)
        inputs = [ct.TensorType(name=n, shape=shapes[n], dtype=np.float16) for n in names]
        outputs = [depth_out]
    module = module.eval()
    print(f"=== part={args.part} cut={K} layout={args.layout} head_layout={args.head_layout} target={args.target} "
          f"{W}x{H} grid {gw}x{gh}")
    with torch.no_grad():
        traced = torch.jit.trace(module, example, check_trace=False)
    t0 = time.time()
    ml = ct.convert(traced, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=target, inputs=inputs, outputs=outputs,
                    compute_units=ct.ComputeUnit.ALL)
    print(f"  converted in {time.time() - t0:.1f}s")
    ml.author = "Depth Anything V2 Small (HF depth-anything/Depth-Anything-V2-Small-hf), converted with coremltools"
    io_doc = {n: (f"float16 [1,384,1,{S}]: DINOv2-S residual stream after block {K} (no norm), channels-first, "
                  f"CLS at index 0 then row-major {gw}x{gh} patch grid" if n == "tokens" else
                  f"float16 [1,384,{gh},{gw}]: DINOv2-S block {3 * int(n[-1])} output, final LayerNorm, no CLS, NCHW")
              for n in names}
    desc = {
        "full": f"DAv2-Small {W}x{H}: BGRA image -> relative inverse depth (larger = nearer), float16 grayscale.",
        "enc": f"DAv2-Small {W}x{H} STAGE A (cut after block {K}): BGRA image -> {', '.join(names)}. "
               f"Feed all outputs by name to the matching stage-B (_head) model.",
        "head": f"DAv2-Small {W}x{H} STAGE B (blocks {K + 1}..12 + DPT head): {', '.join(names)} from the matching "
                f"stage-A (_enc) model -> relative inverse depth (larger = nearer), float16 grayscale {W}x{H}.",
    }[args.part]
    ml.short_description = desc + f" [layout={args.layout if args.part != 'head' else args.head_layout}, target={args.target}]"
    ml.version = "1.1"
    if args.part != "head":
        ml.input_description["image"] = f"{W}x{H} BGRA image, raw 0..255 (normalization inside the model)"
    else:
        for n in names:
            ml.input_description[n] = io_doc[n]
    if args.part == "enc":
        for n in names:
            ml.output_description[n] = io_doc[n]
    else:
        ml.output_description["depth"] = "Relative inverse depth, float16, larger = nearer"
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    ml.save(args.out)
    hist = Counter(op.op_type for op in ml._mil_program.functions["main"].operations if op.op_type != "const")
    print("  saved", args.out)
    print("  MIL ops:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))


def verify(args):
    W, H, K = args.width, args.height, args.cut
    hf = load_hf()
    ref = HFReference(hf).eval()
    units = [getattr(ct.ComputeUnit, u) for u in args.units]
    if args.verify_chain:
        a_ml = ct.models.MLModel(args.verify_chain[0], compute_units=units[0])
        b_ml = ct.models.MLModel(args.verify_chain[1], compute_units=units[-1])
        a_pt = StageA(hf, H, W, "sdpa", K).eval()
        names = io_names(K)
        print(f"  chain {os.path.basename(args.verify_chain[0])} [{units[0]}] -> "
              f"{os.path.basename(args.verify_chain[1])} [{units[-1]}]")
    else:
        full_ml = ct.models.MLModel(args.verify_full, compute_units=units[0])
        print(f"  full {os.path.basename(args.verify_full)} [{units[0]}]")
    ok = True
    for p in PHOTOS:
        pil, t = photo_bgr(p, W, H)
        with torch.no_grad():
            pt = ref(t)[0, 0].numpy()
        extra = ""
        if args.verify_chain:
            outs = a_ml.predict({"image": pil})
            with torch.no_grad():
                a_ref = a_pt(t)
            fc = [corr(np.asarray(outs[n]), a_ref[i].numpy()) for i, n in enumerate(names)]
            cm = as_depth(b_ml.predict({n: np.asarray(outs[n]).astype(np.float16) for n in names})["depth"], W, H)
            extra = " | stage-A outputs corr " + " ".join(f"{n}={c:.5f}" for n, c in zip(names, fc))
        else:
            cm = as_depth(full_ml.predict({"image": pil})["depth"], W, H)
        c = corr(pt, cm)
        rel = float(np.abs(pt - cm).mean() / (np.abs(pt).mean() + 1e-9))
        ok &= c > 0.999
        print(f"  {os.path.basename(p)}: depth corr={c:.5f} mean-rel-err={rel:.5f} "
              f"pt[{pt.min():.3f},{pt.max():.3f}] ml[{cm.min():.3f},{cm.max():.3f}]{extra}")
    print("  numeric check:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=812)
    ap.add_argument("--height", type=int, default=462)
    ap.add_argument("--part", choices=["full", "enc", "head"], default="full")
    ap.add_argument("--cut", type=int, default=12, help="stage A runs blocks 1..cut, stage B the rest + DPT")
    ap.add_argument("--layout", choices=["ane", "sdpa"], default="sdpa", help="block layout of stage A / full")
    ap.add_argument("--head-layout", choices=["ane", "sdpa"], default="ane", help="block layout inside stage B")
    ap.add_argument("--target", default="macOS15")
    ap.add_argument("--out")
    ap.add_argument("--verify-chain", nargs=2, metavar=("ENC", "HEAD"))
    ap.add_argument("--verify-full", metavar="FULL")
    ap.add_argument("--units", nargs="+", default=["CPU_AND_GPU"],
                    help="compute units for verification (chain: stage A then stage B)")
    ap.add_argument("--ane-qchunks", type=int, default=1, help="ane layout: process queries in N chunks")
    ap.add_argument("--ane-mlpchunks", type=int, default=1, help="ane layout: run the MLP on N sequence chunks")
    ap.add_argument("--check-pt", action="store_true", help="check the rewritten PyTorch graphs vs HF")
    args = ap.parse_args()
    global ANE_QCHUNKS, ANE_MLPCHUNKS
    ANE_QCHUNKS, ANE_MLPCHUNKS = args.ane_qchunks, args.ane_mlpchunks
    assert args.width % 14 == 0 and args.height % 14 == 0 and 1 <= args.cut <= NUM_BLOCKS
    W, H = args.width, args.height
    if args.check_pt:
        hf = load_hf()
        ref = HFReference(hf).eval()
        for la, lb in (("sdpa", "ane"), ("ane", "sdpa"), ("sdpa", "sdpa"), ("ane", "ane")):
            m = Full(StageA(hf, H, W, la, args.cut), StageB(hf, H, W, lb, args.cut)).eval()
            for p in PHOTOS:
                _, t = photo_bgr(p, W, H)
                with torch.no_grad():
                    a, b = ref(t), m(t)
                print(f"  torch check cut={args.cut} A={la} B={lb} {os.path.basename(p)}: "
                      f"max abs diff {float((a - b).abs().max()):.2e} (depth max {float(a.max()):.3f})")
    if args.verify_chain or args.verify_full:
        sys.exit(0 if verify(args) else 2)
    if args.out:
        convert(args)


if __name__ == "__main__":
    main()
