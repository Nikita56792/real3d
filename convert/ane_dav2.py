"""
ANE-friendly re-implementation of Depth Anything V2 (DINOv2-S/14 backbone + DPT neck/head)
in the "ml-ane-transformers" layout used by Apple's own DepthAnythingV2SmallF16.mlpackage:

  * tokens live in a [B, C, 1, S] tensor (channels first, sequence last)
  * every nn.Linear becomes a 1x1 nn.Conv2d
  * multi-head attention is done per head with einsum + softmax over the key axis
  * LayerNorm over the channel axis is written so coremltools fuses it into a single
    `layer_norm(axes=[1])` MIL op (fuse_layernorm_or_instancenorm pattern 5)
  * DINOv2 layer-scale (lambda1/lambda2) is folded into the out-proj / fc2 conv weights
  * the attention scale 1/sqrt(64) = 0.125 (exact in fp16) is folded into the q projection
  * positional embeddings are pre-interpolated (bicubic, fp32, same code path as HF) for the
    fixed target grid and stored as a constant

The DPT neck / head are the HF modules unchanged (they already operate on NCHW tensors).
Weights are copied from `transformers` DepthAnythingForDepthEstimation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _conv1x1_from_linear(lin: nn.Linear, out_scale: torch.Tensor | float | None = None) -> nn.Conv2d:
    """1x1 conv with the weights of `lin`; optional per-output-channel (or scalar) scale folded in."""
    w = lin.weight.detach().clone()          # [out, in]
    b = lin.bias.detach().clone() if lin.bias is not None else torch.zeros(w.shape[0])
    if out_scale is not None:
        s = torch.as_tensor(out_scale, dtype=w.dtype)
        if s.ndim == 1:
            w = w * s[:, None]
        else:
            w = w * s
        b = b * s
    conv = nn.Conv2d(w.shape[1], w.shape[0], kernel_size=1, bias=True)
    conv.weight.data.copy_(w.view(w.shape[0], w.shape[1], 1, 1))
    conv.bias.data.copy_(b)
    return conv


class LayerNormANE(nn.Module):
    """LayerNorm over dim 1 of a [B, C, 1, S] tensor, written exactly like ml-ane-transformers so
    coremltools fuses it into one layer_norm op. NOTE: bias is stored pre-divided by weight."""

    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        c = ln.weight.shape[0]
        self.eps = float(ln.eps)
        self.weight = nn.Parameter(ln.weight.detach().clone().view(1, c, 1, 1), requires_grad=False)
        self.bias = nn.Parameter((ln.bias.detach() / ln.weight.detach()).clone().view(1, c, 1, 1),
                                 requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_mean = x.mean(dim=1, keepdim=True)
        zero_mean = x - channels_mean
        zero_mean_sq = zero_mean * zero_mean
        denom = (zero_mean_sq.mean(dim=1, keepdim=True) + self.eps).rsqrt()
        out = zero_mean * denom
        return (out + self.bias) * self.weight


class AttentionANE(nn.Module):
    def __init__(self, hf_attention, lambda1: torch.Tensor, num_heads: int):
        super().__init__()
        sa = hf_attention.attention
        hidden = sa.query.weight.shape[0]
        self.num_heads = num_heads
        self.dim_per_head = hidden // num_heads
        scale = self.dim_per_head ** -0.5                     # 0.125 for DINOv2-S: exact in fp16
        self.q = _conv1x1_from_linear(sa.query, out_scale=scale)
        self.k = _conv1x1_from_linear(sa.key)
        self.v = _conv1x1_from_linear(sa.value)
        self.out = _conv1x1_from_linear(hf_attention.output.dense, out_scale=lambda1.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: [B, C, 1, S]
        q, k, v = self.q(x), self.k(x), self.v(x)
        mh_q = q.split(self.dim_per_head, dim=1)                       # each [B, 64, 1, S]
        mh_k = k.transpose(1, 3).split(self.dim_per_head, dim=3)       # each [B, S, 1, 64]
        mh_v = v.split(self.dim_per_head, dim=1)                       # each [B, 64, 1, S]
        aw = [torch.einsum("bchq,bkhc->bkhq", qi, ki) for qi, ki in zip(mh_q, mh_k)]  # [B, Sk, 1, Sq]
        aw = [w.softmax(dim=1) for w in aw]
        o = [torch.einsum("bkhq,bchk->bchq", wi, vi) for wi, vi in zip(aw, mh_v)]     # [B, 64, 1, Sq]
        return self.out(torch.cat(o, dim=1))


class MLPANE(nn.Module):
    def __init__(self, hf_mlp, lambda2: torch.Tensor):
        super().__init__()
        self.fc1 = _conv1x1_from_linear(hf_mlp.fc1)
        self.fc2 = _conv1x1_from_linear(hf_mlp.fc2, out_scale=lambda2.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class LayerANE(nn.Module):
    def __init__(self, hf_layer, num_heads: int):
        super().__init__()
        self.norm1 = LayerNormANE(hf_layer.norm1)
        self.attn = AttentionANE(hf_layer.attention, hf_layer.layer_scale1.lambda1, num_heads)
        self.norm2 = LayerNormANE(hf_layer.norm2)
        self.mlp = MLPANE(hf_layer.mlp, hf_layer.layer_scale2.lambda1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DepthAnythingV2ANE(nn.Module):
    """Input: [1,3,H,W] float, BGR channel order, raw 0..255. Output: [1,1,H,W] relative inverse depth."""

    def __init__(self, hf_model, height: int, width: int):
        super().__init__()
        cfg = hf_model.config
        bcfg = cfg.backbone_config
        backbone = hf_model.backbone
        self.patch = cfg.patch_size
        self.gh, self.gw = height // self.patch, width // self.patch
        self.hidden = bcfg.hidden_size
        n = self.gh * self.gw

        # (x/255 - mean)/std  ==  (x - 255*mean) * 1/(255*std); stats in BGR order.
        mean = torch.tensor(IMAGENET_MEAN[::-1]).view(1, 3, 1, 1) * 255.0
        inv_std = 1.0 / (torch.tensor(IMAGENET_STD[::-1]).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("mean255", mean)
        self.register_buffer("inv_std255", inv_std)

        # Patch embedding; BGR->RGB folded into the conv's input-channel order.
        pe = backbone.embeddings.patch_embeddings.projection
        self.patch_embed = nn.Conv2d(3, self.hidden, kernel_size=self.patch, stride=self.patch, bias=True)
        self.patch_embed.weight.data.copy_(pe.weight.detach()[:, [2, 1, 0]])
        self.patch_embed.bias.data.copy_(pe.bias.detach())

        # CLS token + pre-interpolated positional embeddings as [1, C, 1, 1+N].
        emb = backbone.embeddings
        with torch.no_grad():
            dummy = torch.zeros(1, 1 + n, self.hidden)
            pos = emb.interpolate_pos_encoding(dummy, height, width)   # [1, 1+N, C] (bicubic, fp32)
        assert pos.shape == (1, 1 + n, self.hidden), pos.shape
        self.register_buffer("pos", pos.detach().transpose(1, 2).reshape(1, self.hidden, 1, 1 + n).contiguous())
        self.register_buffer("cls", emb.cls_token.detach().reshape(1, self.hidden, 1, 1).contiguous())

        self.layers = nn.ModuleList(LayerANE(l, bcfg.num_attention_heads) for l in backbone.encoder.layer)
        # HF out_indices are 1-based stage numbers (stage k == output of encoder layer k-1).
        self.out_layers = [i - 1 for i in bcfg.out_indices]
        self.final_norm = LayerNormANE(backbone.layernorm) if bcfg.apply_layernorm else nn.Identity()

        self.neck = hf_model.neck
        self.head = hf_model.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean255) * self.inv_std255
        # Batch is fixed to 1 (no x.shape lookups: traced sizes trip coremltools' `int` converter).
        x = self.patch_embed(x)                                  # [1, C, gh, gw]
        x = x.reshape(1, self.hidden, 1, self.gh * self.gw)      # [1, C, 1, N]
        x = torch.cat([self.cls, x], dim=3)                      # [1, C, 1, 1+N]
        x = x + self.pos

        feats = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_layers:
                feats.append(x)

        maps = []
        for i, f in enumerate(feats):
            f = self.final_norm(f)[:, :, :, 1:]                  # drop CLS -> [B, C, 1, N]
            f = f.reshape(1, self.hidden, self.gh, self.gw)      # [1, C, gh, gw]
            f = self.neck.reassemble_stage.layers[i](f)          # projection + resize
            f = self.neck.convs[i](f)
            maps.append(f)
        fused = self.neck.fusion_stage(maps)
        depth = self.head(fused, self.gh, self.gw)               # [B, H, W]
        return depth.unsqueeze(1)                                # [B, 1, H, W]
