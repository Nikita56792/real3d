"""Probe the community MoGe-2 ViT-S Core ML package: I/O, GPU latency, and whether metric depth needs recover_focal_shift."""
# Usage: python moge2_coreml_probe.py <moge_vits_normal.mlpackage> <photo1.png> [<photo2.png> ...]
#   package: https://huggingface.co/ebowwa/moge-2-vits-normal-coreml (branch fp16, traced 1x3x512x768, macOS 15)
import os, sys, time, numpy as np, coremltools as ct
from PIL import Image

MODEL = sys.argv[1]
PHOTOS = sys.argv[2:] or ["test_street.png", "test_indoor.png"]
W, H = 768, 512

def uv_grid(width, height):
    ar = width / height; diag = (ar ** 2 + 1) ** 0.5
    sx, sy = ar / diag, 1 / diag
    x = np.linspace(-sx * (width - 1) / width, sx * (width - 1) / width, width, dtype=np.float32)
    y = np.linspace(-sy * (height - 1) / height, sy * (height - 1) / height, height, dtype=np.float32)
    return np.stack(np.meshgrid(x, y, indexing="xy"), -1)  # H,W,2

def solve_focal_shift(uv, xyz):
    """MoGe geometry_numpy.solve_optimal_focal_shift: min_shift,focal |focal * xy/(z+shift) - uv|^2.
    focal is closed-form given shift, so the problem is 1-D in shift -> golden-section (no scipy needed)."""
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)
    def cost(shift):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        return float(np.square(f * xy_proj - uv).sum())
    lo, hi = -float(z.min()) + 1e-3, 20.0
    g = (5 ** 0.5 - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = cost(a), cost(b)
    for _ in range(80):
        if fa < fb: hi, b, fb = b, a, fa; a = hi - g * (hi - lo); fa = cost(a)
        else: lo, a, fa = a, b, fb; b = lo + g * (hi - lo); fb = cost(b)
    shift = (lo + hi) / 2
    xy_proj = xy / (z + shift)[:, None]
    focal = float((xy_proj * uv).sum() / np.square(xy_proj).sum())
    return shift, focal

for units, label in [(ct.ComputeUnit.CPU_AND_GPU, "gpu"), (ct.ComputeUnit.ALL, "all")]:
    t0 = time.time(); ml = ct.models.MLModel(MODEL, compute_units=units); load = time.time() - t0
    for name in PHOTOS:
        img = Image.open(name).convert("RGB").resize((W, H), Image.BICUBIC)
        x = (np.asarray(img).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # 1,3,H,W in 0..1 (RGB)
        out = ml.predict({"image": x})
        if name == PHOTOS[0]:
            ts = []
            for _ in range(15):
                t = time.perf_counter(); ml.predict({"image": x}); ts.append((time.perf_counter() - t) * 1e3)
            ts = ts[3:]
            print(f"[{label}] load {load:.2f}s | predict median {np.median(ts):.1f} ms  min {min(ts):.1f} ms  -> {1000/np.median(ts):.1f} fps (Python overhead incl.)")
        pts = out["points"].astype(np.float32)[0]; mask = out["mask"].astype(np.float32)[0]; scale = float(out["metric_scale"].ravel()[0])
        normal = out["normal"].astype(np.float32)[0]
        z = pts[..., 2]; mb = mask > 0.5
        if label == "gpu":
            print(f"  {name}: outputs points{pts.shape} normal{normal.shape} mask{mask.shape} metric_scale={scale:.4f}  "
                  f"mask>0.5 coverage={mb.mean()*100:.1f}%  raw z range [{z[mb].min():.3f}, {z[mb].max():.3f}] median {np.median(z[mb]):.3f}")
            # recover focal/shift on a 64x64 nearest subsample of masked points (as MoGe does)
            ys = np.linspace(0, H - 1, 64).astype(int); xs = np.linspace(0, W - 1, 64).astype(int)
            uv = uv_grid(W, H)[ys][:, xs]; p_lr = pts[ys][:, xs]; m_lr = mb[ys][:, xs]
            shift, focal = solve_focal_shift(uv[m_lr], p_lr[m_lr])
            fov_x = 2 * np.degrees(np.arctan((W / H) / (1 + (W / H) ** 2) ** 0.5 / focal))
            d_full = (z + shift) * scale; d_noshift = z * scale
            print(f"    recovered shift={shift:+.4f} focal={focal:.4f} (fov_x≈{fov_x:.1f}°)  ->  metric depth (z+shift)*scale: "
                  f"masked range [{d_full[mb].min():.2f}, {d_full[mb].max():.2f}] m median {np.median(d_full[mb]):.2f} m")
            print(f"    WITHOUT shift (z*scale): range [{d_noshift[mb].min():.2f}, {d_noshift[mb].max():.2f}] m median {np.median(d_noshift[mb]):.2f} m  "
                  f"-> abs error of dropping shift = {abs(shift)*scale:.2f} m ({abs(shift)*scale/np.median(d_full[mb])*100:.0f}% of median)")
            np.save(f"moge_{os.path.splitext(os.path.basename(name))[0]}_depth_m.npy", d_full)
            def probe(a, pts_):
                return "  ".join(f"{l}={np.median(a[max(0,int(yf*(H-1))-2):int(yf*(H-1))+3, max(0,int(xf*(W-1))-2):int(xf*(W-1))+3]):.2f}" for l,xf,yf in pts_)
            if "street" in name or "photo" in name:
                print("    probe:", probe(d_full, [("woman_torso", 570/1280, 330/720), ("woman_feet", 565/1280, 600/720), ("road_near", 300/1280, 690/720), ("crowd_mid", 400/1280, 370/720), ("far_end_street", 330/1280, 290/720), ("left_bldg_top", 80/1280, 100/720)]))
            else:
                print("    probe:", probe(d_full, [("cup_left", 650/1280, 440/720), ("table_edge", 750/1280, 600/720), ("sofa_back", 500/1280, 250/720), ("wall", 300/1280, 80/720), ("curtain", 1150/1280, 150/720)]))
