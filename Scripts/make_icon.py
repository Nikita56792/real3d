# Generates Resources/AppIcon.icns: dark rounded tile with red/cyan offset "3D" frames.
import os, subprocess, sys
from PIL import Image, ImageDraw
root = os.path.join(os.path.dirname(__file__), "..")
iconset = os.path.join(root, "build", "AppIcon.iconset")
os.makedirs(iconset, exist_ok=True)
def render(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m = size * 0.08
    d.rounded_rectangle([m, m, size - m, size - m], radius=size * 0.2, fill=(24, 26, 32, 255))
    # two 16:9 frames offset horizontally: red (left eye) and cyan (right eye)
    fw, fh = size * 0.52, size * 0.52 * 9 / 16
    cx, cy = size / 2, size / 2
    off = size * 0.06
    lw = max(2, int(size * 0.035))
    d.rounded_rectangle([cx - fw/2 - off, cy - fh/2, cx + fw/2 - off, cy + fh/2], radius=size*0.03, outline=(255, 70, 70, 230), width=lw)
    d.rounded_rectangle([cx - fw/2 + off, cy - fh/2, cx + fw/2 + off, cy + fh/2], radius=size*0.03, outline=(70, 220, 255, 230), width=lw)
    d.rounded_rectangle([cx - fw/2 + off, cy - fh/2, cx + fw/2 - off, cy + fh/2], radius=size*0.03, fill=(255, 255, 255, 40))
    return img
for s in (16, 32, 128, 256, 512):
    render(s).save(os.path.join(iconset, f"icon_{s}x{s}.png"))
    render(s * 2).save(os.path.join(iconset, f"icon_{s}x{s}@2x.png"))
out = os.path.join(root, "Resources", "AppIcon.icns")
os.makedirs(os.path.dirname(out), exist_ok=True)
subprocess.check_call(["iconutil", "-c", "icns", iconset, "-o", out])
print("wrote", out)
