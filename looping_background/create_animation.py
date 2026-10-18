"""
Seamless looping background: the reference stays exactly as drawn, only the two
bubble / cell clusters (upper-right and lower-left) move.

Run (from any directory):
    blender -b -P create_animation.py -- [options]

Options (all optional):
    --ref PATH            reference image (default: ./source/reference_16x9.webp next to this script)
    --out DIR             output directory (default: directory of this script)
    --seconds S           loop duration (default 16)
    --fps F               frame rate (default 30)
    --res WxH             render resolution (default 3840x2160)
    --samples N           Cycles anti-aliasing samples (default 16)
    --render              render frames after building the scene
    --frames A-B          frame range to render (inclusive, default 0..N-1)
    --frames-dir DIR      where rendered PNG frames go (default OUT/frames/<WxH>)
    --identity            zero every motion amplitude (fidelity test against the reference)
    --no-save             do not write looping_background.blend

Loop design: every animated quantity is a function of the loop phase p = frame / N
(N = fps * seconds unique frames, rendered as frames 0..N-1) through sin/cos(2 pi k p)
with integer k only. Frame N is therefore mathematically identical to frame 0,
including velocity and acceleration.
"""

import bpy
import math
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    cfg = {
        "ref": os.path.join(SCRIPT_DIR, "source", "reference_16x9.webp"),
        "out": SCRIPT_DIR,
        "seconds": 16.0,
        "fps": 30,
        "res": (3840, 2160),
        "samples": 16,
        "render": False,
        "frames": None,
        "frames_dir": None,
        "identity": False,
        "save": True,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--ref":
            cfg["ref"] = os.path.abspath(argv[i + 1]); i += 1
        elif a == "--out":
            cfg["out"] = os.path.abspath(argv[i + 1]); i += 1
        elif a == "--seconds":
            cfg["seconds"] = float(argv[i + 1]); i += 1
        elif a == "--fps":
            cfg["fps"] = int(argv[i + 1]); i += 1
        elif a == "--res":
            w, h = argv[i + 1].lower().split("x"); cfg["res"] = (int(w), int(h)); i += 1
        elif a == "--samples":
            cfg["samples"] = int(argv[i + 1]); i += 1
        elif a == "--render":
            cfg["render"] = True
        elif a == "--frames":
            s, e = argv[i + 1].split("-"); cfg["frames"] = (int(s), int(e)); i += 1
        elif a == "--frames-dir":
            cfg["frames_dir"] = os.path.abspath(argv[i + 1]); i += 1
        elif a == "--identity":
            cfg["identity"] = True
        elif a == "--no-save":
            cfg["save"] = False
        else:
            raise SystemExit(f"unknown option {a}")
        i += 1
    n = cfg["fps"] * cfg["seconds"]
    if abs(n - round(n)) > 1e-9:
        raise SystemExit("fps * seconds must be an integer number of frames")
    cfg["n_frames"] = int(round(n))
    return cfg


CFG = parse_args()
N = CFG["n_frames"]
AMP = 0.0 if CFG["identity"] else 1.0

# Loop frequencies (whole cycles per loop; must be integers for loop closure)
K_SWELL = 4             # each cell swells and relaxes 4x per loop (every 4 s at 16 s)
K_DRIFT = 3             # each cell drifts on a small orbit 3x per loop (every 5.3 s)
K_RIPPLE = 1            # membrane ripple evolves once around per loop

# Scene geometry: the 16:9 reference fills the frame at z=0 (frame height = 2 BU).
CAM_DIST = 5.0          # camera distance from the image plane (BU)
SENSOR_H = 24.0         # mm, vertical sensor fit
FOCAL = SENSOR_H * CAM_DIST / 2.0   # frame half-height at z=0 is exactly 1 BU
IMG_SIZE = 2.0          # image plane height (BU); width = IMG_SIZE * ASPECT
OVERSCAN = 1.04         # bubble planes slightly larger than the frame so their float never shows an edge

# Centre of the background iris behind the BETA wordmark (fraction of the 1672x940
# reference, x right, y down). Used to seed the central body and to keep the
# lines around the wordmark out of the cell detection.
IRIS_REL = (944.0 / 1672, 490.1 / 940)

ASPECT = 16 / 9         # replaced by the real image aspect in build_layers()
TEX_DIR = os.path.join(CFG["out"], "textures")
CELL_STATS = {}
CELL_FIELD_RANGE = None  # stored displacement range (image-height units); set from the data


# ----------------------------------------------------------------------------
# 1. Layer separation and cell analysis (numpy only; Blender's bundled Python)
# ----------------------------------------------------------------------------

def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _dilate(b, r):
    p = np.pad(b, r)
    out = np.zeros_like(b)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                out |= p[r + dy:r + dy + b.shape[0], r + dx:r + dx + b.shape[1]]
    return out


def _reconstruct(seed, mask):
    """Geodesic reconstruction: the connected part of `mask` touching `seed`."""
    cur = seed & mask
    while True:
        nxt = _dilate(cur, 1) & mask
        if np.array_equal(nxt, cur):
            return cur
        cur = nxt


def _blur(a, sigma):
    h, w = a.shape
    pad = int(3 * sigma)
    a = np.pad(a, pad)
    fy = np.fft.fftfreq(a.shape[0])
    fx = np.fft.fftfreq(a.shape[1])
    g = np.exp(-2 * (np.pi * sigma) ** 2 * (fy[:, None] ** 2 + fx[None, :] ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(a) * g))[pad:pad + h, pad:pad + w]


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _save_png16(name, rgb):
    """rgb: (H, W, 3) values in [0, 1], top row first. Saved as raw 16-bit PNG data
    (Non-Color: no transform on write or read; round-trip error < 1e-5). Alpha is
    always 1: Blender un-premultiplies RGB by alpha when writing PNG, so data must
    never live in the alpha channel."""
    h, w = rgb.shape[:2]
    rgba = np.concatenate([rgb, np.ones((h, w, 1), np.float32)], 2)
    path = os.path.join(TEX_DIR, name + ".png")
    img = bpy.data.images.new(name + "_tmp", w, h, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(np.ascontiguousarray(np.clip(rgba[::-1], 0, 1)).astype(np.float32).ravel())
    scn = bpy.context.scene
    st = scn.render.image_settings
    st.file_format, st.color_mode, st.color_depth, st.compression = "PNG", "RGBA", "16", 90
    img.save_render(path, scene=scn)
    bpy.data.images.remove(img)
    return path


def _lanczos_matrix(n_in, n_out, a=3):
    """Dense 1-D Lanczos-a resampling matrix (n_out x n_in), normalised rows."""
    scale = n_out / n_in
    widen = max(1.0, 1.0 / scale)                      # pre-filter when downsampling
    x = (np.arange(n_out) + 0.5) / scale - 0.5
    dist = (x[:, None] - np.arange(n_in)[None, :]) / widen
    k = np.sinc(dist) * np.sinc(dist / a) * (np.abs(dist) < a)
    return (k / k.sum(1, keepdims=True)).astype(np.float32)


def _resample(img, out_w, out_h):
    h, w = img.shape[:2]
    wy, wx = _lanczos_matrix(h, out_h), _lanczos_matrix(w, out_w)
    return np.stack([wy @ img[..., c] @ wx.T for c in range(img.shape[2])], 2)


def _maxpool(a, f):
    h, w = a.shape
    return a[:h // f * f, :w // f * f].reshape(h // f, f, w // f, f).max((1, 3))


def _wall_distance(wall, max_iter):
    """Octagonal distance from every pixel to the nearest wall pixel (inf beyond max_iter)."""
    d = np.where(wall, 0.0, np.inf)
    cur = wall.copy()
    for i in range(1, max_iter + 1):
        p = np.pad(cur, 1)
        n = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        if i % 2 == 0:
            n |= p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
        d[n & ~cur] = i
        cur |= n
    return d


def _is_local_max(a, r):
    p = np.pad(a, r, constant_values=-np.inf)
    m = np.full_like(a, -np.inf)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            m = np.maximum(m, p[r + dy:r + dy + a.shape[0], r + dx:r + dx + a.shape[1]])
    return a >= m


def _find_cells(lum, foam_w, f, wall_thr, r_min, r_max, min_closed, n_dir, ray_max, min_foam):
    """Cell candidates at 1/f resolution: maxima of the distance to the membranes.

    A candidate counts as a cell when rays cast in `n_dir` directions meet a membrane
    (or the frame edge) within `ray_max` radii in at least `min_closed` directions,
    so drawn-open cells with gaps or one missing side are accepted, and when the
    membranes around it belong to the bubble layers (not the lines of the wordmark).
    Returns (cx, cy, r) in reference pixels.
    """
    wall = _maxpool(lum > wall_thr, f)
    d = _wall_distance(wall, int(r_max / f) + 4)
    peaks = _is_local_max(d, 2) & (d >= r_min / f) & (d <= r_max / f) & np.isfinite(d)
    py, px = np.nonzero(peaks)
    pr = d[py, px]
    hh, ww = wall.shape
    closed = np.zeros(len(pr), int)
    foam_ring = np.zeros(len(pr))
    for ang in np.linspace(0, 2 * np.pi, n_dir, endpoint=False):
        hit = np.zeros(len(pr), bool)
        for s in np.linspace(0.9, ray_max, 8):
            yy = (py + np.sin(ang) * pr * s).round().astype(int)
            xx = (px + np.cos(ang) * pr * s).round().astype(int)
            outside = (yy < 0) | (yy >= hh) | (xx < 0) | (xx >= ww)   # cells cut by the frame edge
            hit |= outside | wall[np.clip(yy, 0, hh - 1), np.clip(xx, 0, ww - 1)]
        closed += hit
        yy = np.clip(((py + np.sin(ang) * (pr + 1.0)) * f).round().astype(int), 0, lum.shape[0] - 1)
        xx = np.clip(((px + np.cos(ang) * (pr + 1.0)) * f).round().astype(int), 0, lum.shape[1] - 1)
        foam_ring += foam_w[yy, xx] / n_dir
    keep = (closed >= min_closed) & (foam_ring > min_foam)
    return (px[keep] + 0.5) * f, (py[keep] + 0.5) * f, pr[keep] * f


def detect_cells(lum, foam_w):
    """All bubbles of the two clusters: small sealed ones and big drawn-open ones."""
    small = _find_cells(lum, foam_w, f=2, wall_thr=0.10, r_min=4, r_max=90,
                        min_closed=7, n_dir=8, ray_max=1.9, min_foam=0.5)
    big = _find_cells(lum, foam_w, f=4, wall_thr=0.05, r_min=40, r_max=280,
                      min_closed=11, n_dir=16, ray_max=2.4, min_foam=0.55)
    cx = np.concatenate([big[0], small[0]])
    cy = np.concatenate([big[1], small[1]])
    cr = np.concatenate([big[2], small[2]])
    order = np.argsort(-cr, kind="stable")
    accepted = []
    for i in order:                                     # non-maximum suppression, largest first
        if all(math.hypot(cy[i] - cy[j], cx[i] - cx[j]) >= 0.9 * max(cr[i], cr[j]) + 2 for j in accepted):
            accepted.append(i)
    acc = np.array(accepted, int)
    return cx[acc], cy[acc], cr[acc]


def build_cell_fields(cells, w, h, seed=7):
    """Per-bubble motion baked into four periodic basis vector fields (image-height units, y up).

    Bubble i swells/relaxes by a_i sin(th1 + phi_i) and drifts on a small orbit
    b_i R_i (cos(th2 + psi_i), sin(th2 + psi_i)). Each term expands as
        A sin(th + phi) = sin(th) * A cos(phi) + cos(th) * A sin(phi),
    so hundreds of independently phased bubbles collapse into 'sin' and 'cos' basis
    fields per loop frequency; the shader recombines them with sin/cos of th_k = 2 pi k p.
    """
    rng = np.random.default_rng(seed)
    fw, fh = w // 2, h // 2
    sx = ASPECT / fw
    X = (np.arange(fw) + 0.5) * sx                       # height units
    Y = 1.0 - (np.arange(fh) + 0.5) / fh
    F = {k: np.zeros((fh, fw, 2), np.float32) for k in ("s1", "c1", "s2", "c2")}
    cx, cy, cr = cells
    ccx, ccy, crr = cx / h, 1.0 - cy / h, cr / h          # to height units
    # swell phases travel across each cluster as a slow wave, with per-bubble jitter
    proj = ccx * 0.86 + ccy * 0.51
    phi = -2 * np.pi * proj / 0.45 + rng.uniform(-0.8, 0.8, len(ccx))
    psi = rng.uniform(0, 2 * np.pi, len(ccx))
    a = rng.uniform(0.08, 0.14, len(ccx))                 # swell, fraction of the radius
    b = rng.uniform(0.06, 0.12, len(ccx))                 # orbit, fraction of the radius
    for i in range(len(ccx)):
        R = crr[i]
        ext = 2.1 * R
        c0 = max(int((ccx[i] - ext) / sx), 0); c1 = min(int((ccx[i] + ext) / sx) + 2, fw)
        r0 = max(int((1 - ccy[i] - ext) * fh), 0); r1 = min(int((1 - ccy[i] + ext) * fh) + 2, fh)
        if c0 >= c1 or r0 >= r1:
            continue
        dx = X[c0:c1][None, :] - ccx[i]
        dy = Y[r0:r1][:, None] - ccy[i]
        g = _smoothstep(2.1, 1.0, np.sqrt(dx * dx + dy * dy) / R)   # 1 inside, 0 beyond ~2 radii
        radial = np.stack([dx * g, dy * g], -1)
        sl = (slice(r0, r1), slice(c0, c1))
        F["s1"][sl] += a[i] * math.cos(phi[i]) * radial
        F["c1"][sl] += a[i] * math.sin(phi[i]) * radial
        cp, sp = math.cos(psi[i]), math.sin(psi[i])
        orb = b[i] * R * g
        F["s2"][sl] += np.stack([-sp * orb, cp * orb], -1)
        F["c2"][sl] += np.stack([cp * orb, sp * orb], -1)

    def min_jacobian(scale):
        """Smallest det(I - grad d) over a grid of phases: must stay > 0 (no fold-over)."""
        worst = np.inf
        for t1 in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            for t2 in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                d = scale * (math.sin(t1) * F["s1"] + math.cos(t1) * F["c1"]
                             + math.sin(t2) * F["s2"] + math.cos(t2) * F["c2"])
                dxx = np.gradient(d[..., 0], sx, axis=1); dxy = -np.gradient(d[..., 0], 1 / fh, axis=0)
                dyx = np.gradient(d[..., 1], sx, axis=1); dyy = -np.gradient(d[..., 1], 1 / fh, axis=0)
                worst = min(worst, float(((1 - dxx) * (1 - dyy) - dxy * dyx).min()))
        return worst

    scale = 1.0
    jac = min_jacobian(scale)
    while jac < 0.35 and scale > 0.3:
        scale *= 0.9
        jac = min_jacobian(scale)
    for k in F:
        F[k] *= scale
    peak = max(float(np.abs(F[k]).max()) for k in F)
    global CELL_FIELD_RANGE
    CELL_FIELD_RANGE = max(2.05 * peak, 1e-3)
    CELL_STATS.update({"cells": int(len(ccx)), "amplitude_scale": round(scale, 3),
                       "min_jacobian": round(jac, 3), "max_basis_displacement_px": round(peak * h, 1)})
    enc = lambda v: v / CELL_FIELD_RANGE + 0.5
    zero = np.full((fh, fw, 1), 0.5, np.float32)
    return {   # 8 scalar fields packed into three RGB textures
        "cells_a": _save_png16("cells_a", enc(np.concatenate([F["s1"], F["c1"][..., :1]], -1))),
        "cells_b": _save_png16("cells_b", enc(np.concatenate([F["c1"][..., 1:], F["s2"]], -1))),
        "cells_c": _save_png16("cells_c", np.concatenate([enc(F["c2"]), zero], -1)),
    }


def build_layers(ref_path):
    """Split the reference into additive layers on its black background.

    Because the artwork sits on (near) pure black, it is an additive sum of its
    elements. A soft partition of unity (weights summing to exactly 1) splits it
    into the static central body and the two bubble clusters, whose emission sums
    back to the original in linear light.
    """
    global ASPECT
    os.makedirs(TEX_DIR, exist_ok=True)
    src = bpy.data.images.load(ref_path, check_existing=False)
    w, h = src.size
    px = np.empty(w * h * 4, np.float32)
    src.pixels.foreach_get(px)
    bpy.data.images.remove(src)
    ASPECT = w / h
    srgb = px.reshape(h, w, 4)[::-1, :, :3]          # top row first, sRGB-encoded
    lin = srgb_to_linear(srgb).astype(np.float32)
    lum = srgb.mean(2)

    f = 4                                              # analysis at 1/4 resolution
    hh, ww = h // f, w // f
    small = lum[:hh * f, :ww * f].reshape(hh, f, ww, f).max((1, 3))

    # Central body: wordmark + iris lines + wire sphere form one connected bright body.
    occ = _dilate(small > 0.18, 2)
    seed = np.zeros_like(occ)
    sy, sx = int(IRIS_REL[1] * hh), int(IRIS_REL[0] * ww)
    seed[sy - 10:sy + 10, sx - 10:sx + 10] = True
    centre = _reconstruct(seed, occ)
    # Everything else that is visible: the bubble clusters framing the subject,
    # split along the diagonal into an upper-right and a lower-left group.
    other = _dilate(small > 0.05, 2) & ~_dilate(centre, 6)
    yy, xx = np.mgrid[0:hh, 0:ww]
    tr = other & (yy * ww < xx * hh)
    bl = other & ~(yy * ww < xx * hh)

    def up(b):
        big = np.kron(b.astype(np.float32), np.ones((f, f), np.float32))
        out = np.zeros((h, w), np.float32)
        out[:big.shape[0], :big.shape[1]] = big
        return out

    sig = 10 * h / 1600
    c_s = _blur(up(_dilate(centre, 4)), sig)
    tr_s = _blur(up(_dilate(tr, 4)), sig)
    bl_s = _blur(up(_dilate(bl, 4)), sig)
    tot = c_s + tr_s + bl_s + 1e-3
    w_tr = np.clip(tr_s / tot, 0, 1)
    w_bl = np.clip(bl_s / tot, 0, 1)
    w_c = 1.0 - w_tr - w_bl                            # exact partition of unity

    # Colour layers are Lanczos-resampled to the render resolution here (Cycles' own
    # "Cubic" is a smoothing B-spline), then sampled ~1:1 with linear filtering.
    out_w = CFG["res"][0]
    out_h = int(round(out_w / ASPECT))
    paths = {}
    for name, wt in (("layer_centre", w_c), ("layer_foam_tr", w_tr), ("layer_foam_bl", w_bl)):
        layer = _resample(lin * wt[..., None], out_w, out_h)
        paths[name] = _save_png16(name, np.clip(layer, 0, 1))

    cells = detect_cells(lum, w_tr + w_bl)
    np.savez(os.path.join(TEX_DIR, "cells.npz"), cx=cells[0], cy=cells[1], r=cells[2])
    paths.update(build_cell_fields(cells, w, h))
    return paths


# ----------------------------------------------------------------------------
# 2. Node helpers
# ----------------------------------------------------------------------------

class NodeBuilder:
    def __init__(self, tree):
        self.t = tree
        self.n = tree.nodes
        self.l = tree.links

    def node(self, kind, **props):
        nd = self.n.new(kind)
        for k, v in props.items():
            setattr(nd, k, v)
        return nd

    def link(self, src, dst):
        self.l.new(src, dst)

    def _set(self, sock, val):
        if isinstance(val, bpy.types.NodeSocket):
            self.link(val, sock)
        else:
            sock.default_value = val

    def math(self, op, a, b=0.0, c=None):
        nd = self.node("ShaderNodeMath", operation=op)
        self._set(nd.inputs[0], a)
        self._set(nd.inputs[1], b)
        if c is not None:
            self._set(nd.inputs[2], c)
        return nd.outputs[0]

    def add(self, a, b): return self.math("ADD", a, b)
    def mul(self, a, b): return self.math("MULTIPLY", a, b)
    def sub(self, a, b): return self.math("SUBTRACT", a, b)

    def combine(self, x, y, z=0.0):
        nd = self.node("ShaderNodeCombineXYZ")
        self._set(nd.inputs[0], x)
        self._set(nd.inputs[1], y)
        self._set(nd.inputs[2], z)
        return nd.outputs[0]

    def separate(self, v):
        nd = self.node("ShaderNodeSeparateXYZ")
        self._set(nd.inputs[0], v)
        return nd.outputs

    def angle(self, phase, k, offset=0.0):
        """2*pi*k*phase + offset  (k must be an integer for loop closure)."""
        assert float(k).is_integer()
        return self.math("MULTIPLY_ADD", phase, 2 * math.pi * k, offset)


DRIVERS = []   # (fcurve, expression) for validation


def add_driver(owner, prop, expr, index=-1):
    fc = owner.driver_add(prop, index) if index >= 0 else owner.driver_add(prop)
    fc.driver.type = "SCRIPTED"
    fc.driver.expression = expr
    for m in list(fc.modifiers):
        fc.modifiers.remove(m)
    DRIVERS.append((fc, expr))
    return fc


def loop_clock_group():
    """Node group exposing the loop phase p = frame / N (driven, no Python needed)."""
    g = bpy.data.node_groups.new("LoopClock", "ShaderNodeTree")
    g.interface.new_socket("Phase", in_out="OUTPUT", socket_type="NodeSocketFloat")
    out = g.nodes.new("NodeGroupOutput")
    val = g.nodes.new("ShaderNodeValue")
    val.name = "Phase"
    g.links.new(val.outputs[0], out.inputs[0])
    add_driver(val.outputs[0], "default_value", f"frame / {N}")
    return g


def load_data_image(path):
    img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = "Non-Color"   # values are scene-linear colour or raw data
    return img


def image_node(nb, img, uv, extension="MIRROR"):
    tex = nb.node("ShaderNodeTexImage", interpolation="Linear", extension=extension)
    tex.image = img
    nb.link(uv, tex.inputs["Vector"])
    return tex


def additive_output(nb, color):
    """Pure additive emission: Emission + Transparent (no occlusion of layers behind)."""
    em = nb.node("ShaderNodeEmission")
    nb._set(em.inputs["Color"], color)
    em.inputs["Strength"].default_value = 1.0
    tr = nb.node("ShaderNodeBsdfTransparent")
    add = nb.node("ShaderNodeAddShader")
    nb.link(em.outputs[0], add.inputs[0])
    nb.link(tr.outputs[0], add.inputs[1])
    out = nb.node("ShaderNodeOutputMaterial", target="CYCLES")
    nb.link(add.outputs[0], out.inputs["Surface"])


# ----------------------------------------------------------------------------
# 3. Materials
# ----------------------------------------------------------------------------

def centre_material(layer_img):
    """The wordmark and its lines, exactly as drawn (static)."""
    mat = bpy.data.materials.new("M_Centre")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    nb = NodeBuilder(mat.node_tree)
    uv = nb.node("ShaderNodeTexCoord").outputs["UV"]
    additive_output(nb, image_node(nb, layer_img, uv).outputs["Color"])
    return mat


def bubble_material(name, clock, layer_img, field_imgs, ripple_offset):
    """Living bubbles: the cluster layer is resampled through a periodic per-bubble warp

        uv' = uv - d(uv, t),
        d = sin(th1) S1 + cos(th1) C1 + sin(th2) S2 + cos(th2) C2 + ripple,
        th_k = 2 pi K p.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    nb = NodeBuilder(mat.node_tree)
    gn = nb.node("ShaderNodeGroup")
    gn.node_tree = clock
    p = gn.outputs[0]
    uv = nb.node("ShaderNodeTexCoord").outputs["UV"]
    s = nb.separate(uv)
    th1, th2 = nb.angle(p, K_SWELL), nb.angle(p, K_DRIFT)
    s1, c1 = nb.math("SINE", th1), nb.math("COSINE", th1)
    s2, c2 = nb.math("SINE", th2), nb.math("COSINE", th2)

    def fetch(img):
        t = image_node(nb, img, uv, extension="EXTEND")
        sc = nb.node("ShaderNodeSeparateColor")
        nb.link(t.outputs["Color"], sc.inputs[0])
        return [nb.mul(nb.sub(sc.outputs[i], 0.5), CELL_FIELD_RANGE * AMP) for i in range(3)]

    fa, fb, fc = fetch(field_imgs["cells_a"]), fetch(field_imgs["cells_b"]), fetch(field_imgs["cells_c"])
    S1, C1, S2, C2 = (fa[0], fa[1]), (fa[2], fb[0]), (fb[1], fb[2]), (fc[0], fc[1])
    dx = nb.add(nb.add(nb.mul(s1, S1[0]), nb.mul(c1, C1[0])), nb.add(nb.mul(s2, S2[0]), nb.mul(c2, C2[0])))
    dy = nb.add(nb.add(nb.mul(s1, S1[1]), nb.mul(c1, C1[1])), nb.add(nb.mul(s2, S2[1]), nb.mul(c2, C2[1])))

    # Fine membrane ripple: 4-D noise sampled on a circle (R cos th, R sin th) in its
    # last two dimensions, which is exactly periodic in the loop phase.
    ang = nb.angle(p, K_RIPPLE, ripple_offset)
    wn = nb.node("ShaderNodeTexNoise", noise_dimensions="4D")
    nb.link(nb.combine(nb.mul(s[0], ASPECT * 14.0), nb.mul(s[1], 14.0),
                       nb.mul(nb.math("COSINE", ang), 0.9)), wn.inputs["Vector"])
    nb.link(nb.mul(nb.math("SINE", ang), 0.9), wn.inputs["W"])
    wn.inputs["Detail"].default_value = 1.0
    wc = nb.node("ShaderNodeSeparateColor")
    nb.link(wn.outputs["Color"], wc.inputs[0])
    dx = nb.add(dx, nb.mul(nb.sub(wc.outputs[0], 0.5), 0.0045 * AMP))
    dy = nb.add(dy, nb.mul(nb.sub(wc.outputs[1], 0.5), 0.0045 * AMP))

    uvw = nb.combine(nb.sub(s[0], nb.mul(dx, 1.0 / ASPECT)), nb.sub(s[1], dy))
    additive_output(nb, image_node(nb, layer_img, uvw).outputs["Color"])
    return mat


# ----------------------------------------------------------------------------
# 4. Scene assembly
# ----------------------------------------------------------------------------

def image_plane(name, mat, z, overscan=1.0):
    """Plane at depth z, scaled so the reference projects onto exactly the frame.

    With overscan > 1 the plane is enlarged and its UVs scaled about the centre to
    match, so the image keeps its size while a mirrored margin covers the frame edge.
    """
    s = (CAM_DIST - z) / CAM_DIST * overscan
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, z))
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = (IMG_SIZE * ASPECT * s, IMG_SIZE * s, 1.0)
    ob.data.materials.append(mat)
    for loop_uv in ob.data.uv_layers.active.data:
        loop_uv.uv = ((loop_uv.uv[0] - 0.5) * overscan + 0.5, (loop_uv.uv[1] - 0.5) * overscan + 0.5)
    return ob


def periodic_expr(amp, k, phase0):
    """amp * sin(2 pi k frame / N + phase0) -- integer k keeps it N-periodic."""
    assert float(k).is_integer()
    return f"{amp * AMP:.6f} * sin({2 * math.pi * k:.10f} * frame / {N} + {phase0:.6f})"


def build_scene(paths):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    DRIVERS.clear()
    scn = bpy.context.scene
    scn.name = "LoopingBackground"
    clock = loop_clock_group()

    world = bpy.data.worlds.new("Black")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    scn.world = world

    imgs = {k: load_data_image(v) for k, v in paths.items()}
    for im in imgs.values():
        im.pack()
    fields = {k: imgs[k] for k in ("cells_a", "cells_b", "cells_c")}

    # The wordmark and its lines: static, exactly the reference.
    image_plane("Subject", centre_material(imgs["layer_centre"]), 0.0)

    # The two bubble clusters: every bubble moves (shader warp) and each cluster
    # also floats very slowly as a whole around a pivot inside it.
    for name, layer, z, pivot_uv, ph in (("BubblesTR", "layer_foam_tr", 0.02, (0.86, 0.82), 0.0),
                                         ("BubblesBL", "layer_foam_bl", 0.01, (0.10, 0.22), 2.1)):
        s = (CAM_DIST - z) / CAM_DIST
        piv = bpy.data.objects.new(name + "_Pivot", None)
        scn.collection.objects.link(piv)
        px_ = (pivot_uv[0] - 0.5) * IMG_SIZE * ASPECT * s
        py_ = (pivot_uv[1] - 0.5) * IMG_SIZE * s
        mat = bubble_material("M_" + name, clock, imgs[layer], fields, 3.7 + ph)
        pl = image_plane(name, mat, z, OVERSCAN)
        pl.parent = piv
        pl.location = (-px_, -py_, 0.0)
        piv.location = (px_, py_, z)
        add_driver(piv, "location", f"{px_:.6f} + " + periodic_expr(0.010, 1, ph), 0)
        add_driver(piv, "location", f"{py_:.6f} + " + periodic_expr(0.008, 1, ph + 1.3), 1)
        add_driver(piv, "rotation_euler", periodic_expr(math.radians(0.45), 1, ph + 0.5), 2)
        for ax in range(2):
            add_driver(piv, "scale", "1.0 + " + periodic_expr(0.008, 1, ph + 2.4), ax)

    # Locked-off camera.
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_height = SENSOR_H
    cam_data.lens = FOCAL
    cam_data.clip_start = 0.1
    cam_data.clip_end = 100
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = (0, 0, CAM_DIST)
    scn.collection.objects.link(cam)
    scn.camera = cam

    scn.frame_start = 0
    scn.frame_end = N - 1
    scn.render.fps = CFG["fps"]
    scn.render.resolution_x, scn.render.resolution_y = CFG["res"]
    scn.render.resolution_percentage = 100
    scn.render.engine = "CYCLES"
    cy = scn.cycles
    cy.device = "CPU"
    cy.samples = CFG["samples"]
    cy.use_adaptive_sampling = False
    cy.use_denoising = False            # emission-only scene is noise free; denoisers smear line art
    cy.seed = 0
    cy.use_animated_seed = False        # identical sampling pattern on every frame -> no shimmer
    cy.max_bounces = 0
    cy.transparent_max_bounces = 16
    cy.pixel_filter_type = "BLACKMAN_HARRIS"
    cy.filter_width = 1.0
    scn.render.use_motion_blur = False
    scn.render.dither_intensity = 1.0
    scn.render.film_transparent = False
    scn.render.use_compositing = False
    scn.view_settings.view_transform = "Standard"   # reproduces the reference's own grade exactly
    scn.view_settings.look = "None"
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.render.image_settings.color_depth = "16"
    scn.render.image_settings.compression = 15
    scn["loop_frames"] = N
    scn["loop_fps"] = CFG["fps"]
    return scn


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------

def main():
    if not os.path.isfile(CFG["ref"]):
        raise SystemExit(f"reference image not found: {CFG['ref']}")
    print(f"[loop] reference={CFG['ref']} N={N} fps={CFG['fps']} res={CFG['res']} identity={CFG['identity']}")
    paths = build_layers(CFG["ref"])
    print(f"[loop] cells: {CELL_STATS}  field range={CELL_FIELD_RANGE:.5f}")
    scn = build_scene(paths)
    for fc, expr in DRIVERS:
        if not fc.driver.is_simple_expression:
            raise SystemExit(f"driver is not a simple expression (would need auto-exec): {expr}")
    if CFG["save"]:
        path = os.path.join(CFG["out"], "looping_background.blend")
        bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
        print(f"[loop] saved {path}")
    if CFG["render"]:
        w, h = CFG["res"]
        fdir = CFG["frames_dir"] or os.path.join(CFG["out"], "frames", f"{w}x{h}")
        os.makedirs(fdir, exist_ok=True)
        a, b = CFG["frames"] or (0, N - 1)
        for fr in range(a, b + 1):
            target = os.path.join(fdir, f"{fr:05d}.png")
            if os.path.exists(target):
                continue
            scn.frame_set(fr)
            scn.render.filepath = target
            bpy.ops.render.render(write_still=True)
            print(f"[loop] wrote {target}", flush=True)


main()
