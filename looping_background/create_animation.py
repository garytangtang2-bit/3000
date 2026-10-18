"""
Seamless looping cinematic background from a single reference image.

Run (from any directory):
    blender -b -P create_animation.py -- [options]

Options (all optional):
    --ref PATH            reference image (default: ./source/reference_16x9.webp next to this script)
    --out DIR             output directory (default: directory of this script)
    --bpm B               tempo (default 120)
    --beats K             beats per loop, multiple of 8 (default 32 = 8 bars = 16 s at 120 BPM)
    --fps F               frame rate (default 30)
    --res WxH             render resolution (default 3840x2160)
    --samples N           Cycles AA samples (default 16)
    --render              render frames after building the scene
    --frames A-B          frame range to render (inclusive, default 0..N-1)
    --frames-dir DIR      where rendered PNG frames go (default OUT/frames/<WxH>)
    --identity            zero every motion amplitude (fidelity test against the reference)
    --no-save             do not write looping_background.blend

Loop design: every animated quantity is a function of the loop phase
    p = frame / N      (N = fps * beats * 60 / bpm unique frames, rendered as frames 0..N-1)
through cos/sin(2*pi*k*p) with integer k, or fract(r + k*p) for travelling
elements that are invisible at the wrap point. Frame N is therefore
mathematically identical to frame 0, including all time derivatives.
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
        "bpm": 120.0,
        "beats": 32,
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
        elif a == "--bpm":
            cfg["bpm"] = float(argv[i + 1]); i += 1
        elif a == "--beats":
            cfg["beats"] = int(argv[i + 1]); i += 1
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
    cfg["seconds"] = cfg["beats"] * 60.0 / cfg["bpm"]
    n = cfg["fps"] * cfg["seconds"]
    if abs(n - round(n)) > 1e-9 or cfg["beats"] % 8:
        raise SystemExit("fps * beats * 60 / bpm must be an integer and beats a multiple of 8")
    cfg["n_frames"] = int(round(n))
    return cfg


CFG = parse_args()
N = CFG["n_frames"]
BEATS = CFG["beats"]            # beats per loop (integer -> every beat effect closes the loop)
BARS = BEATS // 4               # 4/4 bars per loop
AMP = 0.0 if CFG["identity"] else 1.0

# Scene geometry: the 16:9 reference fills the frame at z=0 (frame height = 2 BU).
CAM_DIST = 5.0          # camera distance from the image plane (BU)
SENSOR_H = 24.0         # mm, vertical sensor fit
FOCAL = SENSOR_H * CAM_DIST / 2.0   # frame half-height at z=0 is exactly 1 BU
IMG_SIZE = 2.0          # image plane height (BU); width = IMG_SIZE * ASPECT
OVERSCAN = 1.04         # image slightly larger than the frame so camera drift never shows an edge

# Key positions as fractions of the reference (x right, y down), measured on the
# 1672x940 image by template-matching the logo against the square original.
STAR_REL = (979.1 / 1672, 537.9 / 940)      # four-point star at the crossing of the BETA wordmark
IRIS_REL = (944.0 / 1672, 490.1 / 940)      # centre of the background iris
SPHERE_REL = (599.7 / 1672, 222.4 / 940)    # centre of the wire-frame sphere
R_SCALE = 1.085         # size of the subject relative to the earlier square layout (image-height units)

ASPECT = 16 / 9         # replaced by the real image aspect in build_layers()
STAR_UV = IRIS_UV = SPHERE_UV = None        # (x, y) in image-height units, y up; set in build_layers()

TEX_DIR = os.path.join(CFG["out"], "textures")


# ----------------------------------------------------------------------------
# 1. Layer separation (numpy only; runs inside Blender's bundled Python)
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


def _save_exr(name, rgba):
    """rgba: (H, W, 4) float32, top row first. Saved as linear float EXR."""
    h, w = rgba.shape[:2]
    path = os.path.join(TEX_DIR, name + ".exr")
    img = bpy.data.images.new(name + "_tmp", w, h, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"   # store raw values, no implicit transform
    img.pixels.foreach_set(np.ascontiguousarray(rgba[::-1]).astype(np.float32).ravel())
    img.filepath_raw = path
    img.file_format = "OPEN_EXR"
    img.save()
    bpy.data.images.remove(img)
    return path


def build_layers(ref_path):
    """Split the reference into additive layers on its black background.

    Because the artwork sits on (near) pure black, it is an additive sum of its
    elements. A soft partition of unity (weights summing to exactly 1) splits it
    into layers whose emission sums back to the original in linear light.
    """
    os.makedirs(TEX_DIR, exist_ok=True)
    src = bpy.data.images.load(ref_path, check_existing=False)
    w, h = src.size
    px = np.empty(w * h * 4, np.float32)
    src.pixels.foreach_get(px)
    bpy.data.images.remove(src)
    global ASPECT, STAR_UV, IRIS_UV, SPHERE_UV
    ASPECT = w / h
    to_xy = lambda rel: (rel[0] * ASPECT, 1.0 - rel[1])
    STAR_UV, IRIS_UV, SPHERE_UV = to_xy(STAR_REL), to_xy(IRIS_REL), to_xy(SPHERE_REL)
    srgb = px.reshape(h, w, 4)[::-1, :, :3]          # top row first, sRGB-encoded
    lin = srgb_to_linear(srgb).astype(np.float32)
    lum = srgb.mean(2)

    f = 4                                              # analysis at 1/4 resolution
    hh, ww = h // f, w // f
    small = lum[:hh * f, :ww * f].reshape(hh, f, ww, f).max((1, 3))

    # Central subject: logo + iris + wire sphere form one connected bright body.
    occ = _dilate(small > 0.18, 2)
    seed = np.zeros_like(occ)
    sy, sx = int(IRIS_REL[1] * hh), int(IRIS_REL[0] * ww)
    seed[sy - 10:sy + 10, sx - 10:sx + 10] = True
    centre = _reconstruct(seed, occ)
    # Everything else that is visible: the foam / cell clusters framing the subject,
    # split along the diagonal into an upper-right and a lower-left group.
    other = _dilate(small > 0.05, 2) & ~_dilate(centre, 6)
    yy, xx = np.mgrid[0:hh, 0:ww]
    tr = other & (yy * ww < xx * hh)                   # top-right of the diagonal
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
    eps = 1e-3
    tot = c_s + tr_s + bl_s + eps
    w_tr = np.clip(tr_s / tot, 0, 1)
    w_bl = np.clip(bl_s / tot, 0, 1)
    w_c = 1.0 - w_tr - w_bl                            # exact partition of unity

    ones = np.ones((h, w, 1), np.float32)
    paths = {}
    for name, wt in (("layer_centre", w_c), ("layer_foam_tr", w_tr), ("layer_foam_bl", w_bl)):
        paths[name] = _save_exr(name, np.concatenate([lin * wt[..., None], ones], 2))

    # Control masks for light animation on the central layer.
    v, u = np.mgrid[0:h, 0:w]
    u = (u + 0.5) / h                                  # image-height units (isotropic)
    v = 1.0 - (v + 0.5) / h
    lum_b = _blur(lum, 2)
    R = R_SCALE
    chrome = w_c * _smoothstep(0.30, 0.75, lum_b)      # bright chrome of the logo
    r_iris = np.hypot(u - IRIS_UV[0], v - IRIS_UV[1])
    iris = w_c * _smoothstep(0.36 * R, 0.26 * R, r_iris) * (1 - _smoothstep(0.22, 0.5, lum_b))
    r_sph = np.hypot((u - SPHERE_UV[0]) / 1.0, (v - SPHERE_UV[1]) / 0.95)
    logoish = _smoothstep(0.25, 0.45, _blur(lum, 4))   # wide grey spikes crossing the sphere
    sphere = w_c * _smoothstep(0.27 * R, 0.19 * R, r_sph) * (1 - logoish)
    detail = w_c * (1 - _smoothstep(0.3, 0.6, lum_b))  # everything that is not chrome
    paths["control"] = _save_exr("control", np.stack([chrome, iris, sphere, detail], 2).astype(np.float32))
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
        if hasattr(val, "is_output") or isinstance(val, bpy.types.NodeSocket):
            self.link(val, sock)
        else:
            sock.default_value = val

    def math(self, op, a, b=0.0, c=None, clamp=False):
        nd = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self._set(nd.inputs[0], a)
        self._set(nd.inputs[1], b)
        if c is not None:
            self._set(nd.inputs[2], c)
        return nd.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        nd = self.node("ShaderNodeVectorMath", operation=op)
        self._set(nd.inputs[0], a)
        if b is not None:
            self._set(nd.inputs[1], b)
        if scale is not None:
            self._set(nd.inputs[3], scale)
        out = nd.outputs["Value"] if op in ("LENGTH", "DOT_PRODUCT", "DISTANCE") else nd.outputs["Vector"]
        return out

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

    def mix_rgb(self, op, fac, a, b):
        nd = self.node("ShaderNodeMix", data_type="RGBA", blend_type=op)
        self._set(nd.inputs[0], fac)
        self._set(nd.inputs[6], a)
        self._set(nd.inputs[7], b)
        return nd.outputs[2]

    # ---- composite helpers -------------------------------------------------
    def add(self, a, b): return self.math("ADD", a, b)
    def mul(self, a, b): return self.math("MULTIPLY", a, b)
    def sub(self, a, b): return self.math("SUBTRACT", a, b)

    def smoothstep(self, e0, e1, x):
        nd = self.node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP", clamp=True)
        self._set(nd.inputs["Value"], x)
        nd.inputs["From Min"].default_value = e0
        nd.inputs["From Max"].default_value = e1
        return nd.outputs["Result"]

    def angle(self, phase, k=1, offset=0.0):
        """2*pi*(k*phase) + offset  (k must be an integer for loop closure)."""
        assert float(k).is_integer()
        return self.math("MULTIPLY_ADD", phase, 2 * math.pi * k, offset)

    def loop_noise(self, xy, phase, scale, radius, offset=0.0, detail=3.0, rough=0.55, k=1):
        """4D noise sampled on a circle in (z, w): exactly periodic in phase.

        n(x, y, R cos(2 pi k p + o), R sin(2 pi k p + o)) is a smooth closed
        path through noise space, so position and all derivatives match at p and p+1.
        """
        ang = self.angle(phase, k, offset)
        z = self.mul(self.math("COSINE", ang), radius)
        wv = self.mul(self.math("SINE", ang), radius)
        s = self.separate(xy)
        vec = self.combine(s[0], s[1], z)
        nd = self.node("ShaderNodeTexNoise", noise_dimensions="4D")
        self.link(vec, nd.inputs["Vector"])
        self.link(wv, nd.inputs["W"])
        nd.inputs["Scale"].default_value = scale
        nd.inputs["Detail"].default_value = detail
        nd.inputs["Roughness"].default_value = rough
        return nd.outputs["Fac"]

    def beat_env(self, phase, per_loop, decay, attack, offset=0.0):
        """Percussive envelope repeating `per_loop` (integer) times per loop.

        b = fract(per_loop * p + offset);  e(b) = (1 - exp(-a b)) (exp(-d b) - exp(-d)) / (1 - exp(-d)),
        normalised to peak 1. e(0) = e(1) = 0, so consecutive beats (and the loop
        boundary, which is just another beat boundary) join without a step.
        """
        assert float(per_loop).is_integer()
        bs = np.linspace(0, 1, 20001)
        ev = (1 - np.exp(-attack * bs)) * (np.exp(-decay * bs) - math.exp(-decay)) / (1 - math.exp(-decay))
        peak = float(ev.max())
        b = self.math("FRACT", self.math("MULTIPLY_ADD", phase, float(per_loop), offset))
        rise = self.math("SUBTRACT", 1.0, self.math("EXPONENT", self.mul(b, -attack)))
        fall = self.sub(self.math("EXPONENT", self.mul(b, -decay)), math.exp(-decay))
        return self.mul(self.mul(rise, fall), 1.0 / ((1 - math.exp(-decay)) * peak))


DRIVERS = []   # (id_data, data_path, index, expression) for validation


def add_driver(owner, prop, expr, index=-1):
    fc = owner.driver_add(prop, index) if index >= 0 else owner.driver_add(prop)
    drv = fc.driver
    drv.type = "SCRIPTED"
    drv.expression = expr
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


def phase_socket(nb, clock):
    gn = nb.node("ShaderNodeGroup")
    gn.node_tree = clock
    return gn.outputs[0]


def load_linear(path):
    img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = "Non-Color"   # data is already scene-linear
    img.alpha_mode = "CHANNEL_PACKED"      # alpha carries a mask, never premultiply
    return img


def additive_output(nb, color, strength=1.0):
    """Pure additive emission: Emission + Transparent (no occlusion of layers behind)."""
    em = nb.node("ShaderNodeEmission")
    nb._set(em.inputs["Color"], color)
    nb._set(em.inputs["Strength"], strength)
    tr = nb.node("ShaderNodeBsdfTransparent")
    add = nb.node("ShaderNodeAddShader")
    nb.link(em.outputs[0], add.inputs[0])
    nb.link(tr.outputs[0], add.inputs[1])
    out = nb.node("ShaderNodeOutputMaterial", target="CYCLES")
    nb.link(add.outputs[0], out.inputs["Surface"])


def image_node(nb, img, uv):
    tex = nb.node("ShaderNodeTexImage", interpolation="Cubic", extension="MIRROR")
    tex.image = img
    nb.link(uv, tex.inputs["Vector"])
    return tex


def edge_feather(nb, uv, width=0.035):
    """Soft fade at the square image border so the 16:9 frame has no hard edges."""
    s = nb.separate(uv)
    f = 1.0
    for comp in (s[0], s[1]):
        a = nb.smoothstep(0.0, width, comp)
        b = nb.smoothstep(1.0, 1.0 - width, comp)
        f = nb.mul(nb.mul(a, b), f) if not isinstance(f, float) else nb.mul(a, b)
    return f


# ----------------------------------------------------------------------------
# 3. Materials
# ----------------------------------------------------------------------------

def centre_material(clock, layer_img, ctrl_img):
    mat = bpy.data.materials.new("M_Centre")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    nb = NodeBuilder(mat.node_tree)
    p = phase_socket(nb, clock)
    uv = nb.node("ShaderNodeTexCoord").outputs["UV"]
    base = image_node(nb, layer_img, uv).outputs["Color"]
    ctrl = image_node(nb, ctrl_img, uv)
    cs = nb.node("ShaderNodeSeparateColor")
    nb.link(ctrl.outputs["Color"], cs.inputs[0])
    chrome, iris, sphere = cs.outputs[0], cs.outputs[1], cs.outputs[2]
    detail = ctrl.outputs["Alpha"]
    uvs = nb.separate(uv)
    u, v = nb.mul(uvs[0], ASPECT), uvs[1]      # image-height units, isotropic
    R = R_SCALE
    lc = (IRIS_UV[0], IRIS_UV[1])              # wordmark / iris centre

    # Musical clock: BEATS beats per loop, 4 beats per bar (all counts integer).
    kick = nb.beat_env(p, BEATS, decay=5.0, attack=40.0)            # every beat
    downbeat = nb.beat_env(p, BARS, decay=4.0, attack=150.0)        # beat 1 of every bar
    phrase = nb.beat_env(p, BARS // 2, decay=3.0, attack=300.0)     # every 2 bars

    # (a) Specular sheen sweeping diagonally across the chrome, once every 2 bars.
    #     Band centre travels s = -1.1 .. +1.1 linearly with fract(); the logo spans
    #     |s| < 0.5, so the wrap happens while the band (and its soft skirt) is outside it.
    diag = nb.add(nb.mul(nb.sub(u, lc[0]), 0.82), nb.mul(nb.sub(v, lc[1]), 0.57))   # ~35 deg
    sweep = nb.math("MULTIPLY_ADD", nb.math("FRACT", nb.math("MULTIPLY_ADD", p, BARS // 2, 0.5)), 2.4, -1.2)
    d = nb.sub(diag, sweep)
    band = nb.math("EXPONENT", nb.mul(nb.mul(d, d), -1.0 / (2 * 0.045 ** 2)))
    band_soft = nb.math("EXPONENT", nb.mul(nb.mul(d, d), -1.0 / (2 * 0.14 ** 2)))
    sheen = nb.mul(nb.add(nb.mul(band, 1.25), nb.mul(band_soft, 0.25)), nb.mul(chrome, AMP))

    # (b) Drifting light across the chrome plus a beat-synced glow on the BETA wordmark.
    drift = nb.loop_noise(nb.combine(nb.mul(u, 2.2), nb.mul(v, 2.2)), p, 1.0, 0.9, 0.3, 2.0)
    pulse = nb.add(nb.mul(kick, 0.16), nb.mul(downbeat, 0.24))
    chrome_mod = nb.mul(nb.add(nb.mul(nb.sub(drift, 0.5), 0.35), pulse), nb.mul(chrome, AMP))

    # (c) Rotating scan light around the wire-frame sphere (one revolution per 4 bars).
    du = nb.sub(u, SPHERE_UV[0])
    dv = nb.sub(v, SPHERE_UV[1])
    theta = nb.math("ARCTAN2", dv, du)
    scan = nb.math("COSINE", nb.sub(theta, nb.angle(p, BARS // 4, 0.0)))
    scan = nb.math("POWER", nb.math("MULTIPLY_ADD", scan, 0.5, 0.5), 6.0)
    sph = nb.mul(nb.add(nb.mul(scan, 1.6), nb.mul(kick, 0.25)), nb.mul(sphere, AMP))

    # (d) Light ring expanding outward through the iris fibres, once per bar, launched
    #     on the downbeat; windowed to zero at both ends (pupil and outer rim).
    di = nb.vmath("DISTANCE", nb.combine(u, v), nb.combine(IRIS_UV[0], IRIS_UV[1]))
    fr = nb.math("FRACT", nb.mul(p, float(BARS)))
    rho = nb.math("MULTIPLY_ADD", nb.math("POWER", fr, 0.7), 0.34 * R, 0.03 * R)   # fast launch, easing out
    win = nb.math("POWER", nb.math("SINE", nb.mul(fr, math.pi)), 1.5)
    dr = nb.sub(di, rho)
    ring = nb.math("EXPONENT", nb.mul(nb.mul(dr, dr), -1.0 / (2 * (0.03 * R) ** 2)))
    iris_pulse = nb.mul(nb.mul(nb.mul(ring, win), 1.6), nb.mul(iris, AMP))

    # (e) Caustic-like light wandering over the dim background detail.
    caus = nb.loop_noise(nb.combine(nb.mul(u, 3.0), nb.mul(v, 3.0)), p, 1.0, 1.2, 1.9, 3.0)
    caus = nb.mul(nb.mul(nb.sub(caus, 0.5), 0.7), nb.mul(detail, AMP))

    # Chrome boosts are scaled by the remaining headroom of the pixel, so bright metal
    # gains sparkle without clipping into flat white and the engraved detail survives.
    bw = nb.node("ShaderNodeRGBToBW")
    nb.link(base, bw.inputs[0])
    headroom = nb.math("POWER", nb.math("SUBTRACT", 1.0, nb.math("MINIMUM", bw.outputs[0], 1.0)), 0.8)
    chrome_gain = nb.mul(nb.add(sheen, chrome_mod), nb.math("MULTIPLY_ADD", headroom, 1.6, 0.15))
    gain = nb.add(nb.add(nb.add(nb.add(chrome_gain, sph), iris_pulse), caus), 1.0)
    gain = nb.math("MAXIMUM", gain, 0.55)
    col = nb.mix_rgb("MULTIPLY", 1.0, base, nb.combine(gain, gain, gain))

    # cool white tint added along the sheen so it reads as reflected light
    tint = nb.mix_rgb("MULTIPLY", 1.0, nb.combine(0.55, 0.85, 1.0), nb.combine(*(nb.mul(sheen, 0.06),) * 3))
    col = nb.mix_rgb("ADD", 1.0, col, tint)

    # (f) Four-point star glint at the logo's crossing, flaring on every 2-bar phrase.
    sx = nb.sub(u, STAR_UV[0])
    sy = nb.sub(v, STAR_UV[1])
    rot = [(1.0, 0.0), (0.7071, 0.7071)]
    star = None
    for (cx, sn), amp_ in zip(rot, (1.0, 0.45)):
        a = nb.add(nb.mul(sx, cx), nb.mul(sy, sn))
        b = nb.sub(nb.mul(sy, cx), nb.mul(sx, sn))
        for (p1, p2) in ((a, b), (b, a)):
            ray = nb.mul(nb.math("EXPONENT", nb.mul(nb.math("ABSOLUTE", p1), -1.0 / (0.035 * R))),
                         nb.math("EXPONENT", nb.mul(nb.mul(p2, p2), -1.0 / (2 * (0.0018 * R) ** 2))))
            ray = nb.mul(ray, amp_)
            star = ray if star is None else nb.add(star, ray)
    rr = nb.add(nb.mul(sx, sx), nb.mul(sy, sy))
    core = nb.math("EXPONENT", nb.mul(rr, -1.0 / (2 * (0.006 * R) ** 2)))
    star = nb.add(star, nb.mul(core, 1.5))
    star = nb.mul(nb.mul(star, nb.add(phrase, nb.mul(kick, 0.12))), 1.0 * AMP)
    col = nb.mix_rgb("ADD", 1.0, col, nb.mix_rgb("MULTIPLY", 1.0, nb.combine(0.75, 0.95, 1.0),
                                                   nb.combine(star, star, star)))

    additive_output(nb, col)
    return mat


def foam_material(name, clock, layer_img, seed_offset):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    nb = NodeBuilder(mat.node_tree)
    p = phase_socket(nb, clock)
    uv = nb.node("ShaderNodeTexCoord").outputs["UV"]
    base = image_node(nb, layer_img, uv).outputs["Color"]
    s = nb.separate(uv)
    # Refraction-like light gliding over the cell membranes.
    n = nb.loop_noise(nb.combine(nb.mul(s[0], 4.0), nb.mul(s[1], 4.0)), p, 1.0, 0.8, seed_offset, 3.0)
    g = nb.math("MULTIPLY_ADD", nb.sub(n, 0.5), 1.4 * AMP, 1.0)
    offbeat = nb.beat_env(p, BEATS, decay=6.0, attack=40.0, offset=0.5)   # "and" of every beat
    g = nb.add(g, nb.mul(offbeat, 0.35 * AMP))
    g = nb.math("MAXIMUM", g, 0.5)
    col = nb.mix_rgb("MULTIPLY", 1.0, base, nb.combine(g, g, g))
    additive_output(nb, col)
    return mat


def haze_material(name, clock, color, strength, scale, seed_offset, clear_centre):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    nb = NodeBuilder(mat.node_tree)
    p = phase_socket(nb, clock)
    uv = nb.node("ShaderNodeTexCoord").outputs["UV"]      # plane UV, 0..1
    s = nb.separate(uv)
    # plane is 16:9-ish; stretch u so the haze cells are round
    x = nb.mul(s[0], 1.8)
    y = s[1]
    # domain warp with a second periodic noise for a smoke-like flow
    warp = nb.loop_noise(nb.combine(nb.mul(x, 1.3), nb.mul(y, 1.3)), p, 1.0, 0.45, seed_offset + 2.0, 2.0)
    sway = nb.mul(nb.math("SINE", nb.angle(p, 1, seed_offset)), 0.06 * AMP)
    xw = nb.add(nb.add(x, nb.mul(warp, 0.35)), sway)
    yw = nb.add(y, nb.mul(warp, 0.25))
    n = nb.loop_noise(nb.combine(xw, yw), p, scale, 0.65 * AMP + 1e-4, seed_offset, 2.0, 0.45)
    dens = nb.math("POWER", nb.smoothstep(0.30, 0.85, n), 1.6)
    # keep the subject readable: thin the haze over the logo
    dc = nb.vmath("DISTANCE", nb.combine(s[0], s[1]), nb.combine(0.5, 0.5))
    keep = nb.math("MULTIPLY_ADD", nb.smoothstep(0.05, 0.32, dc), 1.0 - clear_centre, clear_centre)
    # fade at plane borders
    edge = edge_feather(nb, uv, 0.12)
    dens = nb.mul(nb.mul(dens, keep), edge)
    col = nb.mix_rgb("MULTIPLY", 1.0, nb.combine(*color), nb.combine(dens, dens, dens))
    additive_output(nb, col, strength)
    return mat


def mote_material():
    mat = bpy.data.materials.new("M_Mote")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    nb = NodeBuilder(mat.node_tree)
    uv = nb.node("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="mote_uv").outputs["Vector"]
    d = nb.vmath("DISTANCE", uv, nb.combine(0.5, 0.5))
    disc = nb.math("EXPONENT", nb.mul(nb.mul(d, d), -1.0 / (2 * 0.16 ** 2)))
    fade = nb.node("ShaderNodeAttribute", attribute_type="INSTANCER", attribute_name="mote_fade").outputs["Fac"]
    tint = nb.node("ShaderNodeAttribute", attribute_type="INSTANCER", attribute_name="mote_tint").outputs["Fac"]
    i = nb.mul(disc, fade)
    c = nb.mix_rgb("MIX", tint, nb.combine(0.62, 0.86, 1.0), nb.combine(0.95, 0.97, 1.0))
    col = nb.mix_rgb("MULTIPLY", 1.0, c, nb.combine(i, i, i))
    additive_output(nb, col, 1.0)
    return mat


# ----------------------------------------------------------------------------
# 4. Floating motes (Geometry Nodes, deterministic and periodic)
# ----------------------------------------------------------------------------

def motes_object(mat, count=170):
    mesh = bpy.data.meshes.new("MotesBase")
    obj = bpy.data.objects.new("Motes", mesh)
    bpy.context.scene.collection.objects.link(obj)
    mod = obj.modifiers.new("Motes", "NODES")
    g = bpy.data.node_groups.new("MotesGN", "GeometryNodeTree")
    g.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    g.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    mod.node_group = g
    nb = NodeBuilder(g)
    nb.node("NodeGroupInput")
    gout = nb.node("NodeGroupOutput")

    st = nb.node("GeometryNodeInputSceneTime")
    p = nb.math("DIVIDE", st.outputs["Frame"], float(N))

    def rnd(seed, lo=0.0, hi=1.0):
        r = nb.node("FunctionNodeRandomValue", data_type="FLOAT")
        r.inputs[2].default_value = lo
        r.inputs[3].default_value = hi
        r.inputs["Seed"].default_value = seed
        return r.outputs[1]

    def rint(seed, lo, hi):
        r = nb.node("FunctionNodeRandomValue", data_type="INT")
        r.inputs[4].default_value = lo
        r.inputs[5].default_value = hi
        r.inputs["Seed"].default_value = seed
        return r.outputs[2]

    z = nb.math("POWER", rnd(2), 0.8)
    z = nb.math("MULTIPLY_ADD", z, 6.0, -3.0)                    # depth -3 .. +3
    persp = nb.math("DIVIDE", nb.sub(CAM_DIST, z), CAM_DIST)      # half-height of view at z
    # vertical travel: wraps once per loop over a span larger than the view
    span = nb.mul(nb.mul(persp, 2.6), rnd(3, 1.0, 1.6))
    f = nb.math("FRACT", nb.add(rnd(4), nb.mul(p, 2.0 * AMP)))     # two rises per loop
    y = nb.mul(nb.sub(f, 0.5), span)
    # horizontal sway with an integer number of cycles per loop
    m = rint(6, 1, 2)
    sway_ang = nb.math("MULTIPLY_ADD", nb.mul(p, m), 2 * math.pi, nb.mul(rnd(5), 2 * math.pi))
    sway = nb.mul(nb.math("SINE", sway_ang), nb.mul(rnd(7, 0.01, 0.05), AMP))
    x = nb.add(nb.mul(nb.mul(rnd(1, -1.0, 1.0), persp), 1.9), sway)

    pts = nb.node("GeometryNodePoints")
    pts.inputs["Count"].default_value = count
    nb.link(nb.combine(x, y, z), pts.inputs["Position"])

    # brightness: invisible at the wrap (fract seam), gentle twinkle, dimmer when defocused
    edge = nb.math("POWER", nb.math("SINE", nb.mul(f, math.pi)), 0.6)
    k = rint(8, 2, 4)
    tw_ang = nb.math("MULTIPLY_ADD", nb.mul(p, k), 2 * math.pi, nb.mul(rnd(9), 2 * math.pi))
    tw = nb.math("MULTIPLY_ADD", nb.math("SINE", tw_ang), 0.35 * AMP, 0.65)
    near = nb.smoothstep(1.0, 3.0, z)                              # close motes: big & soft
    far = nb.smoothstep(-1.0, -3.0, z)
    bright = nb.mul(nb.mul(edge, tw), nb.math("POWER", rnd(10), 2.0))   # mostly faint, few bright
    bright = nb.mul(bright, 0.35)
    bright = nb.mul(bright, nb.math("MULTIPLY_ADD", near, -0.85, 1.0))
    bright = nb.mul(bright, nb.math("MULTIPLY_ADD", far, -0.4, 1.0))
    size = nb.mul(rnd(11, 0.008, 0.018), nb.math("MULTIPLY_ADD", near, 2.5, 1.0))

    store1 = nb.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
    store1.inputs["Name"].default_value = "mote_fade"
    nb.link(pts.outputs[0], store1.inputs["Geometry"])
    nb.link(bright, store1.inputs["Value"])
    store2 = nb.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
    store2.inputs["Name"].default_value = "mote_tint"
    nb.link(store1.outputs[0], store2.inputs["Geometry"])
    nb.link(rnd(12), store2.inputs["Value"])

    grid = nb.node("GeometryNodeMeshGrid")
    grid.inputs["Size X"].default_value = 1.0
    grid.inputs["Size Y"].default_value = 1.0
    grid.inputs["Vertices X"].default_value = 2
    grid.inputs["Vertices Y"].default_value = 2
    store_uv = nb.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT_VECTOR", domain="CORNER")
    store_uv.inputs["Name"].default_value = "mote_uv"
    nb.link(grid.outputs["Mesh"], store_uv.inputs["Geometry"])
    nb.link(grid.outputs["UV Map"], store_uv.inputs["Value"])
    setm = nb.node("GeometryNodeSetMaterial")
    setm.inputs["Material"].default_value = mat
    nb.link(store_uv.outputs[0], setm.inputs["Geometry"])

    inst = nb.node("GeometryNodeInstanceOnPoints")
    nb.link(store2.outputs[0], inst.inputs["Points"])
    nb.link(setm.outputs[0], inst.inputs["Instance"])
    nb.link(size, inst.inputs["Scale"])
    nb.link(inst.outputs[0], gout.inputs[0])
    obj.data.materials.append(mat)
    return obj


# ----------------------------------------------------------------------------
# 5. Scene assembly
# ----------------------------------------------------------------------------

def image_plane(name, mat, z, size=IMG_SIZE, aspect=None):
    """Plane at depth z, scaled so it projects to the same frame area as at z=0.

    UVs stay 0..1 over the reference; the plane is enlarged by OVERSCAN and the UVs
    scaled about the centre to match, so the image keeps its size but the mirrored
    margin covers the frame edge during camera drift and layer float.
    """
    aspect = ASPECT if aspect is None else aspect
    s = (CAM_DIST - z) / CAM_DIST * OVERSCAN
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, z))
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = (size * s * aspect, size * s, 1.0)
    ob.data.materials.append(mat)
    ob.visible_shadow = False
    for loop_uv in ob.data.uv_layers.active.data:
        loop_uv.uv = ((loop_uv.uv[0] - 0.5) * OVERSCAN + 0.5, (loop_uv.uv[1] - 0.5) * OVERSCAN + 0.5)
    return ob


def periodic_expr(amp, k, phase0):
    """amp * sin(2 pi k frame / N + phase0) -- integer k keeps it N-periodic."""
    assert float(k).is_integer()
    return f"{amp * AMP:.6f} * sin({2 * math.pi * k:.10f} * frame / {N} + {phase0:.6f})"


def downbeat_expr(amp, decay=4.0, attack=12.0):
    """Driver expression of the same percussive envelope as NodeBuilder.beat_env, per bar.

    Zero at both ends of each bar, so the camera push-in joins seamlessly.
    """
    bs = np.linspace(0, 1, 20001)
    peak = float(((1 - np.exp(-attack * bs)) * (np.exp(-decay * bs) - math.exp(-decay))).max())
    b = f"(frame * {BARS} / {N} - floor(frame * {BARS} / {N}))"
    return (f"{amp * AMP / peak:.8f} * (1 - exp(-{attack} * {b})) * "
            f"(exp(-{decay} * {b}) - {math.exp(-decay):.10f})")


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

    img_c = load_linear(paths["layer_centre"])
    img_tr = load_linear(paths["layer_foam_tr"])
    img_bl = load_linear(paths["layer_foam_bl"])
    img_ctrl = load_linear(paths["control"])
    for im in (img_c, img_tr, img_bl, img_ctrl):
        im.pack()

    # --- central subject, static in place (only light moves over it)
    image_plane("Subject", centre_material(clock, img_c, img_ctrl), 0.0)

    # --- foam clusters: nearer the camera (parallax) and gently floating
    for name, img, z, pivot_uv, ph in (("FoamTR", img_tr, 0.9, (0.86, 0.82), 0.0),
                                       ("FoamBL", img_bl, 0.6, (0.10, 0.22), 2.1)):
        s = (CAM_DIST - z) / CAM_DIST
        piv = bpy.data.objects.new(name + "_Pivot", None)
        scn.collection.objects.link(piv)
        px_, py_ = (pivot_uv[0] - 0.5) * IMG_SIZE * ASPECT * s, (pivot_uv[1] - 0.5) * IMG_SIZE * s
        pl = image_plane(name, foam_material("M_" + name, clock, img, 3.7 + ph), z)
        pl.parent = piv
        pl.location = (-px_, -py_, 0.0)
        add_driver(piv, "location", f"{px_:.6f} + " + periodic_expr(0.010, 1, ph), 0)
        add_driver(piv, "location", f"{py_:.6f} + " + periodic_expr(0.008, 1, ph + 1.3), 1)
        add_driver(piv, "location", f"{z:.6f} + " + periodic_expr(0.0, 1, 0.0), 2)
        piv.location[2] = z
        add_driver(piv, "rotation_euler", periodic_expr(math.radians(0.45), 1, ph + 0.5), 2)
        for ax in range(2):
            add_driver(piv, "scale", "1.0 + " + periodic_expr(0.008, 1, ph + 2.4), ax)

    # --- haze: one veil behind the subject, one thin veil in front
    vis_w = 2 * 16 / 9
    for name, z, col, strength, scale, off, clear in (
            ("HazeBack", -2.0, (0.0022, 0.0042, 0.0052), 1.0, 2.2, 0.0, 0.15),
            ("HazeFront", 1.6, (0.0012, 0.0022, 0.0027), 1.0, 3.0, 5.3, 0.0)):
        s = (CAM_DIST - z) / CAM_DIST
        mat = haze_material("M_" + name, clock, col, strength, scale, off, clear)
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, z))
        ob = bpy.context.active_object
        ob.name = name
        ob.scale = (vis_w * 1.25 * s, 2 * 1.25 * s, 1)
        ob.data.materials.append(mat)
        ob.hide_render = CFG["identity"]

    motes_object(mote_material()).hide_render = CFG["identity"]

    # --- camera: stationary framing with a slow periodic drift (Lissajous 1:2)
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_height = SENSOR_H
    cam_data.lens = FOCAL
    cam_data.clip_start = 0.1
    cam_data.clip_end = 100
    cam = bpy.data.objects.new("Camera", cam_data)
    scn.collection.objects.link(cam)
    scn.camera = cam
    add_driver(cam, "location", periodic_expr(0.030, 1, 0.0), 0)
    add_driver(cam, "location", periodic_expr(0.016, 2, 0.9), 1)
    add_driver(cam, "location", f"{CAM_DIST} + " + periodic_expr(0.06, 1, 1.7)
               + " - " + downbeat_expr(0.035), 2)

    # --- render settings
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
    cy.transparent_max_bounces = 32
    cy.caustics_reflective = cy.caustics_refractive = False
    cy.pixel_filter_type = "BLACKMAN_HARRIS"
    cy.filter_width = 1.5
    scn.render.use_motion_blur = False
    scn.render.dither_intensity = 1.0
    scn.render.film_transparent = False
    scn.view_settings.view_transform = "Standard"   # reproduces the reference's own grade exactly
    scn.view_settings.look = "None"
    scn.view_settings.exposure = 0.0
    scn.view_settings.gamma = 1.0
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.render.image_settings.color_depth = "16"
    scn.render.image_settings.compression = 15
    scn["loop_frames"] = N
    scn["loop_fps"] = CFG["fps"]

    build_compositor(scn)
    return scn


def build_compositor(scn):
    scn.use_nodes = True
    t = scn.node_tree
    t.nodes.clear()
    rl = t.nodes.new("CompositorNodeRLayers")
    glare = t.nodes.new("CompositorNodeGlare")
    glare.glare_type = "FOG_GLOW"
    glare.quality = "HIGH"
    glare.threshold = 0.85
    glare.size = 8
    glare.mix = -0.90 if AMP else -1.0      # restrained bloom on the chrome highlights
    lens = t.nodes.new("CompositorNodeLensdist")
    lens.inputs["Distortion"].default_value = 0.0
    lens.inputs["Dispersion"].default_value = 0.004 * AMP
    out = t.nodes.new("CompositorNodeComposite")
    t.links.new(rl.outputs["Image"], glare.inputs["Image"])
    t.links.new(glare.outputs["Image"], lens.inputs["Image"])
    t.links.new(lens.outputs["Image"], out.inputs["Image"])


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------

def main():
    if not os.path.isfile(CFG["ref"]):
        raise SystemExit(f"reference image not found: {CFG['ref']}")
    print(f"[loop] reference={CFG['ref']} N={N} fps={CFG['fps']} res={CFG['res']} identity={CFG['identity']}")
    paths = build_layers(CFG["ref"])
    scn = build_scene(paths)
    for fc, expr in DRIVERS:
        if not fc.driver.is_simple_expression:
            raise SystemExit(f"driver is not a simple expression (would need auto-exec): {expr}")
    if CFG["save"]:
        bpy.ops.file.pack_all()
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
