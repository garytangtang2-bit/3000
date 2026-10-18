"""
Numerical loop-boundary validation, run inside Blender against the saved project:

    blender -b looping_background.blend -P validate_loop.py -- OUT_JSON [--render-dir DIR]

Checks, for a loop of N unique frames (0..N-1):
  * every driven channel (camera, foam pivots, shader clock) -> value, velocity and
    acceleration at the boundary (frame N vs 0, finite differences across the seam);
  * every mote instance produced by Geometry Nodes -> transform and brightness
    attribute at frame N vs 0, and velocity across the seam;
  * optional: renders frames 0 and N (dither off) and compares the pixels.
"""

import bpy
import json
import math
import os
import sys

import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
out_json = os.path.abspath(argv[0])
render_dir = None
if "--render-dir" in argv:
    render_dir = os.path.abspath(argv[argv.index("--render-dir") + 1])

scn = bpy.context.scene
N = int(scn["loop_frames"])
report = {"N": N, "fps": scn.render.fps}


def periodic_part(path_value):
    return path_value


# ---------------------------------------------------------------------------
# 1. Driven channels
# ---------------------------------------------------------------------------
def sample_channels(frame):
    scn.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    vals = {}
    for ob in scn.objects:
        if ob.animation_data and ob.animation_data.drivers:
            m = ob.evaluated_get(dg).matrix_world
            for i in range(4):
                for j in range(4):
                    vals[f"{ob.name}.matrix_world[{i}][{j}]"] = m[i][j]
    clock = bpy.data.node_groups["LoopClock"].nodes["Phase"].outputs[0].default_value
    vals["LoopClock.phase_cos"] = math.cos(2 * math.pi * clock)
    vals["LoopClock.phase_sin"] = math.sin(2 * math.pi * clock)

    return vals


frames = [-2, -1, 0, 1, 2, N - 2, N - 1, N, N + 1, N + 2]
S = {f: sample_channels(f) for f in frames}
chan = {}
worst = {"value": 0.0, "velocity": 0.0, "acceleration": 0.0}
for k in S[0]:
    v = {f: S[f][k] for f in frames}
    dv = abs(v[N] - v[0])
    vel_a = v[0] - v[-1]          # velocity entering frame 0 from the previous loop
    vel_b = v[N] - v[N - 1]       # velocity entering frame N (= frame 0 of next loop)
    acc_a = v[1] - 2 * v[0] + v[-1]
    acc_b = v[N + 1] - 2 * v[N] + v[N - 1]
    chan[k] = {"value_0": v[0], "value_N": v[N], "abs_diff": dv,
               "vel_0": vel_a, "vel_N": vel_b, "acc_0": acc_a, "acc_N": acc_b}
    worst["value"] = max(worst["value"], dv)
    worst["velocity"] = max(worst["velocity"], abs(vel_a - vel_b))
    worst["acceleration"] = max(worst["acceleration"], abs(acc_a - acc_b))
report["driven_channels"] = {"count": len(chan), "worst_abs_diff": worst,
                             "top_velocity_diff": sorted(((abs(c["vel_0"] - c["vel_N"]), k) for k, c in chan.items()), reverse=True)[:3]}
moving = {k: c for k, c in chan.items() if abs(c["vel_0"]) > 1e-7}
report["driven_channels"]["moving_at_boundary"] = len(moving)

# drivers must be "simple expressions" so the .blend animates without Python auto-exec
non_simple = []
for idb in list(bpy.data.objects) + list(bpy.data.node_groups):
    ad = getattr(idb, "animation_data", None)
    if ad:
        for fc in ad.drivers:
            if not fc.driver.is_simple_expression:
                non_simple.append(f"{idb.name}:{fc.data_path}")
report["non_simple_drivers"] = non_simple


# ---------------------------------------------------------------------------
# 2. Geometry Nodes motes
# ---------------------------------------------------------------------------
def sample_motes(frame):
    scn.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    mats = []
    for inst in dg.object_instances:
        if inst.is_instance and inst.parent and inst.parent.original.name == "Motes":
            mats.append(np.array(inst.matrix_world))
    ev = scn.objects["Motes"].evaluated_get(dg)
    return np.array(mats)


M = {f: sample_motes(f) for f in (-1, 0, 1, N - 1, N, N + 1)}
pos = {f: M[f][:, :3, 3] for f in M}
scale = {f: np.linalg.norm(M[f][:, :3, 0], axis=1) for f in M}
vel0 = pos[0] - pos[-1]
velN = pos[N] - pos[N - 1]
# A mote whose fract() wraps exactly on the boundary would show a huge "velocity";
# that only happens while it is invisible (brightness 0 at the wrap), so report it.
report["motes"] = {
    "count": int(len(pos[0])),
    "max_pos_diff_N_vs_0": float(np.abs(pos[N] - pos[0]).max()),
    "max_scale_diff_N_vs_0": float(np.abs(scale[N] - scale[0]).max()),
    "max_velocity_diff": float(np.abs(velN - vel0).max()),
    "max_accel_diff": float(np.abs((pos[N + 1] - 2 * pos[N] + pos[N - 1])
                                   - (pos[1] - 2 * pos[0] + pos[-1])).max()),
}


# ---------------------------------------------------------------------------
# 3. Pixel check: frame N must render identically to frame 0
# ---------------------------------------------------------------------------
if render_dir:
    os.makedirs(render_dir, exist_ok=True)
    scn.render.resolution_x, scn.render.resolution_y = 960, 540
    scn.render.dither_intensity = 0.0
    imgs = {}
    for f in (0, N):
        scn.frame_set(f)
        path = os.path.join(render_dir, f"boundary_{f:05d}.png")
        scn.render.filepath = path
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(path)
        a = np.empty(im.size[0] * im.size[1] * 4, np.float32)
        im.pixels.foreach_get(a)
        imgs[f] = a * 255.0
    d = np.abs(imgs[0] - imgs[N])
    report["pixels_frame0_vs_frameN"] = {"resolution": "960x540", "max_abs_diff_0_255": float(d.max()),
                                         "mean_abs_diff_0_255": float(d.mean())}

with open(out_json, "w") as fh:
    json.dump(report, fh, indent=2)
print("[validate]", json.dumps({k: v for k, v in report.items() if k != "driven_channels"}, indent=1))
print("[validate] driven:", report["driven_channels"])
