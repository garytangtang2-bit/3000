"""
Numerical loop-boundary validation, run inside Blender against the saved project:

    blender -b looping_background.blend -P validate_loop.py -- OUT_JSON [--render-dir DIR]

Checks, for a loop of N unique frames (0..N-1):
  * every driven channel (bubble-cluster pivots) and the shader loop clock ->
    value, velocity and acceleration at the boundary (frame N vs 0, finite
    differences across the seam);
  * that every driver is a "simple expression" (animates without Python auto-exec);
  * optional renders (dither off, 960x540):
      - frame N vs frame 0 must be identical;
      - the central region (BETA wordmark and its lines) must not change at all
        between distant frames, i.e. only the bubbles move.
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
LOOP_FREQS = (1, 3, 4)          # cluster float, bubble drift, bubble swell (cycles per loop)


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
    p = bpy.data.node_groups["LoopClock"].nodes["Phase"].outputs[0].default_value
    for k in LOOP_FREQS:            # what the shaders actually consume
        vals[f"shader sin(2pi*{k}*p)"] = math.sin(2 * math.pi * k * p)
        vals[f"shader cos(2pi*{k}*p)"] = math.cos(2 * math.pi * k * p)
    return vals


frames = [-2, -1, 0, 1, 2, N - 2, N - 1, N, N + 1, N + 2]
S = {f: sample_channels(f) for f in frames}
chan = {}
worst = {"value": 0.0, "velocity": 0.0, "acceleration": 0.0}
for k in S[0]:
    v = {f: S[f][k] for f in frames}
    vel_a, vel_b = v[0] - v[-1], v[N] - v[N - 1]
    acc_a, acc_b = v[1] - 2 * v[0] + v[-1], v[N + 1] - 2 * v[N] + v[N - 1]
    chan[k] = {"abs_diff": abs(v[N] - v[0]), "vel_0": vel_a, "vel_N": vel_b, "acc_0": acc_a, "acc_N": acc_b}
    worst["value"] = max(worst["value"], abs(v[N] - v[0]))
    worst["velocity"] = max(worst["velocity"], abs(vel_a - vel_b))
    worst["acceleration"] = max(worst["acceleration"], abs(acc_a - acc_b))
report["channels"] = {
    "count": len(chan),
    "moving_at_boundary": sum(abs(c["vel_0"]) > 1e-7 for c in chan.values()),
    "worst_abs_diff_frameN_vs_frame0": worst,
    "largest_velocity_mismatch": sorted(((abs(c["vel_0"] - c["vel_N"]), k) for k, c in chan.items()),
                                        reverse=True)[:3],
}

non_simple = []
for idb in list(bpy.data.objects) + list(bpy.data.node_groups):
    ad = getattr(idb, "animation_data", None)
    if ad:
        for fc in ad.drivers:
            if not fc.driver.is_simple_expression:
                non_simple.append(f"{idb.name}:{fc.data_path}")
report["non_simple_drivers"] = non_simple

if render_dir:
    os.makedirs(render_dir, exist_ok=True)
    scn.render.resolution_x, scn.render.resolution_y = 960, 540
    scn.render.dither_intensity = 0.0
    imgs = {}
    for f in (0, N, N // 3):
        scn.frame_set(f)
        path = os.path.join(render_dir, f"check_{f:05d}.png")
        scn.render.filepath = path
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(path)
        a = np.empty(im.size[0] * im.size[1] * 4, np.float32)
        im.pixels.foreach_get(a)
        imgs[f] = a.reshape(im.size[1], im.size[0], 4)[::-1, :, :3] * 255.0
    d = np.abs(imgs[0] - imgs[N])
    report["pixels_frame0_vs_frameN"] = {"max_abs_diff_0_255": float(d.max()), "mean_abs_diff_0_255": float(d.mean())}
    # central box around the wordmark (no bubbles there)
    h, w = imgs[0].shape[:2]
    box = (slice(int(0.30 * h), int(0.78 * h)), slice(int(0.30 * w), int(0.74 * w)))
    dc = np.abs(imgs[0][box] - imgs[N // 3][box])
    report["central_region_static"] = {"frames": [0, N // 3], "box_xy_frac": [0.30, 0.74, 0.30, 0.78],
                                       "max_abs_diff_0_255": float(dc.max()), "mean_abs_diff_0_255": float(dc.mean())}
    db = np.abs(imgs[0] - imgs[N // 3])
    report["whole_frame_change_0_vs_N3"] = {"mean_abs_diff_0_255": float(db.mean()),
                                            "pixels_changed_gt_2_levels_pct": float((db.max(2) > 2).mean() * 100)}

with open(out_json, "w") as fh:
    json.dump(report, fh, indent=2)
print("[validate]", json.dumps(report, indent=1))
