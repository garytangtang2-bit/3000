"""
Measure temporal continuity of an encoded loop, including the wrap last -> first frame.

    python3 check_video_seam.py VIDEO [--fps 30 --beats-per-loop 32]

Decodes at 960x540 (enough for motion statistics), computes for every playback
transition i -> i+1 (with the wrap N-1 -> 0) the mean absolute frame difference
(~velocity) and the second difference |f[i+1] - 2 f[i] + f[i-1]| (~acceleration).
The wrap lands on a bar downbeat, so it is compared with the other downbeat
transitions as well as with all transitions.
"""
import json
import subprocess
import sys

import numpy as np

path = sys.argv[1]
beats = 32
if "--beats-per-loop" in sys.argv:
    beats = int(sys.argv[sys.argv.index("--beats-per-loop") + 1])
W, H = 960, 540

probe = json.loads(subprocess.check_output(
    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
     "stream=codec_name,width,height,r_frame_rate,nb_read_frames,pix_fmt,duration",
     "-of", "json", path]))["streams"][0]
raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H}:flags=area",
                               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
F = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(np.float32)
N = len(F)
nxt = np.roll(np.arange(N), -1)
prv = np.roll(np.arange(N), 1)
vel = np.array([np.abs(F[nxt[i]] - F[i]).mean() for i in range(N)])            # transition i -> i+1
acc = np.array([np.abs(F[nxt[i]] - 2 * F[i] + F[prv[i]]).mean() for i in range(N)])  # around frame i

bar = N // (beats // 4)
down_idx = [k * bar - 1 for k in range(1, beats // 4 + 1)]       # transitions into a downbeat
wrap = N - 1
others = [i for i in down_idx if i != wrap]
res = {
    "file": path, "codec": probe.get("codec_name"), "size": f"{probe['width']}x{probe['height']}",
    "pix_fmt": probe.get("pix_fmt"), "fps": probe.get("r_frame_rate"), "frames": N,
    "wrap_velocity": float(vel[wrap]),
    "downbeat_velocity_other_bars": [round(float(vel[i]), 4) for i in others],
    "all_velocity_median": float(np.median(vel)), "all_velocity_max": float(vel.max()),
    "wrap_accel_at_frame0": float(acc[0]),
    "downbeat_accel_other_bars": [round(float(acc[i + 1]), 4) for i in others],
    "all_accel_median": float(np.median(acc)), "all_accel_max": float(acc.max()),
    "wrap_velocity_rank_desc": int((vel > vel[wrap]).sum()) + 1,
    "first_vs_last_frame_mad": float(np.abs(F[0] - F[-1]).mean()),
}
print(json.dumps(res, indent=1))
