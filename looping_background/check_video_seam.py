"""
Measure temporal continuity of an encoded loop, including the wrap last -> first frame.

    python3 check_video_seam.py VIDEO

Decodes at 960x540 (enough for motion statistics), computes for every playback
transition i -> i+1 (with the wrap N-1 -> 0) the mean absolute frame difference
(~velocity) and the second difference |f[i+1] - 2 f[i] + f[i-1]| (~acceleration).
The wrap always lands on a keyframe (frame 0), so it is compared with the other
keyframe transitions (every 30 frames) as well as with all transitions.
"""
import json
import subprocess
import sys

import numpy as np

path = sys.argv[1]
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

gop = 30
gop_idx = [i for i in range(gop - 1, N, gop)]                 # transitions into a keyframe
inner = np.delete(np.arange(N), gop_idx)
res = {
    "file": path, "codec": probe.get("codec_name"), "size": f"{probe['width']}x{probe['height']}",
    "pix_fmt": probe.get("pix_fmt"), "fps": probe.get("r_frame_rate"), "frames": N,
    "wrap_velocity_last_to_first": float(vel[N - 1]),
    "keyframe_transition_velocity_mean": float(vel[gop_idx[:-1]].mean()),
    "inner_transition_velocity_median": float(np.median(vel[inner])),
    "all_velocity_max": float(vel.max()),
    "wrap_velocity_rank_desc": int((vel > vel[N - 1]).sum()) + 1,
    "wrap_accel_at_first_frame": float(acc[0]),
    "all_accel_median": float(np.median(acc)), "all_accel_max": float(acc.max()),
}
print(json.dumps(res, indent=1))
