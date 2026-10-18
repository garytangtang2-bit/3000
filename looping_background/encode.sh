#!/usr/bin/env bash
# Encode rendered PNG frames (0..N-1, the N unique loop frames) into the deliverables.
# Usage: ./encode.sh [preview|final|all]
# Frame N (== frame 0) is never rendered or encoded, so playback wraps N-1 -> 0 cleanly.
set -euo pipefail
cd "$(dirname "$0")"
FPS=30
# Colour tags: frames are sRGB-encoded (Standard view transform) -> tag as BT.709.
TAGS=(-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv)

preview() {
  ffmpeg -y -hide_banner -loglevel error -framerate $FPS -i frames/960x540/%05d.png \
    -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -g $FPS -bf 2 \
    "${TAGS[@]}" -movflags +faststart preview.mp4
}

final() {
  local src=frames/3840x2160/%05d.png
  # H.264 High @ L5.2, near-transparent quality. aq-mode 3 protects dark gradients
  # from banding; 1 s closed GOPs keep scrubbing / cue jumps in VJ software instant.
  ffmpeg -y -hide_banner -loglevel error -framerate $FPS -i $src \
    -c:v libx264 -preset slow -crf 14 -profile:v high -level:v 5.2 -pix_fmt yuv420p \
    -g $FPS -keyint_min $FPS -sc_threshold 0 -x264-params "aq-mode=3:aq-strength=0.9" \
    "${TAGS[@]}" -movflags +faststart looping_background_4k.mp4
  # 10-bit 4:2:2 ProRes 422 HQ master (from 16-bit PNG).
  ffmpeg -y -hide_banner -loglevel error -framerate $FPS -i $src \
    -c:v prores_ks -profile:v 3 -vendor apl0 -pix_fmt yuv422p10le \
    "${TAGS[@]}" looping_background_master.mov
  # HAP Q for VJ playback (GPU-decoded in Resolume / VDMX / TouchDesigner / MadMapper).
  ffmpeg -y -hide_banner -loglevel error -framerate $FPS -i $src \
    -c:v hap -format hap_q -chunks 4 looping_background_4k_hapq.mov
}

case "${1:-all}" in
  preview) preview ;;
  final) final ;;
  all) preview; final ;;
esac
