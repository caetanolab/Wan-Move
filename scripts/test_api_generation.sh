#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8181}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
POLL_SECONDS="${POLL_SECONDS:-10}"
SAMPLE_STEPS="${SAMPLE_STEPS:-40}"
FRAME_NUM="${FRAME_NUM:-81}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/api-test}"

cd "$(dirname "$0")/.."

INPUT_DIR="$OUTPUT_DIR/input"
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

IMAGE_PATH="$INPUT_DIR/input.ppm"
TRACKS_PATH="$INPUT_DIR/tracks.npy"
VISIBILITY_PATH="$INPUT_DIR/visibility.npy"
CREATE_RESPONSE="$OUTPUT_DIR/create-response.json"
JOB_RESPONSE="$OUTPUT_DIR/job-response.json"

python3 - "$IMAGE_PATH" "$TRACKS_PATH" "$VISIBILITY_PATH" "$FRAME_NUM" <<'PY'
import struct
import sys
from pathlib import Path

image_path = Path(sys.argv[1])
tracks_path = Path(sys.argv[2])
visibility_path = Path(sys.argv[3])
frame_num = int(sys.argv[4])

width = 480
height = 832

pixels = bytearray()
for y in range(height):
    for x in range(width):
        r = 225
        g = 235
        b = 245
        if 145 <= x <= 255 and 345 <= y <= 455:
            r, g, b = 235, 55, 55
        if 20 <= x <= 460 and (y in (20, 812)):
            r, g, b = 40, 40, 40
        if 20 <= y <= 812 and (x in (20, 460)):
            r, g, b = 40, 40, 40
        pixels.extend((r, g, b))

with image_path.open("wb") as f:
    f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
    f.write(pixels)


def write_npy(path, shape, dtype_descr, values):
    if dtype_descr == "<f4":
        payload = b"".join(struct.pack("<f", float(v)) for v in values)
    elif dtype_descr == "|b1":
        payload = bytes(1 if bool(v) else 0 for v in values)
    else:
        raise ValueError(dtype_descr)

    shape_text = ", ".join(str(x) for x in shape)
    if len(shape) == 1:
        shape_text += ","
    header = (
        "{'descr': '"
        + dtype_descr
        + "', 'fortran_order': False, 'shape': ("
        + shape_text
        + "), }"
    )
    header_bytes = header.encode("latin1")
    pad_len = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * pad_len + b"\n"

    with path.open("wb") as f:
        f.write(b"\x93NUMPY")
        f.write(bytes([1, 0]))
        f.write(struct.pack("<H", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)


tracks = []
visibility = []
for i in range(frame_num):
    t = i / max(frame_num - 1, 1)
    # Shape [1, F, N, 2], with two visible points on the red block.
    x0 = 185 + 170 * t
    y0 = 390 + 120 * t
    x1 = 220 + 130 * t
    y1 = 420 + 80 * t
    tracks.extend([x0, y0, x1, y1])
    visibility.extend([True, True])

write_npy(tracks_path, (1, frame_num, 2, 2), "<f4", tracks)
write_npy(visibility_path, (1, frame_num, 2), "|b1", visibility)
PY

echo "Generated test inputs:"
echo "  image:      $IMAGE_PATH"
echo "  tracks:     $TRACKS_PATH"
echo "  visibility: $VISIBILITY_PATH"

echo "Checking API health at $API_URL/health"
curl --fail --silent --show-error "$API_URL/health" > "$OUTPUT_DIR/health.json"
python3 - "$OUTPUT_DIR/health.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    health = json.load(f)
print("Health:", json.dumps(health, sort_keys=True))
if health.get("status") != "ok":
    raise SystemExit("API health status is not ok")
PY

PROMPT="A red square block on a pale background moves diagonally down and to the right, following the supplied trajectory points. Keep the scene simple and the motion easy to inspect."

echo "Submitting trajectory generation job"
curl --fail --silent --show-error \
  -X POST "$API_URL/v1/generations" \
  -F "prompt=$PROMPT" \
  -F "image=@$IMAGE_PATH;type=image/x-portable-pixmap" \
  -F "tracks=@$TRACKS_PATH;type=application/octet-stream" \
  -F "visibility=@$VISIBILITY_PATH;type=application/octet-stream" \
  -F "sample_steps=$SAMPLE_STEPS" \
  -F "frame_num=$FRAME_NUM" \
  -o "$CREATE_RESPONSE"

JOB_ID="$(python3 - "$CREATE_RESPONSE" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    response = json.load(f)
job_id = response.get("job_id")
if not job_id:
    raise SystemExit(f"Missing job_id in response: {response}")
print(job_id)
PY
)"

echo "Job id: $JOB_ID"

deadline=$((SECONDS + TIMEOUT_SECONDS))
last_status=""
while true; do
  curl --fail --silent --show-error "$API_URL/v1/jobs/$JOB_ID" -o "$JOB_RESPONSE"
  status="$(python3 - "$JOB_RESPONSE" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    job = json.load(f)
print(job.get("status", "unknown"))
PY
)"

  if [[ "$status" != "$last_status" ]]; then
    echo "Job status: $status"
    last_status="$status"
  fi

  if [[ "$status" == "succeeded" ]]; then
    break
  fi

  if [[ "$status" == "failed" ]]; then
    python3 - "$JOB_RESPONSE" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    job = json.load(f)
print(json.dumps(job, indent=2, sort_keys=True))
PY
    exit 1
  fi

  if (( SECONDS >= deadline )); then
    echo "Timed out after ${TIMEOUT_SECONDS}s waiting for job $JOB_ID" >&2
    exit 1
  fi

  sleep "$POLL_SECONDS"
done

VIDEO_PATH="$OUTPUT_DIR/$JOB_ID.mp4"
curl --fail --silent --show-error --location "$API_URL/v1/jobs/$JOB_ID/video" -o "$VIDEO_PATH"

if [[ ! -s "$VIDEO_PATH" ]]; then
  echo "Downloaded video is missing or empty: $VIDEO_PATH" >&2
  exit 1
fi

python3 - "$JOB_RESPONSE" "$VIDEO_PATH" <<'PY'
import json
import os
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    job = json.load(f)
video_path = sys.argv[2]
print("Job summary:")
print("  seed:", job.get("seed"))
print("  duration_seconds:", job.get("duration_seconds"))
print("  output:", video_path)
print("  bytes:", os.path.getsize(video_path))
PY

if command -v ffprobe >/dev/null 2>&1; then
  ffprobe -v error \
    -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,duration \
    -of default=noprint_wrappers=1 \
    "$VIDEO_PATH" || true
fi

echo "Open the video to visually inspect whether the red block follows the diagonal trajectory."
