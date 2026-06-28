# Wan-Move API Docker Deployment

## Download weights

PowerShell:

```powershell
.\scripts\download_weights.ps1
```

Bash:

```bash
bash scripts/download_weights.sh
```

Both scripts download `Ruihang/Wan-Move-14B-480P` into `./Wan-Move-14B-480P`, which is the default model mount used by Docker Compose.

## Build the image

PowerShell:

```powershell
.\scripts\build_image.ps1
```

Bash:

```bash
bash scripts/build_image.sh
```

## Run the API

PowerShell:

```powershell
.\scripts\run_api.ps1
```

Bash:

```bash
bash scripts/run_api.sh
```

The API listens on `http://localhost:8000`.

## Run the Gradio demo with Docker

Stop the API container first so Gradio can use the GPUs:

```bash
docker compose down
```

Then run:

```bash
PORT=7860 bash scripts/run_gradio.sh
```

Open:

```text
http://<GPU-host-ip>:7860
```

To rebuild before launching:

```bash
BUILD=1 PORT=7860 bash scripts/run_gradio.sh
```

## Submit an example job

```bash
curl -X POST http://localhost:8000/v1/generations \
  -F "prompt=A laptop is placed on a wooden table. The silver laptop is connected to a small grey external hard drive and transfers data through a white USB-C cable. The video is shot with a downward close-up lens." \
  -F "image=@examples/example.jpg" \
  -F "tracks=@examples/example_tracks.npy" \
  -F "visibility=@examples/example_visibility.npy"
```

Poll the returned job id:

```bash
curl http://localhost:8000/v1/jobs/<job_id>
```

Download the completed video:

```bash
curl -L http://localhost:8000/v1/jobs/<job_id>/video -o output.mp4
```

## Run a self-contained trajectory test

This creates its own synthetic image and trajectory files, submits a generation job, polls until completion, and downloads the MP4:

```bash
API_URL=http://localhost:8181 bash scripts/test_api_generation.sh
```

For a quicker smoke test, reduce sampling steps:

```bash
API_URL=http://localhost:8181 SAMPLE_STEPS=4 bash scripts/test_api_generation.sh
```

Outputs are written under `outputs/api-test/`.

## Useful overrides

PowerShell:

```powershell
.\scripts\run_api.ps1 -WeightsDir D:\models\Wan-Move-14B-480P -OutputsDir D:\wan-outputs -Port 8080
```

Bash:

```bash
WEIGHTS_DIR=/data/models/Wan-Move-14B-480P OUTPUTS_DIR=/data/wan-outputs PORT=8080 bash scripts/run_api.sh
```
