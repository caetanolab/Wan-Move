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

## Useful overrides

PowerShell:

```powershell
.\scripts\run_api.ps1 -WeightsDir D:\models\Wan-Move-14B-480P -OutputsDir D:\wan-outputs -Port 8080
```

Bash:

```bash
WEIGHTS_DIR=/data/models/Wan-Move-14B-480P OUTPUTS_DIR=/data/wan-outputs PORT=8080 bash scripts/run_api.sh
```
