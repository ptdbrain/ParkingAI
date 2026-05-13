# Edge AI Vision-Language IoT Parking System

Production-like scaffold for an edge-first parking system. The code runs with mock inference on a CPU-only Ubuntu laptop and is structured so real YOLO, OCR, Re-ID, VLM, MySQL, and FAISS components can be swapped in without coupling them to FastAPI.

## Architecture

Data flow:

`Camera/RTSP -> frame_queue -> AI worker queues -> YOLO/OCR/ReID/VLM workers -> event_queue -> FastAPI WebSocket + REST services -> MySQL + dashboard`

The FastAPI backend does not call model code directly. It owns API, database, and WebSocket concerns. The AI runtime owns frame routing and event production. Communication happens through bounded `asyncio.Queue` instances to protect low-resource hardware from memory growth.

## Project Structure

- `app/ai`: async pipeline runtime and worker implementations.
- `app/core`: frame and event schema objects shared across layers.
- `app/backend`: FastAPI API routes and dependency injection.
- `app/db`: SQLAlchemy async models, schemas, sessions, repositories.
- `app/services`: business logic layer.
- `app/vector`: FAISS vector search wrapper with a pure-Python fallback.
- `models`: exported model artifact folders.
- `dataset`: curated dataset and calibration-frame area.
- `training`: export and quantization scripts.
- `frontend`: small WebSocket dashboard prototype.
- `tests`: behavior tests for schema and pipeline.
- `scripts`: runnable demo and API entrypoints.

## Event Format

Every AI worker emits this schema:

```json
{
  "frame_id": 1,
  "timestamp": "2026-04-28T10:00:00+00:00",
  "type": "car",
  "bbox": [40, 80, 220, 260],
  "confidence": 0.91,
  "image_crop": "frame-1:crop-40-80-220-260",
  "plate_text": "51A12345",
  "embedding": [0.01, 0.02]
}
```

`type` is strictly one of `car`, `slot`, `fire`, or `anomaly`.

## Run Locally

Create a lightweight environment:

```bash
cd edge_parking_system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or update the conda environment used for this project:

```bash
conda env update -n eps -f environment.yml
conda activate eps
```

For development checks, install the dev tools as well:

```bash
pip install -r requirements-dev.txt
```

Real inference dependencies are intentionally separate because they are much heavier:

```bash
pip install -r requirements-inference.txt
```

Run the mock AI pipeline:

```bash
python scripts/run_pipeline_demo.py
```

Start the API and live event feed:

```bash
python scripts/run_api.py
```

Open `frontend/index.html` in a browser. It connects to `ws://localhost:8000/ws/live`.

## Inference Modes

The default mode is mock inference:

```bash
PARKING_INFERENCE_MODE=mock
```

Mock mode keeps tests, demos, and the API runnable without downloaded OCR/model artifacts. Set `PARKING_INFERENCE_MODE=real` only after installing the optional inference dependencies and placing model artifacts under `models/`.

## Database

The default database URL uses SQLite so the demo starts without MySQL:

```bash
PARKING_DATABASE_URL=sqlite+aiosqlite:///./edge_parking_demo.db
```

For production MySQL:

```bash
PARKING_DATABASE_URL=mysql+asyncmy://parking_user:parking_pass@127.0.0.1:3306/parking
```

Apply migrations for managed databases:

```bash
alembic upgrade head
```

Seed demo data:

```bash
python database/seed_demo.py
```

## CPU Optimization Notes

- Keep one process and one worker per model family on old laptops.
- Use queue sizes from `PARKING_QUEUE_MAX_SIZE` to bound memory.
- Run VLM only on sampled frames or detector-triggered crops.
- Prefer YOLO nano/small models exported to INT8 ONNX/OpenVINO.
- Use FAISS CPU for Re-ID first; Milvus is unnecessary unless you outgrow local search.

## Extension Points

- Replace `YOLOWorker.detect` with ONNX Runtime or OpenVINO inference.
- Run `OCRWorker` in `real` mode with EasyOCR, PaddleOCR, or a compact ONNX OCR model.
- Generate OCR training crops from the prepared manifest with `python scripts/build_ocr_crops.py`.
- Replace `ReIDWorker.extract_embedding` with a real ResNet18/ViT embedding model.
- Replace `VLMWorker.infer` with a quantized Moondream2 or Florence-2 adapter.
- Use Alembic migrations from `database/migrations` before deployment.

## Artifact Policy

Large datasets, training runs, and checkpoints are intentionally ignored by git. Keep generated artifacts in one of these locations and document the exact source:

- `dataset/processed/` for prepared local data.
- `runs/` for local training output.
- `models/` for exported runtime artifacts.
- External storage such as Git LFS, Drive, S3, or a model registry for anything teammates need to reproduce.

## Tests

```bash
pytest
```

The tests validate the strict event schema and the mock async pipeline.
