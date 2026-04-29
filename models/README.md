# Model Artifacts

Place exported CPU-friendly artifacts here:

- `yolo/`: INT8 ONNX or OpenVINO YOLO detector for cars, slots, fire, and smoke.
- `ocr/`: lightweight plate detector/recognizer artifacts.
- `reid/`: ResNet18-style vehicle embedding checkpoint.
- `vlm/`: quantized lightweight VLM artifacts.

The checked-in application uses mock inference so the project is runnable before model export.
