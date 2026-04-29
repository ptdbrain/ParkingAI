from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse VLM quantization arguments."""

    parser = argparse.ArgumentParser(description="Prepare lightweight VLM quantization for edge deployment.")
    parser.add_argument("--model-id", type=str, default="vikhyatk/moondream2", help="Hugging Face model id or local path")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4, help="Target quantization bits")
    parser.add_argument("--output-dir", type=Path, default=Path("models/vlm"), help="Directory for quantized artifacts")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only export planning")
    return parser.parse_args()


def main() -> None:
    """Emit a safe quantization plan without downloading heavy models."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Prepared VLM quantization plan")
    logger.info("model_id=%s bits=%s cpu_only=%s output_dir=%s", args.model_id, args.bits, args.cpu_only, args.output_dir)
    logger.info("TODO: Add optimum/onnxruntime or bitsandbytes flow after selecting the exact VLM checkpoint.")


if __name__ == "__main__":
    main()
