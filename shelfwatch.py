#!/usr/bin/env python3
"""ShelfWatch: a retail shelf stock monitor built on Roboflow hosted inference.

Snap a photo of a retail shelf. ShelfWatch sends it to a Roboflow Universe
object-detection model, counts what is on the shelf, and prints a restock
report. Built as a real-world demo of shipping computer vision with Roboflow.

Usage:
    ROBOFLOW_API_KEY=your_key python shelfwatch.py shelf.jpg
    ROBOFLOW_API_KEY=your_key python shelfwatch.py --url https://.../shelf.jpg

Get a free API key at https://app.roboflow.com (account settings).
"""

import argparse
import json
import os
import sys

from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw

API_URL = "https://serverless.roboflow.com"
MODEL_ID = "retail-shelf-l96n8/1"  # Universe retail-shelf model (Yolonew, CC BY 4.0)
LOW_STOCK_THRESHOLD = 3  # a class with fewer detections than this gets flagged


def get_client(api_key):
    return InferenceHTTPClient(api_url=API_URL, api_key=api_key)


def analyze(predictions):
    """Turn a Roboflow prediction payload into shelf counts."""
    preds = predictions.get("predictions", []) or []
    by_class = {}
    for p in preds:
        cls = p.get("class", "product")
        by_class[cls] = by_class.get(cls, 0) + 1
    low = [c for c, n in by_class.items() if n < LOW_STOCK_THRESHOLD]
    return {
        "total_detections": len(preds),
        "by_class": by_class,
        "low_stock_classes": sorted(low),
    }


def format_report(result, image_name):
    lines = [
        "ShelfWatch report: {}".format(image_name),
        "Products detected: {}".format(result["total_detections"]),
    ]
    if not result["by_class"]:
        lines.append(
            "No products detected. The shelf may be empty, "
            "or the photo angle may need another try."
        )
        return "\n".join(lines)
    for cls, n in sorted(result["by_class"].items(), key=lambda kv: -kv[1]):
        flag = "  <-- LOW, restock?" if cls in result["low_stock_classes"] else ""
        lines.append("  {}: {}{}".format(cls, n, flag))
    return "\n".join(lines)


def annotate_image(image_path, predictions, out_path):
    """Draw detection boxes on a copy of the image and save it."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for p in predictions.get("predictions", []) or []:
        x, y, w, h = p["x"], p["y"], p["width"], p["height"]
        draw.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline="red", width=3)
        draw.text((x - w / 2, y - h / 2 - 12), p.get("class", ""), fill="red")
    img.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="ShelfWatch: shelf stock monitor on Roboflow.")
    parser.add_argument("image", nargs="?", help="Local image path (or use --url).")
    parser.add_argument("--url", help="Image URL instead of a local file.")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"),
                        help="Roboflow API key (or set ROBOFLOW_API_KEY).")
    parser.add_argument("--json-out", help="Also write raw predictions to this JSON file.")
    args = parser.parse_args()

    if not args.api_key:
        print("Missing API key. Set ROBOFLOW_API_KEY or pass --api-key.", file=sys.stderr)
        print("Free key: https://app.roboflow.com -> account settings.", file=sys.stderr)
        sys.exit(2)
    if not args.image and not args.url:
        parser.error("Give an image path or --url.")

    source = args.url or args.image
    print("Running inference on {} ...".format(source))
    client = get_client(args.api_key)
    predictions = client.infer(source, model_id=MODEL_ID)

    result = analyze(predictions)
    print(format_report(result, source))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(predictions, f, indent=2)
        print("Raw predictions: {}".format(args.json_out))

    if args.image and os.path.isfile(args.image):
        out = os.path.splitext(args.image)[0] + "_annotated.jpg"
        annotate_image(args.image, predictions, out)
        print("Annotated image: {}".format(out))


if __name__ == "__main__":
    main()
