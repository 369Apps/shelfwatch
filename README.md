# ShelfWatch

A retail shelf stock monitor built on **Roboflow hosted inference**. Snap a
photo of a retail shelf, and ShelfWatch detects the products on it, counts
them by class, and flags anything running low so you know what to restock.

Built as a real-world demo of shipping computer vision with Roboflow: no
training, no GPU, no server to manage. A Universe model does the detection,
Roboflow's serverless API does the hosting, and this script does the business
logic.

## The real-world angle

I run a barbershop with a small retail shelf (pomades, beard oils, shampoos).
Nobody checks it until a customer asks for something that is not there. This
turns a phone photo into a restock report in seconds.

## Setup

1. Create a free Roboflow account at https://app.roboflow.com
2. Copy your API key from account settings
3. Install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install inference-sdk pillow
```

## Run

```bash
export ROBOFLOW_API_KEY=your_key_here
.venv/bin/python shelfwatch.py path/to/shelf.jpg
.venv/bin/python shelfwatch.py --url https://example.com/shelf.jpg --json-out preds.json
```

Output: a per-product count with low-stock flags, plus an annotated copy of
the image with detection boxes drawn on it.

## How it works

- `inference_sdk.InferenceHTTPClient` talks to `https://serverless.roboflow.com`
- Model: `retail-shelf-l96n8/1`, a Universe community model for retail shelf
  detection (Yolonew, CC BY 4.0). Swap `MODEL_ID` in `shelfwatch.py` for any
  other Universe model or your own trained version.
- `analyze()` groups predictions by class; any class under
  `LOW_STOCK_THRESHOLD` (default 3) is flagged for restock.

## Tests

```bash
.venv/bin/python test_report.py   # offline, no API key needed
```

## Next steps

- Train a custom model on the shop's actual products for exact SKUs
- Wrap it in a Roboflow Workflow with a daily scheduled check
- Push low-stock alerts to the shop's WhatsApp instead of a terminal report
