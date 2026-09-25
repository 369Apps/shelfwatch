"""Offline tests for ShelfWatch report logic (no API key needed)."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shelfwatch import analyze, format_report

FAKE = {
    "predictions": [
        {"x": 10, "y": 10, "width": 20, "height": 20, "class": "pomade", "confidence": 0.9},
        {"x": 40, "y": 10, "width": 20, "height": 20, "class": "pomade", "confidence": 0.8},
        {"x": 70, "y": 10, "width": 20, "height": 20, "class": "pomade", "confidence": 0.85},
        {"x": 100, "y": 10, "width": 20, "height": 20, "class": "pomade", "confidence": 0.7},
        {"x": 130, "y": 10, "width": 20, "height": 20, "class": "beard-oil", "confidence": 0.95},
    ]
}

result = analyze(FAKE)
assert result["total_detections"] == 5, result
assert result["by_class"] == {"pomade": 4, "beard-oil": 1}, result
assert result["low_stock_classes"] == ["beard-oil"], result

report = format_report(result, "shelf.jpg")
assert "Products detected: 5" in report, report
assert "beard-oil: 1  <-- LOW, restock?" in report, report
assert "pomade: 4" in report and "pomade: 4  <--" not in report, report

empty = analyze({"predictions": []})
assert empty["total_detections"] == 0
assert "No products detected" in format_report(empty, "empty.jpg")

print("All ShelfWatch offline tests passed.")
