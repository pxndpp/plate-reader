"""Day-1 check: can EasyOCR read a plate image? Usage: python ocr_smoke.py path/to/plate_crop.jpg"""
import sys

import easyocr

reader = easyocr.Reader(["th", "en"], gpu=True)
for box, text, conf in reader.readtext(sys.argv[1]):
    print(f"{conf:.2f}  {text}  y={box[0][1]:.0f}")
