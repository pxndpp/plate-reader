"""Car Photo -> Readings. detect (YOLO) -> crop -> read (EasyOCR) -> clean.

The UI (app.py) and evaluate.py both call read_plates(); keep all pipeline logic here.
CLI check: python pipeline.py photo.jpg
"""
import re
import sys
from functools import lru_cache

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

WEIGHTS = "weights/best.pt"
PAD = 0.05          # extra margin around the detected box, fraction of box size
OCR_WIDTH = 400     # upscale crops narrower than this before OCR
MIN_DET_CONF = 0.25


@lru_cache(maxsize=1)
def _detector():
    return YOLO(WEIGHTS)


@lru_cache(maxsize=1)
def _reader():
    return easyocr.Reader(["th", "en"], gpu=True)  # falls back to CPU if no GPU


def group_lines(results):
    """EasyOCR results [(box, text, conf)] -> list of lines [(text, mean_conf)], top to bottom.

    Tokens whose vertical centres are close share a line (EasyOCR often splits "สก" and "2484").
    """
    if not results:
        return []
    items = []
    for box, text, conf in results:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({"cy": sum(ys) / 4, "h": max(ys) - min(ys), "x": min(xs), "text": text, "conf": conf})
    items.sort(key=lambda i: i["cy"])
    lines, cur = [], [items[0]]
    for it in items[1:]:
        # min(): tilted plates inflate token heights, max() would merge the province line into the registration
        if it["cy"] - cur[-1]["cy"] <= 0.5 * min(it["h"], cur[-1]["h"]):
            cur.append(it)
        else:
            lines.append(cur)
            cur = [it]
    lines.append(cur)
    out = []
    for ln in lines:
        ln.sort(key=lambda i: i["x"])
        out.append((" ".join(i["text"] for i in ln), sum(i["conf"] for i in ln) / len(ln)))
    return out


def clean_registration(text):
    """Keep digits, Thai consonants and single spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^0-9ก-ฮ ]", "", text)).strip()


def clean_province(text):
    # ponytail: no snapping to the 77 provinces yet (day 6, if time) - raw Thai text only.
    return re.sub(r"\s+", " ", re.sub(r"[^ก-๛ ]", "", text)).strip()


def read_crop(crop_bgr):
    """One Plate Crop -> (registration, province, reading_confidence)."""
    w = crop_bgr.shape[1]
    s = max(1, OCR_WIDTH // max(1, w))
    gray = cv2.cvtColor(cv2.resize(crop_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    lines = group_lines(_reader().readtext(gray))
    if not lines:
        return "", "", 0.0
    # top line = Registration; Province = next line (dealer stickers may follow it)
    reg = clean_registration(lines[0][0])
    prov = clean_province(lines[1][0]) if len(lines) > 1 else ""
    return reg, prov, float(lines[0][1])


def read_plates(image_bgr, imgsz=640):
    """Car Photo (BGR array) -> list of Readings, one per detected Plate."""
    H, W = image_bgr.shape[:2]
    plates = []
    for (x1, y1, x2, y2), dconf in zip(*_boxes(image_bgr, imgsz)):
        px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
        a, b = max(0, int(x1 - px)), max(0, int(y1 - py))
        c, d = min(W, int(x2 + px)), min(H, int(y2 + py))
        crop = image_bgr[b:d, a:c]
        if crop.size == 0:
            continue
        reg, prov, rconf = read_crop(crop)
        plates.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "detection_confidence": float(dconf),
            "registration": reg,
            "province": prov,
            "reading_confidence": rconf,
            "crop": crop,
        })
    return plates


def _boxes(image_bgr, imgsz):
    r = _detector()(image_bgr, imgsz=imgsz, conf=MIN_DET_CONF, verbose=False)[0]
    return r.boxes.xyxy.tolist(), r.boxes.conf.tolist()


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    if img is None:
        sys.exit(f"cannot read {sys.argv[1]}")
    found = read_plates(img)
    if not found:
        print("ไม่พบป้ายทะเบียน")
    for p in found:
        print(f"det={p['detection_confidence']:.2f} read={p['reading_confidence']:.2f} "
              f"reg='{p['registration']}' province='{p['province']}' bbox={p['bbox']}")
