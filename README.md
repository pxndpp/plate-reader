# Thai License Plate Reader

Upload a photo of a car and get back where its license plates are and what they say.

1. **Detect**: a YOLO11n model finds every plate in the photo.
2. **Crop**: each plate is cut out of the photo.
3. **Read**: EasyOCR reads the Thai letters and digits.
4. **Clean**: the top line becomes the registration (e.g. `กข 1234`), the next line the province.

A Gradio web page shows the photo with boxes drawn on it, each cropped plate, and a table of readings with confidence scores.

Class project. It works best on clear photos of ordinary cars; see [Limitations](#limitations).

## Setup

Needs **Python 3.12** (PyTorch and EasyOCR do not support 3.14 yet). An NVIDIA GPU is optional; everything runs on CPU, only slower.

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

The `--extra-index-url` is needed because `requirements.txt` pins the CUDA build of PyTorch. The trained detector is included at `weights/best.pt`, so no training is needed to run the app. On first run EasyOCR downloads its Thai and English models, so the first reading takes longer.

## Run the web page

```bash
.venv/Scripts/python app.py
```

Then open http://127.0.0.1:7860, upload a car photo, and press **Read this plate**. Boxes are green when the detector is at least 50% sure of a plate and red below that.

Run commands from the project folder: the code loads the model from the relative path `weights/best.pt`.

## Use from Python

```python
import cv2
from pipeline import read_plates

for plate in read_plates(cv2.imread("car.jpg")):
    print(plate["registration"], plate["province"], plate["detection_confidence"])
```

Each plate is a dict with `bbox`, `detection_confidence`, `registration`, `province`, `reading_confidence` and `crop` (the cropped image). No plate found returns an empty list.

From the command line:

```bash
.venv/Scripts/python pipeline.py car.jpg
```

## Retrain the detector (optional)

1. Get a free API key from [Roboflow](https://roboflow.com), copy `.env.example` to `.env`, and put the key in it. Never commit `.env`.
2. Download the dataset and train (about 40 minutes on an RTX 3050 laptop GPU):

```bash
.venv/Scripts/python get_data.py --workspace license-plate-detection-el2to --project test-thai-license-plate-detect --version 1 --out data/plates_el2to
```

```bash
.venv/Scripts/python train.py --data data/plates_el2to/data.yaml --epochs 30 --name plate_el2to
```

`train.py` copies the best result to `weights/best.pt`. On a 4 GB GPU, lower `batch` in `train.py` if you run out of memory.

## Results

**Plate detection** (mAP on each dataset's held-out test split):

| Test set | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Car photos (135, training dataset's test split) | 0.96 | 0.90 | 0.97 | 0.67 |
| Front-view cars (28, separate dataset) | 0.98 | 1.00 | 0.99 | 0.73 |

**Reading** (24 real car photos, checked by eye, not yet against typed answers): the registration was fully correct on 14 of 24 photos (58%), with 3 more unclear.

## Limitations

- **Small or blurry plates** (under about 150 px wide in the photo) are often misread or not read at all. Use high-resolution photos.
- **Similar Thai letters** get confused, e.g. `ธ`/`ร`, `ฒ`/`ผ`, `ฎ`/`ฏ`.
- **Motorcycle plates** (square, three lines) and **truck plates** with a `THAILAND` header are not supported.
- **Province** is raw OCR text and is often slightly misspelled; it is not matched against the 77 provinces yet.
- **Close-up photos of just the plate** are often missed: the detector was trained on whole-car photos.

## Tests

```bash
.venv/Scripts/python tests/test_pipeline.py
```

Checks the text logic (line grouping and cleaning) without loading any model.

## Project files

| File | What it does |
|---|---|
| `app.py` | Gradio web page |
| `pipeline.py` | Detect, crop, read, clean: `read_plates()` |
| `train.py` | Fine-tune YOLO11n on a plate dataset |
| `get_data.py` | Download a dataset from Roboflow |
| `ocr_smoke.py` | Quick check that EasyOCR reads a plate image |
| `weights/best.pt` | Trained plate detector |
| `GLOSSARY.md` | Terms used in this project |

## Credits

- Training data: [Test Thai License Plate Detect](https://universe.roboflow.com/license-plate-detection-el2to/test-thai-license-plate-detect/dataset/1) by license-plate-detection-el2to, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Datasets used for testing and comparison, also CC BY 4.0: [Thai License Plate](https://universe.roboflow.com/naruesorn/thai-license-plate-j6y9l) by naruesorn, [License Plate](https://universe.roboflow.com/training-101/license-plate-eql7j-nskv2) by training-101, [Thai License Plate Detector](https://universe.roboflow.com/th-support-cytron-yvkeg/thai-license-plate-detector) by TH Support Cytron.
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [EasyOCR](https://github.com/JaidedAI/EasyOCR), [Gradio](https://www.gradio.app).
