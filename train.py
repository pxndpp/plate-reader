"""Fine-tune YOLO11n on the plate dataset (1 class: plate). Run get_data.py first.

Example: python train.py --data data/plates_big/data.yaml --epochs 30 --name plate_big
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/plates/data.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--name", default="plate")
    a = p.parse_args()

    model = YOLO("yolo11n.pt")
    # batch 8 fits RTX 3050 4GB; lower it on out-of-memory.
    results = model.train(data=a.data, epochs=a.epochs, imgsz=640, batch=8, workers=2, project="runs", name=a.name)
    best = Path(results.save_dir) / "weights" / "best.pt"  # ultralytics may nest the output dir
    if best.exists():
        Path("weights").mkdir(exist_ok=True)
        shutil.copy(best, "weights/best.pt")
        print("copied ->", "weights/best.pt")


if __name__ == "__main__":  # required on Windows (dataloader workers)
    main()
