"""Download the plate dataset from Roboflow into data/.

Needs a free Roboflow API key in the env var ROBOFLOW_API_KEY (or a .env file, see .env.example).
Default dataset: th-support-cytron-yvkeg/thai-license-plate-detector (1 class, CC BY 4.0).
Check the version number on the dataset's Universe page and pass it with --version.
"""
import argparse
import os

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

p = argparse.ArgumentParser()
p.add_argument("--workspace", default="th-support-cytron-yvkeg")
p.add_argument("--project", default="thai-license-plate-detector")
p.add_argument("--version", type=int, default=1)
p.add_argument("--out", default="data/plates")
a = p.parse_args()

key = os.environ.get("ROBOFLOW_API_KEY")
if not key:
    raise SystemExit("Set ROBOFLOW_API_KEY (env var or .env file).")

ds = Roboflow(api_key=key).workspace(a.workspace).project(a.project).version(a.version)
ds = ds.download("yolov8", location=a.out)
print("data.yaml:", ds.location + "/data.yaml")
