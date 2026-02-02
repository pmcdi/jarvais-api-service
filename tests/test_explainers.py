"""
Live endpoint sanity checks for explainer routes, similar to test_box_plot style.
Requires the FastAPI server running (default http://localhost:8888).
Run with: pixi run python tests/routers/test_explainers.py
"""

import json
import os
import sys
import tempfile

import pandas as pd
import requests


SERVER_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8888")


def _fail(msg: str) -> None:
    print(f"❌ {msg}")
    sys.exit(1)


# def create_tiny_dataset() -> str:
#     df = pd.DataFrame(
#         {
#             "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
#             "feature2": [0.1, 0.2, 0.1, 0.3, 0.25, 0.4, 0.35, 0.5],
#             "target": [0, 1, 0, 1, 0, 1, 0, 1],
#         }
#     )
#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
#     df.to_csv(tmp.name, index=False)
#     return tmp.name

RADCURE_DATA_PATH = "/Users/sejinkim/projects/jarvais/data/RADCURE_Clinical_v04_20241219_minusone.csv"

def upload_data(csv_path: str) -> str:
    with open(csv_path, "rb") as f:
        resp = requests.post(f"{SERVER_URL}/upload", files={"file": f}, timeout=300)
    if resp.status_code not in (200, 201):
        _fail(f"Upload failed: {resp.status_code} {resp.text}")
    analyzer_id = resp.json().get("analyzer_id")
    if not analyzer_id:
        _fail("Upload response missing analyzer_id")
    print(f"✅ Uploaded data, analyzer_id={analyzer_id}")
    return analyzer_id


def train_model(analyzer_id: str) -> str:
    payload = {
        "analyzer_id": analyzer_id,
        "target_variable": "Status",
        "task": "binary",
        "k_folds": 2,
    }
    resp = requests.post(f"{SERVER_URL}/trainers", json=payload, timeout=300)
    if resp.status_code != 200:
        _fail(f"Training failed: {resp.status_code} {resp.text}")
    trainer_id = resp.json().get("trainer_id")
    if not trainer_id:
        _fail("Training response missing trainer_id")
    print(f"✅ Trained model, trainer_id={trainer_id}")
    return trainer_id


def call_explainer(trainer_id: str, path: str, params=None):
    url = f"{SERVER_URL}/explainers/{trainer_id}/{path}"
    resp = requests.get(url, params=params or {}, timeout=300)
    if resp.status_code != 200:
        _fail(f"{path} failed: {resp.status_code} {resp.text}")
    print(f"✅ {path}")
    return resp.json()


def cleanup(analyzer_id: str, trainer_id: str):
    try:
        requests.delete(f"{SERVER_URL}/trainers/{trainer_id}", timeout=30)
    except Exception:
        pass
    try:
        requests.delete(f"{SERVER_URL}/analyzers/{analyzer_id}", timeout=30)
    except Exception:
        pass


def main():
    # csv_path = create_tiny_dataset()
    analyzer_id = None
    trainer_id = None

    try:
        analyzer_id = upload_data(RADCURE_DATA_PATH)
        trainer_id = train_model(analyzer_id)

        resp = call_explainer(trainer_id, "shap/summary")
        # save the response to a file
        with open("archive/shap_summary.json", "w") as f:
            json.dump(resp, f)

        resp = call_explainer(trainer_id, "shap/bar")
        with open("archive/shap_bar.json", "w") as f:
            json.dump(resp, f)

        # Use a real numeric feature from the RADCURE dataset
        resp = call_explainer(trainer_id, "pdp", params={"feature": "Age"})
        with open("archive/pdp.json", "w") as f:
            json.dump(resp, f)

        resp = call_explainer(trainer_id, "permutation_importance")
        with open("archive/permutation_importance.json", "w") as f:
            json.dump(resp, f)

        resp = call_explainer(trainer_id, "distribution", params={"feature": "Age"})
        with open("archive/distribution.json", "w") as f:
            json.dump(resp, f)

        print("🎉 Explainer endpoint checks completed")
    finally:
        cleanup(analyzer_id, trainer_id)


if __name__ == "__main__":
    main()

