import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from sugarcane_variety import api
from sugarcane_variety.verification import build_feature_verifier, verification_result


@pytest.fixture
def client(monkeypatch):
    # No lifespan: each test explicitly controls the loaded model boundaries.
    monkeypatch.setattr(api, "loaded_models", {"resnet18": object(), "yolov8": None})
    monkeypatch.setattr(api, "model_classes", {"resnet18": ["524__MATURE", "524__NOT_MATURE"],
                                              "yolov8": ["524__MATURE", "524__NOT_MATURE"]})
    monkeypatch.setattr(api, "feature_verifier", None)
    monkeypatch.setattr(api, "verifier_load_error", None)
    monkeypatch.setattr(api, "model_checkpoints", {})
    monkeypatch.setattr(api, "model_load_errors", {"resnet18": None, "yolov8": None})
    monkeypatch.setattr(api, "loaded_model_type", "resnet18")
    for name in ("SUGARCANE_VERIFIER_CHECKPOINT", "MODEL_CHECKPOINT_RESNET", "MODEL_CHECKPOINT_YOLO"):
        monkeypatch.delenv(name, raising=False)
    return TestClient(api.app)


def image_upload():
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "green").save(output, format="PNG")
    return {"file": ("input.png", output.getvalue(), "image/png")}


class StubVerifier:
    def __init__(self, result):
        self.result = result

    def verify(self, image):
        assert image.shape == (3, 32, 32)
        return self.result


@pytest.mark.parametrize("endpoint", ["/predict", "/predict/annotated"])
@pytest.mark.parametrize("model_type", ["resnet18", "yolov8"])
def test_unrecognized_images_skip_both_classifiers_and_overlays(client, monkeypatch, policy, endpoint, model_type):
    monkeypatch.setattr(api, "loaded_models", {name: object() if name == model_type else None
                                              for name in api.SUPPORTED_MODELS})
    monkeypatch.setattr(api, "feature_verifier", StubVerifier(verification_result(0.8, policy)))
    def forbidden(*args, **kwargs):
        pytest.fail("Rejected images must not reach variety classification or annotation.")
    monkeypatch.setattr(api, "_predict_probabilities", forbidden)
    monkeypatch.setattr(api, "_draw_prediction_overlay", forbidden)
    response = client.post(endpoint, files=image_upload())
    assert response.status_code == 422
    assert response.json()["verification"]["reason"] == "unrecognized_image"
    assert response.json()["models"] == {}
    assert "X-Predicted-Maturity" not in response.headers


@pytest.mark.parametrize("endpoint", ["/predict", "/predict/annotated"])
def test_missing_verifier_prevents_unchecked_predictions(client, endpoint):
    response = client.post(endpoint, files=image_upload())
    assert response.status_code == 503
    assert "verifier is unavailable" in response.json()["detail"]


@pytest.mark.parametrize("endpoint", ["/predict", "/predict/annotated"])
def test_verifier_failure_prevents_classification(client, monkeypatch, endpoint):
    class BrokenVerifier:
        def verify(self, image):
            raise ValueError("invalid features")
    monkeypatch.setattr(api, "feature_verifier", BrokenVerifier())
    response = client.post(endpoint, files=image_upload())
    assert response.status_code == 503
    assert "no classification" in response.json()["detail"]


@pytest.mark.parametrize("endpoint", ["/predict", "/predict/annotated"])
def test_corrupt_file_is_not_reported_as_non_sugarcane(client, endpoint):
    response = client.post(endpoint, files={"file": ("bad.jpg", b"not an image", "image/jpeg")})
    assert response.status_code == 400


@pytest.mark.parametrize("model_type", ["resnet18", "yolov8"])
def test_accepted_images_retain_classification_and_annotation(client, monkeypatch, policy, model_type):
    monkeypatch.setattr(api, "loaded_models", {name: object() if name == model_type else None
                                              for name in api.SUPPORTED_MODELS})
    monkeypatch.setattr(api, "feature_verifier", StubVerifier(verification_result(0.1, policy)))
    calls = []
    def probabilities(image, model_type):
        calls.append(model_type)
        return torch.tensor([0.9, 0.1])
    monkeypatch.setattr(api, "_predict_probabilities", probabilities)
    response = client.post("/predict", files=image_upload())
    assert response.status_code == 200
    assert response.json()["verification"]["status"] == "accepted"
    assert response.json()["models"][model_type]["prediction"]["class_name"] == "524__MATURE"
    annotated = client.post("/predict/annotated", files=image_upload())
    assert annotated.status_code == 200
    assert annotated.headers["content-type"] == "image/png"
    assert annotated.headers["X-Verification-Status"] == "accepted"
    assert annotated.headers["X-Predicted-Maturity"] == "MATURE"
    assert calls == [model_type, model_type]
    with Image.open(io.BytesIO(annotated.content)) as image:
        assert image.size == (32, 32)


def test_invalid_top_k_and_no_models(client, monkeypatch):
    assert client.post("/predict?top_k=0", files=image_upload()).status_code == 422
    monkeypatch.setattr(api, "loaded_models", {"resnet18": None, "yolov8": None})
    assert client.post("/predict", files=image_upload()).status_code == 503


def test_startup_reports_missing_verifier(client, monkeypatch, tmp_path):
    def load_model(**kwargs):
        pass
    monkeypatch.setattr(api, "_load_model", load_model)
    monkeypatch.setenv("SUGARCANE_VERIFIER_CHECKPOINT", str(tmp_path / "missing.pt"))
    api.startup()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["prediction_ready"] is False
    assert response.json()["verification"]["loaded"] is False
    assert "FileNotFoundError" in response.json()["verification"]["error"]


def test_model_load_discovers_verifier_beside_checkpoint(client, tmp_path, sugarcane_dataset):
    prepared, checkpoint = sugarcane_dataset
    gate_path = tmp_path / "verifier" / "feature_verifier.pt"
    build_feature_verifier(str(prepared), str(checkpoint), str(gate_path.parent), workers=0)
    response = client.post("/models/load", json={"checkpoint_path": str(checkpoint), "model_type": "resnet18"})
    assert response.status_code == 200
    health = client.get("/health").json()
    assert health["prediction_ready"] is True
    assert health["verification"]["checkpoint"] == str(gate_path)
    filename = next((prepared / "train" / "524__MATURE").iterdir())
    result = client.post("/predict", files={"file": (filename.name, filename.read_bytes(), "image/png")})
    assert result.status_code == 200
    assert result.json()["verification"]["status"] == "accepted"
