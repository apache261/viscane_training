import math

import pytest
import torch
from PIL import Image
from torchvision import transforms

from sugarcane_variety.verification import (
    SugarcaneFeatureVerifier, build_feature_verifier, feature_distances,
    select_distance_threshold, verification_result,
)


def test_feature_distance_checks_features_instead_of_class_confidence(policy):
    references = torch.tensor([[1., 0.], [0., 1.]])
    scores = feature_distances(torch.tensor([[5., 0.], [-1., 0.]]), references)
    assert scores.tolist() == pytest.approx([0., 1.])
    assert verification_result(float(scores[0]), policy)["status"] == "accepted"
    assert verification_result(float(scores[1]), policy)["status"] == "uncertain"
    assert verification_result(0.2, policy)["status"] == "accepted"


@pytest.mark.parametrize("values", [[[math.nan, 1.]], [[0., 0.]], [[math.inf, 1.]]])
def test_invalid_features_cannot_pass(values):
    with pytest.raises(ValueError):
        feature_distances(torch.tensor(values), torch.tensor([[1., 0.]]))


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1, 2.1])
def test_invalid_distances_cannot_pass(value, policy):
    with pytest.raises(ValueError):
        verification_result(value, policy)


def test_calibration_uses_requested_validation_quantile():
    distances = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.9])
    assert select_distance_threshold(distances, 0.7) == pytest.approx(0.04)
    with pytest.raises(ValueError):
        select_distance_threshold(torch.zeros(10), 0.95)
    with pytest.raises(ValueError):
        select_distance_threshold(distances, 1.0)


def test_build_load_and_inference_with_real_encoder(tmp_path, sugarcane_dataset):
    prepared, checkpoint = sugarcane_dataset
    metadata = build_feature_verifier(str(prepared), str(checkpoint), str(tmp_path / "gate"), workers=0)
    assert metadata["verification"]["calibration_split"] == "val"
    assert metadata["verification"]["test_samples"] == 4
    assert metadata["verification"]["non_sugarcane_false_acceptance_rate"] is None
    gate = SugarcaneFeatureVerifier(metadata["checkpoint_path"], torch.device("cpu"))
    with Image.open(prepared / "train" / "524__MATURE" / "0.png") as image:
        result = gate.verify(transforms.PILToTensor()(image.convert("RGB")))
    assert result["status"] == "accepted"
    assert result["feature_distance"] == pytest.approx(0, abs=1e-5)
    # Changing only test images must not influence the calibrated threshold.
    for path in (prepared / "test").rglob("*.png"):
        with Image.open(path) as image:
            image.transpose(Image.Transpose.ROTATE_90).save(path)
    second = build_feature_verifier(str(prepared), str(checkpoint), str(tmp_path / "gate2"), workers=0)
    assert second["verification"]["distance_threshold"] == metadata["verification"]["distance_threshold"]
    # The serialized policy must be required, not reconstructed from defaults.
    bundle = torch.load(metadata["checkpoint_path"], weights_only=True)
    del bundle["verification"]["distance_threshold"]
    torch.save(bundle, tmp_path / "broken.pt")
    with pytest.raises(ValueError):
        SugarcaneFeatureVerifier(tmp_path / "broken.pt", torch.device("cpu"))


def test_builder_blocks_duplicate_leakage(tmp_path, sugarcane_dataset):
    prepared, checkpoint = sugarcane_dataset
    (prepared / "val" / "524__MATURE" / "0.png").write_bytes(
        (prepared / "train" / "524__MATURE" / "0.png").read_bytes())
    with pytest.raises(ValueError, match="duplicate"):
        build_feature_verifier(str(prepared), str(checkpoint), str(tmp_path / "gate"), workers=0)
