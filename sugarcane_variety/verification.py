"""Reject unfamiliar images using features from known sugarcane examples.

This is a one-class similarity check, not a trained sugarcane/non-sugarcane
classifier. Negative-image accuracy cannot be measured with positive data only.
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sugarcane_variety.train import (
    IMAGENET_MEAN, IMAGENET_STD, _build_resnet18, _collate_raw_images,
    _prepare_images_on_device, _training_device,
)

VERIFIER_VERSION = 1
PREPROCESSING = "rgb_uint8_direct_resize_imagenet_v1"
DEFAULT_VERIFIER_CHECKPOINT = "content/data/sugarcane_artifacts/resnet18/verifier/feature_verifier.pt"


def feature_distances(features: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    """Nearest cosine distance for unit-normalized penultimate ResNet features."""
    if features.ndim != 2 or references.ndim != 2 or features.shape[1] != references.shape[1]:
        raise ValueError("Feature dimensions do not match the reference bank.")
    if not len(features) or not len(references):
        raise ValueError("Features and references must not be empty.")
    for values in (features, references):
        if not torch.isfinite(values).all() or (values.norm(dim=1) < 1e-8).any():
            raise ValueError("Non-finite or empty feature vectors cannot be verified.")
    features = nn.functional.normalize(features, dim=1)
    references = nn.functional.normalize(references, dim=1)
    return (1.0 - (features @ references.T).max(dim=1).values).clamp(0.0, 2.0)


def select_distance_threshold(distances: torch.Tensor, acceptance_quantile: float) -> float:
    if not math.isfinite(acceptance_quantile) or not 0 < acceptance_quantile < 1:
        raise ValueError("acceptance_quantile must be strictly between 0 and 1.")
    if distances.ndim != 1 or not len(distances) or not torch.isfinite(distances).all():
        raise ValueError("Calibration needs finite validation distances.")
    if (distances < 0).any() or (distances > 2).any():
        raise ValueError("Cosine distances must be in [0, 2].")
    threshold = float(torch.quantile(distances.float(), acceptance_quantile, interpolation="higher"))
    if threshold <= 0 or threshold >= 2:
        raise ValueError("Degenerate validation threshold; check dataset diversity and duplicate leakage.")
    return threshold


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("version") != VERIFIER_VERSION or policy.get("method") != "resnet18_nearest_cosine":
        raise ValueError("Unsupported feature verifier version/method.")
    if policy.get("preprocessing") != PREPROCESSING:
        raise ValueError("Unsupported feature verifier preprocessing.")
    threshold = policy.get("distance_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold < 2:
        raise ValueError("Invalid feature verifier distance threshold.")


def verification_result(distance: float, policy: dict[str, Any]) -> dict[str, Any]:
    validate_policy(policy)
    if not math.isfinite(distance) or not 0 <= distance <= 2:
        raise ValueError("Invalid feature distance.")
    accepted = distance <= policy["distance_threshold"]
    return {
        "status": "accepted" if accepted else "uncertain",
        "reason": "matches_known_sugarcane_features" if accepted else "unrecognized_image",
        "feature_distance": distance,
        "distance_threshold": policy["distance_threshold"],
        "method": policy["method"],
        "verifier_version": policy["version"],
        "message": (
            "Image resembles the known sugarcane examples."
            if accepted else "Image is not recognized as a supported sugarcane image. Try a clear sugarcane photo."
        ),
    }


class SugarcaneFeatureVerifier:
    def __init__(self, checkpoint_path: str | Path, device: torch.device):
        path = Path(checkpoint_path).expanduser().resolve()
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        self.policy = bundle["verification"]
        validate_policy(self.policy)
        self.image_size = bundle["image_size"]
        if type(self.image_size) is not int or self.image_size < 1:
            raise ValueError("Verifier image_size must be a positive integer.")
        self.device = device
        self.checkpoint_path = str(path)
        self.model = _build_resnet18(num_classes=1, pretrained=False)
        self.model.fc = nn.Identity()
        self.model.load_state_dict(bundle["encoder_state_dict"])
        self.model.to(device).eval()
        self.references = bundle["reference_features"].to(device=device, dtype=torch.float32)
        if self.references.ndim != 2 or self.references.shape[1] != 512:
            raise ValueError("Verifier needs a reference bank of 512-dimensional ResNet18 features.")
        feature_distances(self.references[:1], self.references)

    @torch.no_grad()
    def verify(self, image: torch.Tensor) -> dict[str, Any]:
        batch = _prepare_images_on_device([image], self.device, self.image_size, training=False)
        distance = feature_distances(self.model(batch), self.references)[0]
        return verification_result(float(distance.cpu()), self.policy)


@torch.no_grad()
def _extract_features(model: nn.Module, dataset: Any, device: torch.device,
                      image_size: int, batch_size: int, workers: int) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, collate_fn=_collate_raw_images)
    features = []
    for images, _ in loader:
        batch = _prepare_images_on_device(images, device, image_size, training=False)
        features.append(model(batch).cpu())
    if not features:
        raise ValueError("Cannot build a feature verifier from an empty split.")
    return torch.cat(features)


def _check_duplicate_leakage(split_datasets: dict[str, Any]) -> None:
    """Do not calibrate on exact training duplicates, including across class names."""
    seen: dict[str, str] = {}
    for split, dataset in split_datasets.items():
        for filename, _ in dataset.samples:
            digest = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                raise ValueError(f"Exact duplicate across {seen[digest]} and {split}; fix split leakage before building the verifier.")
            seen[digest] = split


def build_feature_verifier(prepared_dir: str, checkpoint_path: str,
                           output_dir: str = "content/data/sugarcane_artifacts/resnet18/verifier",
                           acceptance_quantile: float = 0.95, references_per_class: int = 256,
                           batch_size: int = 32, workers: int = 2, seed: int = 42,
                           overwrite: bool = False) -> dict[str, Any]:
    """Use train references, validation thresholds, and untouched test reporting.

    Reuses the trained ResNet18; no new labels or classifier retraining required.
    The bundle embeds the encoder weights to keep features and thresholds aligned.
    """
    if not 0 < acceptance_quantile < 1 or not math.isfinite(acceptance_quantile):
        raise ValueError("acceptance_quantile must be strictly between 0 and 1.")
    if references_per_class < 1 or batch_size < 1 or workers < 0:
        raise ValueError("Reference and batch sizes must be positive; workers must be non-negative.")
    root, out = Path(prepared_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    output_path = out / "feature_verifier.pt"
    if output_path.exists() and not overwrite:
        raise ValueError("Use a new verifier output directory to preserve the existing verifier.")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type", "resnet18") != "resnet18":
        raise ValueError("Feature verification requires a trained ResNet18 checkpoint.")
    classes = list(checkpoint["classes"])
    split_datasets = {split: datasets.ImageFolder(root / split, transform=transforms.PILToTensor())
                      for split in ("train", "val", "test")}
    for split, dataset in split_datasets.items():
        if dataset.classes != classes:
            raise ValueError(f"{split} class ordering does not match the trained checkpoint.")
    _check_duplicate_leakage(split_datasets)
    model = _build_resnet18(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.fc = nn.Identity()
    device, image_size = _training_device(), int(checkpoint["image_size"])
    model.to(device).eval()
    generator = torch.Generator().manual_seed(seed)
    selected = []
    targets = torch.tensor(split_datasets["train"].targets)
    for class_index in range(len(classes)):
        indices = (targets == class_index).nonzero().flatten()
        selected.extend(indices[torch.randperm(len(indices), generator=generator)[:references_per_class]].tolist())
    references = _extract_features(model, Subset(split_datasets["train"], selected), device, image_size, batch_size, workers)
    # Keep distance calculations on CPU to bound GPU memory for large banks.
    def distances_for(split: str) -> torch.Tensor:
        features = _extract_features(model, split_datasets[split], device, image_size, batch_size, workers)
        return torch.cat([feature_distances(chunk, references) for chunk in features.split(batch_size)])

    val_distances = distances_for("val")
    threshold = select_distance_threshold(val_distances, acceptance_quantile)
    test_distances = distances_for("test")
    policy = {
        "version": VERIFIER_VERSION, "method": "resnet18_nearest_cosine", "preprocessing": PREPROCESSING,
        "distance_threshold": threshold, "acceptance_quantile": acceptance_quantile,
        "calibration_split": "val", "reference_split": "train", "reference_count": len(references),
        "source_classes": classes, "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "validation_samples": len(val_distances),
        "validation_acceptance_rate": float((val_distances <= threshold).float().mean()),
        "test_samples": len(test_distances),
        "test_false_rejection_rate": float((test_distances > threshold).float().mean()),
        "non_sugarcane_false_acceptance_rate": None,
        "limitation": "Sugarcane-only calibration; rejection of non-sugarcane images has not been evaluated.",
    }
    bundle = {"verification": policy, "image_size": image_size,
              "encoder_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
              "reference_features": nn.functional.normalize(references, dim=1)}
    out.mkdir(parents=True, exist_ok=True)
    # A failed write must not replace the previous usable bundle on retraining.
    with tempfile.NamedTemporaryFile(dir=out, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(bundle, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    metadata = {"checkpoint_path": str(output_path), "image_size": image_size, "verification": policy,
                "input_format": "RGB uint8 CHW", "normalization_mean": IMAGENET_MEAN,
                "normalization_std": IMAGENET_STD, "required_before_variety_prediction": True,
                "mobile_export_available": False}
    metadata_path = out / "feature_verifier_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
