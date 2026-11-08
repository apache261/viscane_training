from pathlib import Path
import sys

import pytest
import torch
from PIL import Image

from sugarcane_variety import cli, colab_compatible, train


@pytest.mark.parametrize("entrypoint", ["cli-train", "cli-all", "colab"])
def test_readme_workflows_build_verifier_from_supplied_paths(monkeypatch, tmp_path, sugarcane_dataset, entrypoint):
    prepared, template = sugarcane_dataset
    output = tmp_path / "my trained models"
    classes = ["524__MATURE", "847__MATURE"]
    raw = tmp_path / "my dataset"
    for label in classes:
        folder = raw / label
        folder.mkdir(parents=True)
        for index in range(10):
            pixels = torch.randint(0, 256, (40, 48, 3), dtype=torch.uint8).numpy()
            Image.fromarray(pixels).save(folder / f"{index}.png")

    # Substitute the expensive optimizer/pretrained download, while exercising
    # real path handling, preprocessing, feature extraction, and artifact output.
    # This is not a model accuracy test.
    def trained_checkpoint(**kwargs):
        assert kwargs["prepared_dir"] == str(prepared)
        assert kwargs["output_dir"] == str(output)
        output.mkdir(exist_ok=True)
        checkpoint = output / "best_model.pt"
        checkpoint.write_bytes(template.read_bytes())
        return train.TrainSummary(0.5, 0.5, classes, str(checkpoint))
    monkeypatch.setattr(train, "_run_resnet18_training", trained_checkpoint)

    def run_workflow():
        if entrypoint == "colab":
            return colab_compatible.run_all_for_colab(
                raw_dir=str(raw), prepared_dir=str(prepared), output_dir=str(output),
                resize=None, epochs=1, workers=0, preprocess_workers=1, model_type="resnet18",
            )[1]
        arguments = ["main.py", "train" if entrypoint == "cli-train" else "all",
                     "--prepared-dir", str(prepared), "--output-dir", str(output),
                     "--epochs", "1", "--workers", "0", "--model-type", "resnet18"]
        if entrypoint == "cli-all":
            arguments += ["--raw-dir", str(raw), "--preprocess-workers", "1", "--preprocess-device", "cpu"]
        monkeypatch.setattr(sys, "argv", arguments)
        cli.main()
    summary = run_workflow()
    gate_path = output / "verifier" / "feature_verifier.pt"
    assert gate_path.is_file()
    if summary is not None:
        assert Path(summary.feature_verifier_path) == gate_path
    if entrypoint == "cli-train":
        run_workflow()  # Retraining into the same output folder refreshes the gate.
