import pytest
import torch
from PIL import Image

from sugarcane_variety.train import _build_resnet18
from sugarcane_variety.verification import PREPROCESSING


@pytest.fixture(autouse=True)
def bound_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def policy():
    return {"version": 1, "method": "resnet18_nearest_cosine", "preprocessing": PREPROCESSING,
            "distance_threshold": 0.2}


@pytest.fixture
def sugarcane_dataset(tmp_path):
    """Real ImageFolder files and ResNet weights; not a trained quality benchmark."""
    torch.manual_seed(17)
    classes = ["524__MATURE", "847__MATURE"]
    for split in ("train", "val", "test"):
        for label in classes:
            folder = tmp_path / "prepared" / split / label
            folder.mkdir(parents=True)
            for index in range(2):
                pixels = torch.randint(0, 256, (40, 48, 3), dtype=torch.uint8).numpy()
                Image.fromarray(pixels).save(folder / f"{index}.png")
    checkpoint = tmp_path / "model.pt"
    model = _build_resnet18(2, pretrained=False)
    torch.save({"classes": classes, "image_size": 32, "model_type": "resnet18",
                "model_state_dict": model.state_dict()}, checkpoint)
    return tmp_path / "prepared", checkpoint
