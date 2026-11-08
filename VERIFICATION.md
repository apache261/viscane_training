# Rejecting unrecognized images

The variety models assign every readable image one of their learned labels.
The API now checks similarity to known sugarcane features first. This works
with the existing sugarcane-only dataset; it does not require new negative
labels or retraining the variety classifiers.

The verifier extracts the trained ResNet18's 512-dimensional features before
its classification layer. It compares an image to a reference bank selected
from the training split using nearest cosine distance. A distance above the
validation-selected threshold returns `uncertain` / `unrecognized_image` and
prevents variety/maturity prediction. This approach is informed by
[nearest-neighbor out-of-distribution detection](https://proceedings.mlr.press/v162/sun22d.html).
It is a similarity heuristic: unrelated images with similar features can pass.

## Use the existing training workflow

Trainers keep setting their own dataset and output paths as shown in the
README. ResNet18 training automatically builds the feature verifier from the
same prepared dataset. There are no additional required arguments or setup
steps for a new training run.

```bash
python3 main.py all \
  --raw-dir /path/to/your_raw_dataset \
  --prepared-dir data/prepared \
  --output-dir artifacts/resnet18 \
  --model-type resnet18
```

The same happens with `main.py train`, `bare.py`, API training, and the existing
Colab functions. For example, keep your original Colab call:

```python
from sugarcane_variety.colab_compatible import run_all_for_colab

prep, train = run_all_for_colab(
    raw_dir="/content/drive/MyDrive/sugarcane_raw",
    prepared_dir="/content/data/prepared",
    output_dir="/content/drive/MyDrive/sugarcane_artifacts",
    label_mode="variety_maturity",
    workers=2,
    model_type="resnet18",
)
print(train.feature_verifier_path)
```

`bare.py` retains the path constants at the top of its file. It deletes previous
prepared data and its training outputs before starting a full run, as before.

The new outputs follow the trainer's chosen output directory:

- `<output_dir>/verifier/feature_verifier.pt`: encoder weights, training reference
  features, preprocessing version, threshold, and evaluation metadata together.
- `<output_dir>/verifier/feature_verifier_metadata.json`: readable policy and
  evaluation summary. `non_sugarcane_false_acceptance_rate` is `null` because no
  negatives were tested.

Retraining in the same output directory refreshes its verifier automatically.
The API loads it from the `verifier/` directory beside the ResNet18 checkpoint.
YOLOv8 predictions use this same verifier, so train ResNet18 as well when using
YOLOv8. No separate binary dataset is required.

## Existing checkpoints and calibration details

Only older training outputs need the standalone command. Supply the prepared
dataset used for that checkpoint; the output defaults to `verifier/` beside it:

```bash
python3 main.py build-verifier \
  --prepared-dir /path/to/existing/prepared \
  --checkpoint-path /path/to/trained/best_model.pt
```

Or call `build_feature_verifier(prepared_dir=..., checkpoint_path=...,
output_dir=...)` from `sugarcane_variety.colab_compatible` for an older Colab run.
The standalone builder preserves an existing verifier unless explicitly called
with `overwrite=True`; automatic training refreshes its own verifier.

The gate extracts training references, calibrates on validation images, and
reports test sugarcane false rejections. Exact duplicates across splits cause
an error. Check near duplicates with `audit-splits`, and split by capture
session where possible. A failed calibration fails the training workflow
instead of reporting a ready verifier.

The default acceptance quantile is 0.95: the higher empirical 95th percentile
of validation distances. This is a starting operating point, not a guarantee
of 95% acceptance on new images. Lower values make the check stricter and can
reject more valid sugarcane photos. Do not tune it using the test split. The
reference bank uses at most 256 training examples per class, selected with the
training seed. Advanced calibration options remain on the standalone builder.

## Serve and handle responses

The API discovers `verifier/feature_verifier.pt` beside the loaded ResNet18
checkpoint, including when that checkpoint is selected through `/models/load`.
An optional override is available for separately stored bundles:

```bash
SUGARCANE_VERIFIER_CHECKPOINT=/path/to/feature_verifier.pt \
python3 -m uvicorn sugarcane_variety.api:app --host 127.0.0.1 --port 8000
```

Restart the API or reload the ResNet18 model after changing its verifier. `/health` includes
`prediction_ready`, verifier loading errors, and the policy. A missing, invalid,
or failing verifier gives HTTP 503 for prediction; the server never silently
bypasses the check. At least one variety model must also be loaded.

Both `/predict` and `/predict/annotated` apply the same verifier before running
either classifier:

| Outcome | HTTP | Response |
| --- | --- | --- |
| Similar to known examples | 200 | `/predict`: `verification` plus existing `models` results; annotated endpoint: PNG with `X-Verification-Status: accepted` |
| Too far from known examples | 422 | JSON with `verification.status: uncertain`, `verification.reason: unrecognized_image`, and `models: {}` |
| Unreadable image | 400 | Invalid image error |
| Verifier/model unavailable | 503 | Setup/unavailability error |

Clients must handle the JSON error response from `/predict/annotated` instead
of assuming every response is a PNG. The verifier's distance is not a
probability of sugarcane being present. A failed check means unsupported or
unrecognized, not a confirmed non-sugarcane diagnosis. Poor lighting, new
backgrounds, and valid varieties absent from the dataset can also fail.

The gate is separate from the existing variety exports. Existing Android ONNX
and Lite classifiers do not gain rejection automatically. This change does
not include a mobile export of the feature verifier; metadata marks that
limitation explicitly. API checks also apply when YOLOv8 is the only loaded
variety classifier, using the encoder embedded in the verifier bundle.

## Verification and remaining evidence

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q
```

Tests cover feature distances, threshold selection, invalid artifacts,
duplicate leakage, real encoder artifact building/loading, and routed API
behavior for accepted, unrecognized, corrupt, and unavailable inputs.
Synthetic images and mocked classifier scores verify software behavior only.
Evaluate real sugarcane photos for false rejections and collect unrelated images
for false acceptances before claiming reliable non-sugarcane rejection.
