# Agent guidance for viscane_train

## Project and entry points

This repository prepares sugarcane images, trains ResNet18 and YOLOv8
classifiers, evaluates them, exports Android artifacts, and serves predictions
and reports through FastAPI. Run commands from the repository root.

- `main.py` delegates to `sugarcane_variety/cli.py`.
- `sugarcane_variety/preprocess.py` handles label collection, image validation,
  resizing, train/validation/test splits, and duplicate audits.
- `sugarcane_variety/train.py` owns training, evaluation, label decoding,
  device transforms, and model exports.
- `sugarcane_variety/colab_compatible.py` wraps the same pipeline for notebooks.
- `sugarcane_variety/api.py` owns model loading, prediction endpoints,
  annotations, training jobs, and HTML reports.
- `sugarcane_variety/verification.py` builds and loads the sugarcane-only feature
  similarity gate. `VERIFICATION.md` describes setup, contracts, and limitations.
- Preserve the README training interface: trainers set `raw_dir`, `prepared_dir`,
  and `output_dir` directly (or the existing CLI flags). Do not add environment
  variables or require trainers to give an agent their dataset path. Shared
  `run_training` builds the verifier automatically after ResNet18 training,
  using the supplied prepared dataset and `<output_dir>/verifier/`. The returned
  `TrainSummary.feature_verifier_path` identifies the bundle. The API discovers
  it beside the loaded ResNet18 checkpoint, including after `/models/load`.
- `bare.py` runs full ResNet18/YOLOv8 training; `diagnose.py` runs multiple
  diagnostic experiments. Both delete previous outputs at startup.
- `README.md` describes usage; research Markdown/PDF files describe historical
  experiments. Verify behavior against source and checkpoint metadata.

## Known issue: unrelated images receive sugarcane predictions

The variety classifiers assign any image one of their learned labels. The API
now uses a separate feature similarity gate before allowing those predictions.
This is not a binary sugarcane detector and can still accept similar unrelated
images. The user has only sugarcane data; do not require nonexistent negative
labels to build the current gate or claim measured accuracy on negative images.

- `_open_uploaded_image` verifies that Pillow can decode the upload and converts
  it to RGB. A valid image file is not necessarily a valid sugarcane image.
- `_predict_probabilities` returns ResNet18 softmax scores or YOLO classification
  probabilities. `_build_predictions` always selects the highest scoring labels.
- Both prediction endpoints require `SugarcaneFeatureVerifier`. Too-distant
  images return HTTP 422 with `uncertain` / `unrecognized_image`, empty `models`,
  and no annotations. Missing/invalid verifier artifacts return HTTP 503.
- Accepted `/predict` calls return results for every loaded model;
  `/predict/annotated` selects ResNet18 when available, otherwise YOLOv8.
- YOLO uses `yolov8n-cls.pt` by default: this is image classification.
  `_draw_prediction_overlay` draws the image perimeter; it does not localize a
  sugarcane object.
- The bundled `sugarcane_artifacts (2).zip` metadata contains eight variety/maturity labels
  for `524`, `847`, and `Mauritio`, with no non-sugarcane class. This describes
  that archive only; inspect the actual loaded checkpoint for current labels.

A high class score is a preference among the learned classes, not proof that
sugarcane is present. A confidence threshold alone cannot reliably reject all
unrelated inputs. The current `unknown` maturity fallback means a missing
decoded maturity label; it is not an out-of-distribution rejection result.

### Rejection requirements

Keep these properties when changing rejection:

1. The current gate uses ResNet18 penultimate features, training references,
   and validation-calibrated cosine distance. It reports only unrecognized
   inputs, not confirmed non-sugarcane. To measure negative accuracy or train a
   binary detector later, collect representative unrelated images. Localization
   requires a detector trained with bounding-box annotations.
2. Separate unreadable files, recognized non-sugarcane images, and uncertain
   inputs in the response contract. Rejected or uncertain inputs must not be
   presented as confirmed variety/maturity predictions or receive maturity
   overlays. Apply the same policy to both prediction endpoints.
3. Define behavior when a required verifier or its metadata is unavailable;
   do not silently accept all images and claim verification succeeded.
4. Choose thresholds using validation data. The current positive-only method
   reports test sugarcane false rejections and marks negative false acceptance
   as unknown. Measure that rate when negative images become available. Keep
   threshold selection off the test set.
5. Version verifier artifacts, preprocessing, labels, and thresholds together.
   Document any API/schema changes and update mobile export metadata when
   exported inference is affected. Report what is still unverified.

## Data and model invariants

- Preserve `variety`, `maturity`, and `variety_maturity` label modes. Combined
  labels use `<variety>__<maturity>`; inspect collection and decoding code before
  adding a rejection label. A new folder alone does not implement rejection.
- Preserve class ordering across datasets, checkpoints, API responses, ONNX/
  Lite exports, and Android metadata. The archived labels include a trailing
  space in `524__NOT_MATURE `; do not rename labels in only one consumer.
- Keep inference resizing, channel order, numeric range, and normalization
  aligned with the corresponding training/evaluation pipeline and export.
- Keep exact duplicates grouped within each class when splitting. Audit both
  exact and near duplicates across splits; do not assume the split is clean.
- Use validation metrics for model/threshold selection and reserve the test
  split for final evaluation. `bare.py` currently compares models by test
  accuracy; disclose that limitation when interpreting its best-model result.
- Preserve CPU/CUDA/MPS support where implemented. API device selection currently
  uses CUDA or CPU; do not claim it uses MPS merely because training supports it.
- Thread new pipeline options through CLI, Colab wrappers, API request models,
  reports, and metadata where applicable.

## Safe workflow and verification

- Inspect `git status --short` first and preserve unrelated changes.
- Use `python3`. If dependencies are needed, use an isolated environment and
  `python3 -m pip install -r requirements.txt`.
- Read CLI options with `python3 main.py --help` and subcommand `--help`.
- Do not run training or destructive preprocessing as a routine smoke check.
  `run_preprocess` removes its existing output directory. Use temporary paths
  and small fixtures for focused verification.
- Start local API verification with
  `python3 -m uvicorn sugarcane_variety.api:app --host 127.0.0.1 --port 8000`.
  Check `/health` and `/classes` before interpreting predictions.
- API checkpoint overrides are `MODEL_CHECKPOINT_RESNET` and
  `MODEL_CHECKPOINT_YOLO`; `SUGARCANE_VERIFIER_CHECKPOINT` selects the gate;
  `ARTIFACTS_DIR` selects the artifacts directory.
  Some README examples use older checkpoint settings and response shapes.
- Run `python3 -m pytest -q` (test dependencies: `requirements-dev.txt`). For
  behavior changes, add focused tests of affected entry points. Cover sugarcane,
  unrelated images, uncertain scores, corrupt uploads, missing verifiers, both
  model paths, and both prediction endpoints. Synthetic/mock scores verify
  response logic; actual held-out images are needed to establish model quality.
- Documentation-only changes need a source consistency and diff check, not
  training. State exactly what was checked and any missing dependencies/data.
- Preserve datasets, checkpoints, archives, and existing research results.
  Do not extract archives over working data, rewrite historical metrics, or
  commit generated binaries/caches as part of an unrelated change.
- Do not print secrets or `.env` values. Do not push or deploy unless requested.
