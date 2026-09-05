# Model development

The automatic analysis path uses a temporal-background localizer to propose the free subject's body box, then SuperAnimal TopViewMouse / ResNet50 for landmark predictions within that box. It does not substitute a foreground centroid for the anatomical body center or a contour tip for the nose.

## Reviewed labels

`/labeling` reads the local `labeling/queue.json`, source images, optional model suggestions, and annotations. An empty checkout has an empty labeling queue. Its data is not supplied by the repository.

Current queue-building utilities are development scripts tied to the original sample project:

- `scripts/build_label_queue.py` expects the local pilot frame map and/or extracted DLC label frames.
- `scripts/build_proposals.py` can add difficult frames from aligned prediction CSVs.
- The original recordings and pilot outputs must be restored locally to reproduce that exact queue.

Label only the free subject's nose tip, body center excluding tail, and tail base. Each point must be explicitly placed or marked hidden, with annotator and manual-review confirmation. Suggestions remain unreviewed until checked. Older ear annotations are retained in history and original export records.

Exported DLC PNGs and CSV/HDF5 coordinates use the native arena crop. The original full-frame annotations, crop profile, and per-recording split are retained so coordinates can be audited.

## Fine-tuning

After exporting reviewed labels:

```bash
.dlc-env/bin/python scripts/finetune_specialized.py \
  --export labeling/exports/labels-TIMESTAMP-ID
```

This prepares a separate project with SuperAnimal encoder/decoder initialization. Add `--train` to train and evaluate; defaults are 100 pose epochs and 30 detector epochs. Memory replay is disabled, so automatic predictions are not promoted to human labels.

The script currently enforces **675/678 for training and 685 for development validation**. It expects the referenced source videos in `prepared/`. Adapt the split rules explicitly before using another dataset; splitting adjacent frames randomly is not a substitute for independent-recording validation.

Training requires at least 20 visible training frames and 10 visible validation frames. These are minimum software guards, not an adequate-data guarantee. Three-landmark and complete legacy five-landmark exports are supported. Export provenance and authoritative HDF coordinates are checked before training.

Saving labels alone does not train a network. Completing training also does not automatically replace the pretrained model in batch inference; deployment of a validated custom snapshot remains a separate integration step. The current preparation/training workflow has not been exercised end to end with sufficient reviewed data.

## Evaluation

```bash
.venv/bin/python scripts/evaluate_reviewed.py \
  --tracks reports/aligned-predictions.csv \
  --output reports/reviewed-evaluation.json
```

The default evaluates source 685; `--subject all` audits all available reviewed footage. Reports include coverage, median/p95 pixel error, and the fraction within a diagnostic pixel tolerance **including missed predictions**. Body proposal availability and landmark likelihood are not anatomical correctness.

Detector evaluation uses actual detector snapshots separately from the ground-truth-box pose diagnostic. Generalization claims require unseen animals/sessions, a justified positional tolerance, and checks at cup-zone boundaries. Correlated video frames do not provide independent accuracy trials.

References: [DeepLabCut SuperAnimal fine-tuning](https://deeplabcut.github.io/DeepLabCut/examples/COLAB/COLAB_YOURDATA_SuperAnimal.html), [SuperAnimal paper](https://doi.org/10.1038/s41467-024-48792-2).
