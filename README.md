# xView2 Building Damage Segmentation

A PyTorch pipeline that segments per-building disaster damage from paired
pre-/post-disaster satellite imagery, using the [xBD](https://xview2.org/)
dataset. Given a pre-disaster and post-disaster image of the same location,
the model predicts a per-pixel damage class for every building footprint.

Built as part of an ACM AI project team (Spring 2026), mentored by Alan.
This repo contains the data pipeline and segmentation model I personally
implemented; it reflects my individual contribution to the team's project,
not the team's full codebase.

## Problem

Natural disasters (hurricanes, wildfires, earthquakes, floods) damage
structures at a scale that's slow and dangerous for humans to assess
building-by-building. Satellite imagery captured before and after a disaster
can be used to automatically flag which buildings were damaged and how
severely, to help prioritize emergency response.

The team explored two framings of this problem:
1. **Classification** — label the single worst-damaged structure in an image.
2. **Segmentation** — label damage per-pixel/per-building footprint across
   the whole image.

This repo implements the **segmentation** approach.

## Approach

- **Input**: a 6-channel tensor — pre-disaster RGB stacked with post-disaster
  RGB for the same location ([dataset.py](dataset.py)).
- **Labels**: xBD ships building footprints as WKT polygons with a damage
  `subtype` per building. [mask_utils.py](mask_utils.py) rasterizes those
  polygons into a dense per-pixel integer mask:
  `0=no-damage, 1=minor, 2=major, 3=destroyed, 4=background`, with
  unclassified/ambiguous polygons mapped to an ignore index (255) excluded
  from the loss.
- **Model**: a standard UNet encoder-decoder with skip connections
  ([model.py](model.py)) — 6 input channels in, 5 class logits out per pixel.
- **Training** ([train.py](train.py)): configurable epochs/batch size/image
  size/base channel width, with horizontal/vertical flip augmentation
  (applied consistently to the image pair and mask), auto device selection
  (CUDA/MPS/CPU), and evaluation via a confusion matrix → per-class IoU, mean
  IoU, and pixel accuracy.

```bash
python train.py --images-dir data/train/images \
                 --labels-dir data/train/labels \
                 --epochs 30 --batch-size 4 --img-size 512
```

## Results

This implementation was not trained to a final checkpoint against the full
xBD dataset as part of the team's timeline, so there are no segmentation
metrics (IoU/pixel accuracy) to report from this code specifically.

Separately, another track of the team's project — a **classification**
approach (DenseCNN and a Vision Transformer, built by teammates, not part of
this repo) — was trained and evaluated. DenseCNN was the stronger of the two,
reaching an overall accuracy of ~0.67 and macro F1 of ~0.50 across the four
damage classes, outperforming the ViT. That work is part of the team's
broader project but isn't reflected in the segmentation code here.

## Limitations & future work

(from the team's project retrospective)
- Limited timeline meant only a subset of disaster events in xBD were used.
- Human-annotated damage labels can be inconsistent across annotators.
- Different disaster types (fire vs. flood vs. earthquake) have visually
  distinct damage signatures that a single model may not generalize across.
- Future work: larger training sets, training the segmentation model here to
  completion, longer training schedules for the classification track.

## Credit

This was a team project for ACM AI, mentored by Alan. This repo isolates and
presents the data pipeline + UNet segmentation implementation I wrote; the
classification (DenseCNN/ViT) work referenced above was built by other team
members.
