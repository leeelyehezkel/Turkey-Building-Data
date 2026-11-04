---
library_name: segmentation-models-pytorch
license: other
license_name: nvidia-source-code-license-segformer
license_link: LICENSE
pipeline_tag: image-segmentation
base_model: smp-hub/segformer-b2-1024x1024-city-160k
tags:
- segformer
- semantic-segmentation
- earthquake
- building-damage
- disaster-response
---

# SegFormer-B2 for building damage in post-earthquake photos

A SegFormer-B2 model that labels every pixel of a street-level photo taken after the February 2023 earthquakes in Türkiye. It separates background, heavy machinery and rescue personnel from six levels of building damage, A to F. The per-pixel output can be reduced to one damage score per photo, and geotagged scores can be interpolated into a damage map.

Code, training and the full evaluation: [github.com/leeelyehezkel/Turkey-Building-Data](https://github.com/leeelyehezkel/Turkey-Building-Data).

## Classes

| ID | Class | Meaning |
|---|---|---|
| 0 | Background | everything not in another class |
| 1 | Machinery | heavy machinery at the site |
| 2 | Personnel | people |
| 3 | Damage A | intact |
| 4 | Damage B | minor damage (broken windows etc.) |
| 5 | Damage C | standing building with cracks |
| 6 | Damage D | fallen building, mosaic of pieces with hanging elements and voids |
| 7 | Damage E | pancake collapse, no voids |
| 8 | Damage F | debris pile |

The damage levels are the INSARAG worksite triage categories A to F, as described in the accompanying paper.

## Training

The model starts from `smp-hub/segformer-b2-1024x1024-city-160k` (Cityscapes). The 19-class segmentation head was replaced with a new, randomly initialised 9-class head. The encoder was frozen and only the decoder and head were trained. Inputs are 1024 × 1024.

The data are 361 annotated photos ([images](https://www.kaggle.com/datasets/leeelyehezkel/turkey-gt-images), [masks](https://www.kaggle.com/datasets/leeelyehezkel/turkey-gt-masks)), split 249 / 53 / 59 into train, validation and test with a multilabel stratification on the share of each damage class. Training used cross-entropy weighted by median class frequency, batch size 4, augmentation (horizontal flips, 90° rotations, random shift, scale and rotation, colour jitter), Adam (learning rate 1e-3), a reduce-on-plateau schedule and early stopping with patience 8. The released checkpoint is from epoch 21, the epoch with the lowest validation loss (1.563). Test loss is 1.406. In this run the augmentation was also applied to the validation and test images when the losses were computed, so both losses are on augmented images. The figures below are computed on the original, unaugmented test photos.

## Results on the test set (59 photos)

| Measure | Value |
|---|---|
| Pixel level: overall accuracy (mean of the diagonal of the row-normalised confusion matrix) | 0.56 |
| Pixel level: mean IoU, all 9 classes / damage classes A–F | 0.169 / 0.108 |
| Image level: MAE of the weighted-average score, in classes | 0.56 |
| Image level: MAE of the mode score, in classes | 1.00 |
| Image level: balanced accuracy, weighted average rounded / mode | 0.49 / 0.43 |

The paper also compares damage heatmaps interpolated (IDW, power 5) from predicted and from ground-truth image scores. That comparison uses all 361 geotagged photos, including the training images, and gives a 2D correlation of 0.81 and an MAE of 0.37 classes.

At pixel level the model is most accurate for background, machinery, personnel and damage level E, and least accurate for levels B and D. Most errors fall between neighbouring damage levels. At image level, the weighted-average score has almost no overall bias; its largest errors are in levels A to C, which it tends to overestimate.

These numbers come from a small dataset from one earthquake. The model is a research artefact meant for first-pass decision support, not for final worksite classification.

## Use

```python
import torch
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download

model = smp.from_pretrained("smp-hub/segformer-b2-1024x1024-city-160k")
model.segmentation_head = torch.nn.Conv2d(
    model.segmentation_head[0].in_channels, 9, kernel_size=1
)

path = hf_hub_download("leeely/segformer-b2-turkey-building-damage", "best_model.pt")
checkpoint = torch.load(path, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

Preprocess images the same way as the base model (`albumentations.Compose.from_pretrained("smp-hub/segformer-b2-1024x1024-city-160k")`), resize to 1024 × 1024, and take the argmax over the 9 output channels. `best_model.pt` also stores the epoch (`epoch`), the validation loss (`best_val_loss`), the per-epoch training and validation losses (`train_losses`, `val_losses`) and the class weights used in the loss (`class_weights`).

## Licence

The weights are derived from NVIDIA's SegFormer and are distributed under the [NVIDIA Source Code License for SegFormer](LICENSE), which allows non-commercial use (research or evaluation) only. The code in the GitHub repository is under PolyForm Noncommercial 1.0.0.

## Data

The photos and masks are on Kaggle under CC BY-NC 4.0. Please cite them if you use them:

LeeEl Yehezkel and Barak Fishbain (2026). Turkey gt images [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/16676318

Barak Fishbain, Alisa Yudovsky and LeeEl Yehezkel (2026). Turkey gt masks [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/16676359

## Citation

See [`CITATION.cff`](https://github.com/leeelyehezkel/Turkey-Building-Data/blob/main/CITATION.cff) in the GitHub repository.
