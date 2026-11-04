# Damage mapping from ground-level photos of buildings after an earthquake

Code and model for the paper *Post-Earthquake Building Damage Mapping for Urban Search and Rescue: Transformer-Based Transfer Learning on Ground-Level Images for INSARAG Worksite Triage* (submitted to the International Journal of Disaster Risk Reduction).

A SegFormer model, pretrained on the Cityscapes urban dataset, is fine-tuned on 361 geotagged photographs taken by USAR engineers after the February 2023 earthquakes in Türkiye. It labels every pixel as background, machinery, personnel or one of the six INSARAG worksite triage categories A to F. The pixel labels of each photo are reduced to one triage score, and the scores are interpolated between photo locations into damage heatmaps.

| ID | Class | Damage characteristics |
|---|---|---|
| 0 | Background | |
| 1 | Machinery | |
| 2 | Personnel | |
| 3 | A | intact |
| 4 | B | minor damage (broken windows, etc.) |
| 5 | C | standing building with cracks |
| 6 | D | fallen building, mosaic of pieces including hanging elements, voids |
| 7 | E | pancake collapse, no voids |
| 8 | F | debris pile |

## Notebooks

| Notebook | What it does | Paper |
|---|---|---|
| [`01_segformer_training_and_evaluation`](notebooks/01_segformer_training_and_evaluation.ipynb) | data download, stratified split, transfer learning, pixel-level evaluation, image-level triage scores | Tables 2–4, Figures 5–7 |
| [`02_prediction_error_and_spatial_analysis`](notebooks/02_prediction_error_and_spatial_analysis.ipynb) | image-level error, choice of interpolation method, heatmaps from ground truth and predictions, error against distance | Table 5, Table 6, Figures 8 and 11 |
| [`03_publish_model_to_huggingface`](notebooks/03_publish_model_to_huggingface.ipynb) | how the released checkpoint was prepared and uploaded | |

Each notebook opens in Google Colab from the badge at its top and runs on a T4 GPU. The notebooks keep the outputs of the runs reported in the paper. Notebook 02 reads two CSV files written by notebook 01, and the trained model is downloaded from Hugging Face if it is not found locally, so notebook 02 can be run without retraining.

## Data

The photographs and masks are on Kaggle under CC BY-NC 4.0: [images](https://www.kaggle.com/datasets/leeelyehezkel/turkey-gt-images) and [masks](https://www.kaggle.com/datasets/leeelyehezkel/turkey-gt-masks). The mask pixel values are the class IDs in the table above. Downloading them needs a free Kaggle API token; notebook 01 explains the two ways to provide it. The split is 249 training, 53 validation and 59 test images.

## Model

The trained weights are at [huggingface.co/leeely/segformer-b2-turkey-building-damage](https://huggingface.co/leeely/segformer-b2-turkey-building-damage), with a model card. Loading them:

```python
import torch
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download

model = smp.from_pretrained("smp-hub/segformer-b2-1024x1024-city-160k")
model.segmentation_head = torch.nn.Conv2d(model.segmentation_head[0].in_channels, 9, kernel_size=1)
checkpoint = torch.load(hf_hub_download("leeely/segformer-b2-turkey-building-damage", "best_model.pt"), map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
```

## Results

On the 59 test images:

| Measure | Value |
|---|---|
| Pixel level, overall accuracy (mean of the confusion-matrix diagonal) | 0.56 |
| Image level, MAE of the weighted-average score (classes) | 0.56 |
| Image level, MAE of the mode score (classes) | 1.00 |
| Image level, balanced accuracy, weighted average rounded / mode | 0.49 / 0.43 |

Background, machinery, personnel and category E are recognised best, B and D worst, and most errors fall between neighbouring categories. Heatmaps interpolated with IDW (power 5) from the predicted scores of all 361 photos reach a 2D correlation of 0.81 with the heatmaps from the engineers' labels (MAE 0.37 classes).

One detail of the training run is described in notebook 01, Section 6: the augmentation was also active when the validation and test losses were computed. This affects those loss values and the choice of the best epoch, not any of the results above, which were computed on the original test photos.

## Adapting the pipeline

The settings are collected in a configuration cell at the top of each notebook. To train on other data, point the image and mask folders at your own files and change the number of classes; the masks must store class IDs as pixel values. The aggregation to one score per photo assumes ordered damage classes with consecutive IDs (3 to 8 here). The interpolation methods compared in notebook 02 are given as a list of settings in its method-comparison cell (Section 8), so other IDW powers or kriging variograms can be added there.

## Repository layout

```
notebooks/     the three notebooks
huggingface/   model card and licence uploaded with the weights
licenses/      licence texts of this code and of the SegFormer weights
requirements.txt
CITATION.cff
```

## Licences

- **Code:** [PolyForm Noncommercial 1.0.0](LICENSE). Free for research, teaching and other non-commercial use.
- **Model weights:** derived from NVIDIA's SegFormer and distributed under the [NVIDIA Source Code License for SegFormer](licenses/NVIDIA-SegFormer-LICENSE.txt), which allows non-commercial (research or evaluation) use only.
- **Data:** CC BY-NC 4.0, as stated on the Kaggle pages.

## Citation

Please cite the code and model as given in [`CITATION.cff`](CITATION.cff), and the data as:

LeeEl Yehezkel and Barak Fishbain (2026). Turkey gt images [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/16676318

Barak Fishbain, Alisa Yudovsky and LeeEl Yehezkel (2026). Turkey gt masks [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/16676359
