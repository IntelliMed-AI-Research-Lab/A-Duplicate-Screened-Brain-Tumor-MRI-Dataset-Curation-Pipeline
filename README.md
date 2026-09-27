# BDNeuro-MRI: A Duplicate-Screened Brain Tumor MRI Dataset & Curation Pipeline

A fully documented, auditable curation and preprocessing pipeline for building a leakage-free, clinically sourced brain tumor MRI classification dataset — demonstrated on 5,941 MRI images collected from a Bangladeshi clinical facility, and validated by benchmarking seven deep-learning architectures.

> 📄 Companion MethodsX article: *A Duplicate-Screened Data Curation and Preprocessing Pipeline for Building a Bangladeshi Clinical Brain Tumor MRI Classification Dataset*
> 📦 Dataset (images, splits, pretrained weights): [Mendeley Data — DOI: 10.17632/zwr4ntf94j.7](https://doi.org/10.17632/zwr4ntf94j.7)

---

## Overview

Public brain tumor MRI collections are frequently distributed with **undocumented duplicate content**, letting the same underlying scan (re-exported, re-compressed, or minimally cropped) leak across the train/validation/test splits. This silently inflates reported accuracy and makes results hard to trust or reproduce.

This repository provides:

1. **The full method paper** (Word document, tracked-changes version) describing a step-by-step, reusable curation protocol.
2. **An analysis notebook** that independently verifies the released dataset — re-checking class balance, split ratios, and, most importantly, running a post-split **content-hash leakage check** to confirm zero overlap between splits.

The curation protocol combines:

- **MD5 exact-duplicate screening**
- **Perceptual-hash (pHash) near-duplicate screening** (Hamming distance ≤ 5)
- **Post-split content-hash verification** confirming zero train/val/test overlap — a step largely absent from existing public brain tumor MRI collections, even ones that already perform MD5 screening.

The method is dataset-agnostic: Section 6 of the paper ("Applying the method to a new clinical pool") gives a generalized recipe any group can follow with their own clinical MRI pool.

---

## Dataset at a glance

| | |
|---|---|
| **Classes** | Glioma, Meningioma, Pituitary tumor, No tumor |
| **Source** | Epic Health Care, Chittagong, Bangladesh (Ethics Ref: EHC/CARD/2025/0047) |
| **Initial pool** | 11,148 clinically sourced images |
| **Exact duplicates removed (MD5)** | 833 |
| **Near-duplicates removed (pHash, Hamming ≤ 5)** | 4,374 |
| **Final curated dataset** | 5,941 verified-unique images |
| **Split** | 70% train / 15% val / 15% test, per-class stratified |
| **Cross-split overlap** | Zero (confirmed by content hash) |
| **Image format** | JPEG/PNG, resized to 224×224 for modeling |

### Class-wise split distribution

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Glioma | 1534 | 329 | 328 | 2191 |
| Meningioma | 1035 | 222 | 221 | 1478 |
| No Tumor | 551 | 118 | 118 | 787 |
| Pituitary | 1040 | 223 | 222 | 1485 |
| **Total** | **4160** | **892** | **889** | **5941** |

---

## Curation & preprocessing pipeline

```
Clinical MRI collection (ethics-cleared, anonymized)
        │
        ▼
Duplicate screening  ──►  MD5 exact-duplicate removal (−833)
        │                 Perceptual-hash near-duplicate removal (−4,374)
        ▼
Stratified 70/15/15 split (per-class)
        │
        ▼
Post-split content-hash leakage verification  (0 overlap confirmed)
        │
        ▼
Preprocessing:  Resize (224×224) → Split-conditional augmentation
                → Tensor conversion → Channel-wise normalization
        │
        ▼
Benchmarking: CNN · ViT · Hybrid CNN–ViT · ResNet50 (frozen/fine-tuned)
              · DenseNet121 (frozen/fine-tuned)
```

**Training-only augmentation** (5 stochastic operations, intensity-preserving):
random rotation (±15°), horizontal flip (p=0.5), vertical flip (p=0.2), affine translation (±10%), randomly resized crop (scale 0.8–1.0). Brightness/color/noise jitter is deliberately excluded to keep every augmented sample a physically plausible clinical image.

Implemented with `torchvision.transforms` and a PyTorch `DataLoader` (batch size 16, 4 workers).

---

## Baseline benchmark results (held-out test set, 889 images)

| Architecture | Paradigm | Test Accuracy | Test F1 |
|---|---|---|---|
| CNN | From scratch | 0.8470 | 0.8457 |
| ViT | From scratch | 0.8256 | 0.8240 |
| Hybrid CNN–ViT | From scratch | 0.8189 | 0.8201 |
| ResNet50 | Frozen pretrained | 0.8031 | 0.8022 |
| DenseNet121 | Frozen pretrained | 0.8706 | 0.8703 |
| ResNet50 | Fine-tuned | 0.9708 | 0.9708 |
| **DenseNet121** | **Fine-tuned** | **0.9831 (best)** | **0.9831 (best)** |

These seven architectures are trained with standard, unmodified pipelines — their role is purely diagnostic, confirming the curated splits are learnable and free of leakage-inflated performance. The methodological contribution of this work is the upstream curation/verification protocol, not the model architectures.

---

## Repository contents

```
.
├── README.md
├── paper/
│   └── BDNeuro-MRI_MethodsX_Revised_TrackedChanges.docx   # Full method paper
└── notebook/
    └── Brain_Tumor_MRI_Dataset_Final_Analysis.ipynb        # Dataset verification & analysis
```

### About the notebook

`Brain_Tumor_MRI_Dataset_Final_Analysis.ipynb` runs on the **final, already-split** dataset export (train/val/test folders) and performs independent verification and reporting — it does not rebuild or re-split the data. It:

- Discovers all splits, classes, and images from the exported folder structure
- Recomputes MD5 content hashes and confirms **zero image overlap** across train/val/test
- Produces class × split distribution tables and per-class split-balance checks
- Generates publication-quality figures: distribution bar/pie/donut/heatmap plots, representative sample images per class, pixel-intensity histograms, class-wise mean images, and image-quality boxplots (contrast, sharpness, SNR)
- Records the baseline model benchmark table
- Auto-generates a dataset card (`README.md`), `class_distribution.csv`, and `model_benchmark.csv` for the released dataset folder

**Dependencies:** `numpy`, `pandas`, `matplotlib`, `scipy`, `Pillow`, `tqdm` (originally run in Google Colab; the `google.colab`-specific upload cell can be swapped for a local file path when running elsewhere).

---

## Reproducing the method on your own clinical pool

1. Collect and fully anonymize images under appropriate institutional ethics clearance; organize by diagnostic label.
2. Compute an MD5 hash per image; remove byte-identical duplicates.
3. Compute a perceptual hash per remaining image; remove near-duplicates below a chosen Hamming-distance threshold (5 was used here).
4. Perform per-class stratified train/val/test splitting.
5. Run a content-hash comparison across every split pair to confirm zero overlap before any model development.
6. Apply resize → split-conditional augmentation → tensor conversion → normalization, adjusting resolution/augmentation/normalization constants to your target modality and architecture.

---

## Ethics statement

All MRI images were prospectively collected from Epic Health Care, Chittagong, Bangladesh, under institutional ethics clearance (Ref: EHC/CARD/2025/0047), in accordance with the Declaration of Helsinki. Informed consent was obtained from all patients. All images were fully anonymized prior to release, with no personally identifiable information retained.

## Limitations

- Sourced from a single clinical facility under a fixed set of scanner protocols; may require recalibration for generalization to other centres.
- The pHash near-duplicate threshold (Hamming ≤ 5) was chosen empirically and may need adjustment for pools with different resolution/compression characteristics.
- Stratified splitting preserves class proportions but does not correct pre-existing class imbalance in the source pool.
- Screens and labels at the whole-image level only — no pixel-level segmentation masks, bounding boxes, or MRI sequence metadata (T1/T2/post-contrast).

## Citation

If you use this dataset or method, please cite the associated MethodsX article (full citation to be added once published) and the Mendeley Data record:

```
DOI: 10.17632/zwr4ntf94j.7
```

## Authors

Md Irfanul Kabir Hira¹\*, Umme Sara¹, Mst Moriom Akter Bithee¹, Md Sohag Hossain¹, Md. Kowsar Ahmed¹, Abdullah Al Towsif¹, Md Mahmudul Hasan¹, Mohammad Shorif Uddin²

¹ Department of Computer Science and Engineering, National Institute of Textile Engineering and Research (NITER), Constituent Institute of University of Dhaka, Savar, Dhaka-1350, Bangladesh
² Department of Computer Science and Engineering, Jahangirnagar University, Savar, Dhaka-1342, Bangladesh

\* Corresponding author: irfanhira11@niter.edu.bd

## Acknowledgments

The authors thank the consultant neurologist and the Department of Neuromedicine, Epic Health Care, Chittagong, Bangladesh, for clinical supervision and support, and the Department of Computer Science and Engineering, NITER (University of Dhaka), for computational and institutional support.

## License

Add your chosen license here (e.g., MIT for code, CC BY 4.0 for the paper/dataset card). The released dataset itself is distributed under the terms specified on its [Mendeley Data page](https://doi.org/10.17632/zwr4ntf94j.7).
