# Fold-0 Study-Level Fracture Analysis

> Frozen Fold-0 validation; complete physically ordered studies; WL=500/WW=2500; 2.5D; max aggregation; no calibration. Exploratory thresholds are not unbiased test performance.

> Full-study discovery evaluated 1678 DICOM slices. The metadata table contains 1633 rows for these studies; the additional 45 DICOM slices were retained because the competition input is the complete study.

> **Baseline provenance limitation:** The requested V2 epoch-25 random best.pt and its duplicate are zero-byte corrupted files. This comparison uses the surviving genuine seeded-random V1 YOLO11s-P2 best checkpoint (same Fold-0, 768px, 2.5D protocol; best epoch 2, mAP50 0.04758, mAP50-95 0.02694), so conclusions versus the lost V2 baseline are supportive rather than exact.

## A. Checkpoints

| Model | Path | Bytes | SHA256 |
| --- | --- | --- | --- |
| baseline | /home/mahsa-nasehi/Desktop/New Folder/fracture/outputs/fold_0/v1_25d_p2/weights/best.pt | 77516133 | 4c90a583ab4ddfc6896993b73705b269af07065603c5ab97d30bf0c8462ef36f |
| pretrained | /home/mahsa-nasehi/Desktop/New Folder/fracture/outputs/fold_0/v3_25d_p2_coco_pretrained_fixed/weights/best.pt | 19772484 | 0d048bcb3b202438c6af467e4badbc415bdd92f3dfaa4b1064119a3a1e256251 |

## B. Detector metrics

```json
{
  "baseline": {
    "results_path": "/home/mahsa-nasehi/Desktop/New Folder/fracture/outputs/fold_0/v1_25d_p2/results.csv",
    "epochs_completed": 22,
    "best_epoch": 2,
    "precision": 0.2005,
    "recall": 0.0625,
    "mAP50": 0.04758,
    "mAP50_95": 0.02694
  },
  "pretrained": {
    "results_path": "/home/mahsa-nasehi/Desktop/New Folder/fracture/outputs/fold_0/v3_25d_p2_coco_pretrained_fixed/results.csv",
    "epochs_completed": 60,
    "best_epoch": 33,
    "precision": 0.14923,
    "recall": 0.125,
    "mAP50": 0.06802,
    "mAP50_95": 0.03784
  }
}
```

## C. Study-level comparison

| Metric | Random baseline | COCO pretrained |
| --- | --- | --- |
| AUROC | 0.704301 | 0.752688 |
| PR-AUC | 0.388506 | 0.47089 |
| Sensitivity@0.5 | 0.666667 | 0.666667 |
| Specificity@0.5 | 0.709677 | 0.741935 |
| Precision@0.5 | 0.181818 | 0.2 |
| F1@0.5 | 0.285714 | 0.307692 |
| Brier | 0.220744 | 0.193318 |
| Log-loss | 0.693442 | 0.573277 |
| TP | 4 | 4 |
| FP | 18 | 16 |
| TN | 44 | 46 |
| FN | 2 | 2 |

### DIAGNOSTIC / EXPLORATORY ONLY: best-F1 thresholds

- Baseline: threshold=0.852051, F1=0.428571, sensitivity=0.5, specificity=0.919355.
- COCO: threshold=0.94873, F1=0.5, sensitivity=0.333333, specificity=1.

## D. Every positive Fold-0 study

| Series | Patient | GT slices | GT boxes | Baseline max | COCO max | Baseline max on GT | COCO max on GT | Change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2265 | 765202 | 6 | 9 | 0.8521 | 0.9570 | True | True | improved_by_pretrained |
| 3796 | 774302 | 6 | 9 | 0.1412 | 0.9487 | True | True | improved_by_pretrained |
| 271912 | 110ec596f9ba448ad8d35a05eed254e00a1a6fc00ebf2bba8b562c58564068b7 | 5 | 7 | 0.6548 | 0.7568 | False | False | improved_by_pretrained |
| 272528 | 8021fd1fe9a067d4aec4859ab18f2a985d22eab8e7ae8683f2287e783bef0b32 | 14 | 17 | 0.9990 | 0.7373 | True | True | worsened_by_pretrained |
| 9244 | 88028 | 9 | 18 | 0.9443 | 0.2351 | False | True | worsened_by_pretrained |
| 271623 | 95fd74c952aa692b87ab414ebf818ecc073eb43b1ea0ddb7f062393ac6f4eb8c | 16 | 20 | 0.2830 | 0.0772 | False | True | worsened_by_pretrained |

## E. Score distributions

| Model | Class | N | Min | Median | Mean | Max |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | positive | 6 | 0.1412 | 0.7534 | 0.6457 | 0.9990 |
| baseline | negative | 62 | 0.0209 | 0.3412 | 0.3925 | 0.9956 |
| pretrained | positive | 6 | 0.0772 | 0.7471 | 0.6187 | 0.9570 |
| pretrained | negative | 62 | 0.0051 | 0.1981 | 0.3182 | 0.9409 |

See `fold0_score_distribution.png`.

## F. Ranking

```json
{
  "baseline": {
    "positive_ranks": [
      1,
      4,
      8,
      17,
      42,
      59
    ],
    "positive_in_top5": 2,
    "recall_at_top5": 0.3333333333333333,
    "positive_in_top10": 3,
    "recall_at_top10": 0.5,
    "positive_in_top20": 4,
    "recall_at_top20": 0.6666666666666666
  },
  "pretrained": {
    "positive_ranks": [
      1,
      2,
      11,
      14,
      33,
      52
    ],
    "positive_in_top5": 2,
    "recall_at_top5": 0.3333333333333333,
    "positive_in_top10": 2,
    "recall_at_top10": 0.3333333333333333,
    "positive_in_top20": 4,
    "recall_at_top20": 0.6666666666666666
  }
}
```

## G. Slice-level fracture behavior

Meaningful slice prediction is defined as confidence >= 0.01; spatial box detection is IoU >= 0.5 at that confidence. Positive-study details are in `fold0_positive_studies.csv` and all positive visualizations are under `fold0_error_analysis/positive/`.

## H. False negatives

At the official 0.5 study threshold: baseline FN=2, COCO FN=2. Inspect the complete positive table rather than relying only on this threshold.

## I. False positives

At 0.5: baseline FP=18, COCO FP=16. The 32-row union of each model's top-20 negative rankings is in `fold0_top_negative_scores.csv`; images are human-review candidates only.

## J. Box-size findings

```json
{
  "baseline": {
    "detected": {
      "n": 18,
      "median_area_at_768": 4817.25,
      "mean_area_at_768": 5411.5
    },
    "missed": {
      "n": 62,
      "median_area_at_768": 9608.625,
      "mean_area_at_768": 37597.53629032258
    }
  },
  "pretrained": {
    "detected": {
      "n": 23,
      "median_area_at_768": 7632.0,
      "mean_area_at_768": 6584.282608695652
    },
    "missed": {
      "n": 57,
      "median_area_at_768": 9747.0,
      "mean_area_at_768": 39947.64473684211
    }
  }
}
```

## K. Detector-confidence sensitivity

| Model | Detector threshold | AUROC | PR-AUC | Sensitivity@study 0.5 | Detections | Matched GT boxes | GT box recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.001 | 0.7043 | 0.3885 | 0.6667 | 39629 | 42 | 0.5250 |
| pretrained | 0.001 | 0.7527 | 0.4709 | 0.6667 | 1930 | 28 | 0.3500 |
| baseline | 0.005 | 0.7043 | 0.3885 | 0.6667 | 8300 | 22 | 0.2750 |
| pretrained | 0.005 | 0.7527 | 0.4709 | 0.6667 | 856 | 23 | 0.2875 |
| baseline | 0.01 | 0.7043 | 0.3885 | 0.6667 | 3955 | 18 | 0.2250 |
| pretrained | 0.01 | 0.7527 | 0.4709 | 0.6667 | 617 | 23 | 0.2875 |
| baseline | 0.02 | 0.7043 | 0.3885 | 0.6667 | 2022 | 17 | 0.2125 |
| pretrained | 0.02 | 0.7527 | 0.4709 | 0.6667 | 438 | 21 | 0.2625 |
| baseline | 0.05 | 0.7043 | 0.3885 | 0.6667 | 839 | 16 | 0.2000 |
| pretrained | 0.05 | 0.7527 | 0.4709 | 0.6667 | 262 | 19 | 0.2375 |
| baseline | 0.1 | 0.7043 | 0.3885 | 0.6667 | 394 | 12 | 0.1500 |
| pretrained | 0.1 | 0.7339 | 0.4664 | 0.6667 | 179 | 16 | 0.2000 |

Detector confidence threshold is not the final fracture probability threshold; the latter remains 0.5.

## L. Aggregation comparison (DIAGNOSTIC / EXPLORATORY ONLY)

| Model | Aggregation | AUROC | PR-AUC |
| --- | --- | --- | --- |
| baseline | max | 0.7043 | 0.3885 |
| baseline | top3_mean | 0.7339 | 0.4334 |
| baseline | consecutive | 0.7204 | 0.3918 |
| pretrained | max | 0.7527 | 0.4709 |
| pretrained | top3_mean | 0.8038 | 0.5353 |
| pretrained | consecutive | 0.7796 | 0.4985 |

## Required conclusions

### Question 1

Yes. Baseline AUROC/PR-AUC=0.7043/0.3885; COCO=0.7527/0.4709.

### Question 2

COCO matched 23 GT boxes versus 18 for baseline at confidence >=0.01 and IoU >=0.5; this separates actual fracture localization from confidence-only changes.

### Question 3

Positive/negative overlap remains substantial, so the 0.5 errors are not merely a calibration problem.

### Question 4

Lowering detector confidence from 0.01 to 0.001 recovers 5 additional COCO GT-box matches (23 to 28 of 80), but study AUROC, PR-AUC, and sensitivity are unchanged. Thresholding discards some weak localizations, but it is not the main cause of low recall.

### Question 5

No clear evidence that false negatives are disproportionately smaller from this fold.

### Question 6

The selected option below is based on study AUROC and PR-AUC first, with slice/box evidence as supporting analysis.

## Model decision

A. COCO-pretrained V2 should proceed to 5-fold OOF
