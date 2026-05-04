# OCRBench Results Analysis Demo

This repository provides demo scripts to showcase the Consensus Entropy (CE) mechanism using OCRBench evaluation results. For ease of reproduction, the CE calculation in this demo primarily uses edit distance (Levenshtein distance) as the similarity metric.

## Requirements

```setup
pip install pandas Levenshtein numpy openpyxl
```

## Demo Scripts

### 1. CE-based Score Analysis

Demonstrates how model performance changes under different CE thresholds:

```bash
python cal_avg_scores_CE_thresholds.py \
    ./samples/InternVL2_5-8B_OCRBench.xlsx \
    ./samples/Qwen2-VL-7B-Instruct_OCRBench.xlsx \
    -o ./res_avg_scores_CE/test.json
```

This script shows how filtering data based on CE thresholds affects model performance on OCRBench.

### 2. Multi-Model Ensemble with CE

Demonstrates how to aggregate multiple model results using CE:

```bash
python ensemble_consensus_entropy_from_xlsx.py \
    ./samples/InternVL2_5-8B_OCRBench.xlsx \
    ./samples/Qwen2-VL-7B-Instruct_OCRBench.xlsx \
    ./samples/Qwen2.5-VL-7B_OCRBench.xlsx \
    -o ./res_ensemble/
```

This script shows how to combine predictions from multiple models using CE-based ensemble.

## Input/Output

- Input: Excel files (.xlsx) from VLMEvalkit's OCRBench evaluation
- Output: JSON files with analysis results and Excel files for ensemble results
