# Consensus Entropy

[![PyPI](https://img.shields.io/pypi/v/consensus-entropy?color=blue)](https://pypi.org/project/consensus-entropy/)
[![Python](https://img.shields.io/pypi/pyversions/consensus-entropy)](https://pypi.org/project/consensus-entropy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-2504.11101-b31b1b.svg)](https://arxiv.org/abs/2504.11101)
[![Project Page](https://img.shields.io/badge/Project-Page-6f42c1.svg)](https://tianyilt.github.io/consensus-entropy/)

Official implementation of **Consensus Entropy: Harnessing Multi-VLM Agreement for Self-Verifying and Self-Improving OCR**.

Consensus Entropy is a lightweight Python package for measuring agreement among multiple OCR predictions. Given several candidate strings from different vision-language models, OCR engines, or decoding runs, it assigns each candidate a consensus-entropy score based on normalized string disagreement. Lower scores indicate stronger agreement with the candidate set and can be used as a simple, model-agnostic confidence signal for OCR verification and result selection.

## Links

- **Project page**: https://tianyilt.github.io/consensus-entropy/
- **Paper**: https://arxiv.org/abs/2504.11101
- **Code demo**: [`Code_Demo_CE_Ensemble_and_Filtering/`](Code_Demo_CE_Ensemble_and_Filtering/)
- **PyPI package**: https://pypi.org/project/consensus-entropy/
- **Human-labeled OCR dataset**: to be added once the public Hugging Face dataset card is confirmed.

## News

- **2026-05** — Our paper **Consensus Entropy: Harnessing Multi-VLM Agreement for Self-Verifying and Self-Improving OCR** has been accepted to **CVPR 2026**. 🎉
- **2026-05** — Project page and supplementary code demo released.
- **2025-04** — Paper released on arXiv: [arXiv:2504.11101](https://arxiv.org/abs/2504.11101).

## Highlights

- **Self-verifying OCR signal**: estimate which OCR result is most consistent with a group of candidates.
- **Model-agnostic**: works with outputs from any OCR system or vision-language model.
- **Minimal dependency footprint**: small Python package built around normalized Levenshtein distance.
- **Multilingual text support**: applicable to English, Chinese, and other Unicode strings.
- **Simple API**: compute per-candidate scores or directly retrieve the best OCR result.

## Installation

Install from PyPI:

```bash
pip install consensus-entropy
```

Or install from source:

```bash
git clone https://github.com/Aslan-yulong/consensus-entropy.git
cd consensus-entropy
pip install -e .
```

## Quick Start

### Compute consensus entropy scores

```python
from consensus_entropy import calculate_consensus_entropy

ocr_results = [
    "Hello World",
    "Hello Wrld",
    "Hallo World",
]

scores = calculate_consensus_entropy(ocr_results, task_type="ocr")
print([f"{score:.4f}" for score in scores])
# ['0.0909', '0.1364', '0.1364']
```

Each score measures the average normalized edit distance between one candidate and all other candidates. The lower the value, the closer the candidate is to the group consensus.

### Select the best OCR result

```python
from consensus_entropy import get_best_ocr_result

ocr_results = ["Test1", "Test2", "Text2"]
best_result, best_score = get_best_ocr_result(ocr_results, task_type="ocr")

print(best_result)              # Test2
print(f"{best_score:.4f}")      # 0.2000
```

### Measure pairwise OCR difference

```python
from consensus_entropy import calculate_ocr_difference

score = calculate_ocr_difference("Hello World", "Hello Wrld")
print(f"{score:.4f}")
# 0.0909
```

## API Reference

### `calculate_ocr_difference(a, b)`

Computes the normalized Levenshtein distance between two strings.

- **Input**: two string-like values.
- **Output**: a float in `[0, 1]` for most practical OCR cases, where `0.0` means exact match.

### `calculate_consensus_entropy(strings, task_type="ocr")`

Computes a consensus-entropy score for each candidate string.

- **Input**: a list of at least two strings.
- **Output**: a list of floats with the same length as the input.
- **Current task type**: `"ocr"`.

### `get_best_ocr_result(strings, task_type="ocr")`

Returns the candidate with the lowest consensus-entropy score.

- **Input**: a list of at least two strings.
- **Output**: `(best_result, best_score)`.

## Supplementary Demo

The supplementary material includes a compact OCRBench demo for CE filtering and CE-based multi-model ensembling. It consumes VLMEvalKit-style OCRBench Excel outputs and produces JSON/XLSX analysis files.

```bash
cd Code_Demo_CE_Ensemble_and_Filtering
pip install pandas Levenshtein numpy openpyxl

# CE threshold analysis
python cal_avg_scores_CE_thresholds.py   ./samples/InternVL2_5-8B_OCRBench.xlsx   ./samples/Qwen2-VL-7B-Instruct_OCRBench.xlsx   -o ./res_avg_scores_CE/test.json

# CE-based multi-model ensemble
python ensemble_consensus_entropy_from_xlsx.py   ./samples/InternVL2_5-8B_OCRBench.xlsx   ./samples/Qwen2-VL-7B-Instruct_OCRBench.xlsx   ./samples/Qwen2.5-VL-7B_OCRBench.xlsx   -o ./res_ensemble/
```

## Dataset

We also maintain a human-labeled OCR evaluation dataset used for validating Consensus Entropy against human quality judgments. The public Hugging Face link will be added here once the dataset card is confirmed.

## When to Use Consensus Entropy

Consensus Entropy is useful when you have multiple OCR hypotheses for the same image or document region, for example:

- outputs from several VLMs/OCR engines;
- multiple prompts or decoding settings for the same model;
- repeated OCR runs under different preprocessing pipelines;
- self-verification pipelines where external labels are unavailable.

It is especially helpful as a lightweight ranking or filtering signal before downstream correction, human review, or pseudo-label selection.

## Requirements

- Python >= 3.7
- `numpy`
- `python-Levenshtein`

## Limitations

- The current public package focuses on OCR-style string agreement.
- Consensus Entropy is an agreement signal, not a proof of correctness: several systems can agree on the same wrong answer.
- For best results, use diverse OCR/VLM candidates rather than near-duplicate outputs from the same configuration.

## Citation

If you find this project useful, please cite our paper:

```bibtex
@misc{zhang2025consensusentropyharnessingmultivlm,
      title={Consensus Entropy: Harnessing Multi-VLM Agreement for Self-Verifying and Self-Improving OCR},
      author={Yulong Zhang and Tianyi Liang and Xinyue Huang and Erfei Cui and Xu Guo and Pei Chu and Chenhui Li and Ru Zhang and Wenhai Wang and Gongshen Liu},
      year={2025},
      eprint={2504.11101},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.11101}
}
```

The camera-ready CVPR 2026 citation will be updated once the official proceedings entry is available.

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

This repository accompanies the paper **Consensus Entropy: Harnessing Multi-VLM Agreement for Self-Verifying and Self-Improving OCR**. We thank the research community for open OCR and vision-language model resources that make reproducible OCR verification research possible.
