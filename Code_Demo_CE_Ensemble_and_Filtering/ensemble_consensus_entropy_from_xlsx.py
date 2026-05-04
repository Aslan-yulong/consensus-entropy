import json
import Levenshtein
import os
import math
import pandas as pd
import argparse
from pathlib import Path

def read_excel(file_path):
    """Read Excel file and return records as dictionary"""
    df = pd.read_excel(file_path)
    return df.to_dict('records')

def write_excel(file_path, data):
    """Write data to Excel file"""
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)

def compute_consensus_entropy(a, b):
    """Calculate consensus entropy (discrepancy) between two string predictions"""
    distance = Levenshtein.distance(a, b)
    max_len = max(len(a), len(b))
    return distance / max_len if max_len else 0.0

def compute_consensus_entropies(predict_dict):
    """Calculate model-wise consensus entropy and overall Shannon entropy"""
    model_names = list(predict_dict.keys())

    # Calculate pairwise consensus entropy
    entropy_matrix = {}
    for model_a in model_names:
        entropy_matrix[model_a] = {}
        for model_b in model_names:
            if model_a != model_b:
                entropy = compute_consensus_entropy(
                    predict_dict[model_a],
                    predict_dict[model_b]
                )
                entropy_matrix[model_a][model_b] = entropy

    # Calculate average consensus entropy per model
    avg_entropies = {
        model: round(sum(entropies.values()) / (len(entropies) or 1), 4)
        for model, entropies in entropy_matrix.items()
    }

    # Calculate overall Shannon entropy
    all_predictions = [v for v in predict_dict.values()]
    frequency = {}
    for pred in all_predictions:
        frequency[pred] = frequency.get(pred, 0) + 1

    total = len(all_predictions)
    overall_entropy = 0.0
    for count in frequency.values():
        p = count / total
        overall_entropy -= p * math.log(p) if p > 0 else 0

    return avg_entropies, round(overall_entropy, 4)

def main():
    parser = argparse.ArgumentParser(
        description='Ensemble predictions with consensus entropy analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='Input Excel files for ensemble analysis'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='./outputs',
        help='Output directory for results'
    )
    
    args = parser.parse_args()

    # Validate input files
    valid_files = []
    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"Error: File not found - {f}")
            continue
        if path.suffix.lower() != '.xlsx':
            print(f"Error: Invalid file format - {f}, only .xlsx files accepted")
            continue
        valid_files.append(f)
    
    if len(valid_files) < 2:
        print("Error: Minimum 2 valid Excel files required for analysis")
        return

    # Process files
    try:
        # Generate output filename
        model_names = [Path(f).stem for f in valid_files]
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_base = output_dir / f"consensus_entropy_{'&'.join(model_names)}"

        # Read all model data
        all_models_data = {name: read_excel(f) for name, f in zip(model_names, valid_files)}
        min_length = min(len(d) for d in all_models_data.values())

        new_data = []
        for i in range(min_length):
            model_predictions = {}
            base_item = None

            for model in model_names:
                item = all_models_data[model][i]
                if base_item is None:
                    # Preserve all non-prediction fields
                    base_item = {k: v for k, v in item.items() if k != 'prediction'}

                # Collect predictions
                model_predictions[model] = {
                    'prediction': str(item['prediction'])
                }

            # Calculate entropy metrics
            raw_predicts = {k: v['prediction'] for k, v in model_predictions.items()}
            model_entropies, overall_entropy = compute_consensus_entropies(raw_predicts)

            # Merge entropy results
            for model, entropy in model_entropies.items():
                model_predictions[model]['average_consensus_entropy'] = entropy

            # Select prediction with minimum entropy
            min_entropy_model = min(model_entropies.items(), key=lambda x: x[1])[0]

            # Build final entry
            final_item = base_item.copy()
            final_item['model_predictions'] = model_predictions
            final_item['overall_consensus_entropy'] = overall_entropy
            final_item['prediction'] = model_predictions[min_entropy_model]['prediction']

            new_data.append(final_item)

        # Write output files
        write_excel(f"{output_base}.xlsx", new_data)
        with open(f"{output_base}.json", 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)

        print(f"Aggregation completed! Output files:")
        print(f" - Excel: {output_base}.xlsx")
        print(f" - JSON: {output_base}.json")

    except Exception as e:
        print(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()