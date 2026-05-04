import os
import argparse
import pandas as pd
import numpy as np
import Levenshtein
import json

def calculate_difference(a, b):
    """Calculate the normalized edit distance between two strings"""
    max_len = max(len(a), len(b))
    return Levenshtein.distance(a, b) / max_len if max_len != 0 else 0.0

def calculate_consensus_entropy(predictions):
    """Calculate the consensus entropy matrix"""
    n = len(predictions)
    if n < 2:
        return np.zeros(n)
    
    diff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                diff_matrix[i][j] = calculate_difference(str(predictions[i]), str(predictions[j]))
    
    return np.sum(diff_matrix, axis=1) / (n - 1)

def process_files(file_paths, output_json="results.json"):
    """Main processing function"""
    # 1. Read and validate files
    models = {}
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.endswith(".xlsx"):
            raise ValueError(f"Invalid file type: {file_path}")
        
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        models[model_name] = pd.read_excel(file_path)

    # 2. Prepare data structure
    model_names = list(models.keys())
    model_data = {name: {"entropy": [], "score": []} for name in model_names}
    dfs = list(models.values())
    num_rows = len(dfs[0])

    # 3. Calculate entropy for each row
    for i in range(num_rows):
        predictions = [df.iloc[i]["prediction"] for df in dfs]
        scores = [df.iloc[i]["score"] for df in dfs]
        entropies = calculate_consensus_entropy(predictions)

        for j, name in enumerate(model_names):
            model_data[name]["entropy"].append(entropies[j])
            model_data[name]["score"].append(scores[j])

    # 4. Calculate threshold results
    thresholds = [round(1.0 - 0.1*i, 1) for i in range(10)]
    results = {}

    for name in model_names:
        entropy = model_data[name]["entropy"]
        scores = model_data[name]["score"]
        model_results = []

        for threshold in thresholds:
            valid_scores = [s for e, s in zip(entropy, scores) if e <= threshold]
            accuracy = np.mean(valid_scores) if valid_scores else 0.0
            model_results.append({
                "threshold": threshold,
                "accuracy": float(accuracy)
            })

        results[name] = model_results

    # 5. Print results
    print("Results:")
    for model, data in results.items():
        print(f"\nModel: {model}")
        for item in data:
            print(f"Threshold: {item['threshold']:.1f} => Accuracy: {item['accuracy']:.4f}")

    # 6. Save results
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Model consensus analysis with direct file inputs',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'files', 
        nargs='+',
        help='List of XLSX file paths to process\n'
             'Example: python cal_avg_scores_CE_thresholds.py path/to/model1.xlsx path/to/model2.xlsx'
    )
    parser.add_argument(
        '-o', 
        '--output', 
        default='results.json',
        help='Output JSON file path (default: results.json)'
    )

    args = parser.parse_args()

    try:
        process_files(
            file_paths=args.files,
            output_json=args.output
        )
        print(f"\nAnalysis completed. Results saved to {args.output}")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")