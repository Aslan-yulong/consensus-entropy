import pandas as pd
import Levenshtein
import os
import argparse

def calculate_difference(a, b):
    """Calculate normalized edit distance between two strings"""
    str_a, str_b = str(a), str(b)
    max_len = max(len(str_a), len(str_b))
    return Levenshtein.distance(str_a, str_b) / max_len if max_len > 0 else 0.0

def calculate_consensus_entropy(predictions):
    """Calculate Consensus Entropy (CE)"""
    n = len(predictions)
    if n < 2:
        return 0.0
    
    total_diff = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i+1, n):
            total_diff += calculate_difference(predictions[i], predictions[j])
            pair_count += 1
    
    return total_diff / pair_count if pair_count > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description='Calculate Consensus Entropy and add CE column')
    parser.add_argument('test_model', help='Path to test model file')
    parser.add_argument('reference_models', nargs='+', help='Paths to reference model files')
    args = parser.parse_args()

    # Read test model data
    test_df = pd.read_excel(args.test_model)
    if 'prediction' not in test_df.columns:
        raise ValueError("Test model file missing 'prediction' column")

    # Read reference model predictions
    ref_predictions = []
    for ref_path in args.reference_models:
        ref_df = pd.read_excel(ref_path)
        if 'prediction' not in ref_df.columns:
            raise ValueError(f"{os.path.basename(ref_path)} missing 'prediction' column")
        if len(ref_df) != len(test_df):
            raise ValueError(f"{os.path.basename(ref_path)} row count mismatch with test model")
        ref_predictions.append(ref_df['prediction'].tolist())

    # Calculate CE values for each row
    ce_values = []
    for idx in range(len(test_df)):
        row_predictions = [test_df.at[idx, 'prediction']]
        for ref in ref_predictions:
            row_predictions.append(ref[idx])
        ce_values.append(calculate_consensus_entropy(row_predictions))

    # Add CE column and save results
    test_df['CE'] = ce_values
    base_name = os.path.splitext(os.path.basename(args.test_model))[0]
    ref_names = [os.path.splitext(os.path.basename(f))[0] for f in args.reference_models]
    output_path = f"{base_name}_ce_from_{'_'.join(ref_names)}.xlsx"
    test_df.to_excel(output_path, index=False)
    total_ce = test_df['CE'].sum()
    print("CE TOTAL: ", total_ce)
    print(f"Results saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()