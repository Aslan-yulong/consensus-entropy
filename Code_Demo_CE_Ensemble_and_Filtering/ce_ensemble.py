import json
import glob
from Levenshtein import distance as Levenshtein_distance
import numpy as np
from tqdm import tqdm

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def calculate_difference(a, b):
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return Levenshtein_distance(a, b) / max_len

def calculate_consensus_entropy(predict_list):
    n = len(predict_list)
    diff_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            diff_matrix[i][j] = calculate_difference(str(predict_list[i]), str(predict_list[j]))
    
    np.fill_diagonal(diff_matrix, 0)
    model_entropies = np.mean(diff_matrix, axis=1)
    
    counts = {}
    for pred in predict_list:
        pred_str = str(pred)
        counts[pred_str] = counts.get(pred_str, 0) + 1
    
    max_count = max(counts.values(), default=0)
    candidates = [pred for pred, count in counts.items() if count == max_count]
    consensus_pred = candidates[0] if candidates else ''
    
    total_diff = 0.0
    for pred in predict_list:
        total_diff += calculate_difference(consensus_pred, str(pred))
    overall_entropy = total_diff / n
    
    return model_entropies, overall_entropy

# Initialize statistical variables
file_list = ['internvl2_5_26b.json','gpt4o.json','gemini_pro.json']
json_files = [f for f in file_list]
all_data = [read_json(file) for file in json_files]
a_data = all_data[0]

num_models = len(json_files)
model_entropy_sums = [0.0] * num_models
model_counts = [0] * num_models

for i in tqdm(range(len(a_data))):
    predicts = []
    scores = []
    valid_models = []
    
    # Collect valid predictions and model indices
    for model_idx, data in enumerate(all_data):
        if i < len(data):
            predicts.append(data[i]['predict'])
            scores.append(data[i]['score'])
            valid_models.append(model_idx)
    
    # Calculate entropy values
    model_entropies, overall_entropy = calculate_consensus_entropy(predicts)
    
    # Convert to dictionary list format
    model_entropies_dicts = []
    for idx, model_idx in enumerate(valid_models):
        model_name = json_files[model_idx]  # Get model name
        entropy_value = model_entropies[idx]
        model_entropies_dicts.append({model_name: entropy_value})
        
        # Update statistics (maintain index order)
        model_entropy_sums[model_idx] += entropy_value
        model_counts[model_idx] += 1
    
    # Update data
    a_data[i]['predict'] = predicts[np.argmin(model_entropies)]
    a_data[i]['score'] = scores[np.argmin(model_entropies)]
    a_data[i]['model_entropies'] = model_entropies_dicts  # New format
    a_data[i]['overall_entropy'] = overall_entropy

# Calculate final statistics (maintain original logic)
average_model_entropies = [
    model_entropy_sums[i]/model_counts[i] 
    for i in range(num_models) 
    if model_counts[i] > 0
]

# Print statistics (adapted to new format)
print("\nAverage Consensus Entropy per Model:")
for model_idx, avg_entropy in enumerate(average_model_entropies):
    print(f"{json_files[model_idx]}: {avg_entropy:.4f}")

out_dir = './ce_internvl2_5_26b+gpt4o.json+gemini_pro.json_ensemble.json'
write_json(out_dir, a_data)
print(f"\nProcessing completed. Results saved to: {out_dir}")