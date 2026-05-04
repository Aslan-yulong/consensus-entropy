#!/usr/bin/env python3
import os
import json
import base64
import argparse
from tqdm import tqdm
from pathlib import Path
import datasets
import io
from typing import Dict, Any, List, Optional, Tuple
import multiprocessing
from multiprocessing import Pool
from functools import partial
import hashlib
import re
import yaml  # Add yaml import for config loading
from prompt import generate_rephrase_prompt, generate_system_prompt
from diskcache import Cache  # Add diskcache import
from datetime import datetime
import sys
from local_utils.images import compress_image_if_needed  # Import image compression function
from PIL import Image
from time import sleep
import time

# Default configuration
DEFAULT_CONFIG = {
    # "api_keys": [
    #     "your-api-key-here",
    # ],
    # "api_base_url": "https://api.openai.com/v1/",
    # "model": "gpt-4o-2024-11-20",
    # "dataset_path": "/path/to/dataset/OCRBench",
    "similarity_threshold": 123132,
    # "results_dir": "/path/to/results",
    # "num_processes": 4,
    "cache_dir": "/path/to/cache",  # Add cache directory
    "use_cache": True,  # Add cache flag
    "resume": False  # Add resume flag
}

# Constants
BASE_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Function to load config from file
def load_config(config_path):
    """Load configuration from YAML or JSON file"""
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using default configuration")
        return DEFAULT_CONFIG
    
    try:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            print(f"Unsupported config file format. Using default configuration.")
            return DEFAULT_CONFIG
        
        # Merge with defaults for any missing keys
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value
        
        return config
    except Exception as e:
        print(f"Error loading config file: {e}. Using default configuration.")
        return DEFAULT_CONFIG

# Import Agent after defining constants
from agent import Agent

# Cache helpers
def get_cache_key(item: Dict[str, Any]) -> str:
    """Create a unique cache key for an item based on its content"""
    key_str = f"{item['image_path']}_{item['question']}_{json.dumps(item.get('model_predict', {}))}"
    return hashlib.md5(key_str.encode()).hexdigest()

def init_cache(cache_dir: str) -> Optional[Cache]:
    """Initialize the disk cache"""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    return Cache(cache_dir)

# Utility functions
def ensure_dir(dir_path):
    """Ensure directory exists"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def ensure_file(file_path):
    """Ensure file exists"""
    if not Path(file_path).exists():
        with open(file_path, "w") as f:
            pass

def load_dataset_images(dataset_path=None):
    """
    Load the OCRBench dataset images for later retrieval.
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        dataset: The loaded dataset
    """
    if dataset_path is None:
        dataset_path = DEFAULT_CONFIG['dataset_path']
        
    print(f"Loading dataset from {dataset_path}...")
    try:
        data = datasets.load_dataset(dataset_path)
        return data['test']
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def extract_image_data(image_path, dataset_path=None):
    """
    Load image from path relative to dataset_path.
    
    Args:
        image_path: Path to the image file (can be relative)
        dataset_path: Base dataset path to prepend if image_path is relative
        
    Returns:
        PIL.Image or None: The loaded image or None if not found
    """
    try:
        # If dataset_path is provided and image_path doesn't start with / or C:\ etc.
        if dataset_path and not os.path.isabs(image_path):
            full_path = os.path.join(dataset_path, image_path)
        else:
            full_path = image_path
            
        if os.path.exists(full_path):
            with open(full_path,"rb") as image_file:
                image_file_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            return image_file_base64
        else:
            print(f"Image not found at path: {full_path}")
            return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def needs_rephrasing(model_predict: Dict[str, Any], threshold: float = DEFAULT_CONFIG['similarity_threshold']) -> bool:
    """
    Check if the model predictions need rephrasing based on similarity scores
    
    Args:
        model_predict: Dictionary of model predictions
        threshold: Similarity threshold, predictions below this need rephrasing
        
    Returns:
        bool: True if rephrasing is needed, False otherwise
    """
    # If model_predict is empty, no rephrasing needed
    if not model_predict:
        print("No model predictions found")
        return False
    
    # Check if all models have a similarity score below the threshold
    for model_name, pred_info in model_predict.items():
        if pred_info.get("average_similarity") >= threshold:
            return False
    
    return True

def process_item(
    item: Dict[str, Any],
    api_key: str,
    api_base_url: str,
    model: str,
    dataset_path: str = None,  # Add dataset_path parameter
    similarity_threshold: float = DEFAULT_CONFIG['similarity_threshold'],
    retry_count: int = 3,
    cache: Optional[Cache] = None,
    failed_ids_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single item from the prediction results.
    
    Args:
        item: The item to process
        api_key: API key for the LLM
        api_base_url: Base URL for API calls
        model: Model name/path to use
        similarity_threshold: Threshold for filtering predictions
        retry_count: Number of retries
        cache: Optional disk cache instance
        failed_ids_file: Optional file to record failed IDs
        
    Returns:
        Dict: The processed item with rephrased answer
    """
    # Create a copy of the item
    result = item.copy()
    
    # Check cache first if available
    if cache is not None:
        cache_key = get_cache_key(item)
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            print(f"Cache hit for ID {item.get('id')}")
            return cached_result
    
    # Check if the item needs rephrasing
    model_predict = item.get('model_predict', {})
    if not needs_rephrasing(model_predict, similarity_threshold):
        # If no rephrasing needed, return the prediction with highest similarity score
        best_model_name = max(model_predict.keys(), 
                             key=lambda model_name: model_predict[model_name].get('average_similarity', 0))
        result['predict'] = model_predict[best_model_name]['predict']
        result['is_rephrased'] = False
        
        # Cache the result if no rephrasing needed
        if cache is not None:
            cache_key = get_cache_key(item)
            cache.set(cache_key, result)
        return result
    result['is_rephrased'] = True
    
    # Load image using dataset_path
    image_path = item.get('image_path')
    image_data = extract_image_data(image_path, dataset_path)
    
    if image_data is None:
        print(f"Warning: Image not found at path {image_path}")
        result['predict'] = "ERROR: Image not found"
        return result
    
    # Convert image to base64
    # byte_io = io.BytesIO()
    # image_data.save(byte_io, format='JPEG')
    # byte_io.seek(0)
    base64_image = image_data
    
    # Create agent
    agent = Agent(
        api_keys=api_key,
        base_url=api_base_url,
        model=model,
        request_kwargs={
            "temperature": 0.0,
            "max_tokens": 2048
        }
    )
    
    # Generate prompt
    prompt = generate_rephrase_prompt(
        image_path=image_path,
        question=item['question'],
        model_predictions=model_predict,
    )
    
    # Prepare messages
    messages = [{
        "role": "system",
        "content": generate_system_prompt()
    }, {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    }]
    
    # Get LLM response
    try:
        content = agent.chat_completion(messages, stream=False, ttl=retry_count)
        if content is None:
            raise Exception("Failed to get response from API")
        
        content = content.strip()
        
        # Add rephrased answer to result
        result['predict'] = content
        
        # Cache the result
        if cache is not None:
            cache_key = get_cache_key(item)
            cache.set(cache_key, result)
            
        return result
    except Exception as e:
        error_msg = str(e)
        
        # Check if it's a 413 error (Request Entity Too Large)
        if "413" in error_msg and "Request Entity Too Large" in error_msg:
            print(f"Detected 413 error for ID {item.get('id')}, attempting with compressed image...")
            
            # Save image to temporary file for compression
            temp_image_path = f"./tmp/temp_image_{item.get('id')}.jpg"
            os.makedirs("./tmp", exist_ok=True)
            try:
                image_data.save(temp_image_path, format='JPEG')
                
                # Gradually reduce image quality until successful
                for max_size in [0.75, 0.5, 0.25]:
                    try:
                        print(f"Trying compression with max_size={max_size}MB for ID {item.get('id')}")
                        base64_image = compress_image_if_needed(temp_image_path, max_size_mb=max_size)
                        
                        # Update image in messages
                        messages[1]['content'][1]['image_url']['url'] = f"data:image/jpeg;base64,{base64_image}"
                        
                        # Retry API call
                        content = agent.chat_completion(messages, stream=False, ttl=retry_count)
                        if content is None:
                            raise Exception("Failed to get response from API")
                        
                        content = content.strip()
                        
                        # Successfully got response, update result
                        result['predict'] = content
                        
                        # Cache result
                        if cache is not None:
                            cache_key = get_cache_key(item)
                            cache.set(cache_key, result)
                        
                        print(f"Successfully processed ID {item.get('id')} with compressed image (max_size={max_size}MB)")
                        return result
                    except Exception as compress_error:
                        if "413" not in str(compress_error) or max_size == 0.25:
                            # If not a 413 error or already tried the smallest compression size, break loop
                            error_msg = str(compress_error)
                            break
                        # Otherwise continue trying smaller compression sizes
                        print(f"Still getting 413 error with max_size={max_size}MB for ID {item.get('id')}, trying smaller size...")
                        continue
            except Exception as save_error:
                error_msg = f"Error saving/compressing image: {save_error}"
            finally:
                # Delete temporary file
                if os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except:
                        pass
        
        # Handle all other errors or cases where compression still fails
        print(f"Error processing {item['image_path']}: {error_msg}")
        
        # Record IDs that still fail after retries are exhausted
        if failed_ids_file:
            with open(failed_ids_file, 'a', encoding='utf-8') as f:
                failed_item = {
                    'id': item.get('id'),
                    'dataset_name': item.get('dataset_name', ''),
                    'image_path': item.get('image_path', ''),
                    'question': item.get('question', ''),
                    'error': error_msg
                }
                f.write(json.dumps(failed_item, ensure_ascii=False) + '\n')
        
        result['predict'] = f"ERROR: {error_msg}"
        return result

def process_worker_args(args):
    """Helper function to pass to pool.imap"""
    index, item, config = args
    api_keys = config['api_keys']
    output_file = config['output_file']
    similarity_threshold = config['similarity_threshold']
    dataset_path = config['dataset_path']  # Get dataset_path from config
    api_key = api_keys[index % len(api_keys)]
    
    result = process_item(
        item, 
        api_key, 
        config['api_base_url'],
        config['model'],
        dataset_path,  # Pass dataset_path
        similarity_threshold,
        cache=config.get('cache'),
        failed_ids_file=config.get('failed_ids_file')
    )
    
    # Write result to output file atomically
    if result:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    return result

def process_items_batch(
    items: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Process items using a process pool"""
    num_processes = config['num_processes']
    
    args_list = [
        (i, item, config) 
        for i, item in enumerate(items)
    ]
    
    results = []
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_worker_args, args_list),
            total=len(items),
            desc="Processing Items"
        ))
    
    return [r for r in results if r is not None]

def load_processed_ids(output_file: str) -> set:
    """Load IDs of already processed items"""
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            if output_file.endswith(".jsonl"):
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        # Create a unique ID from dataset, type and item ID
                        unique_id = f"{result.get('dataset_name', '')}_{result.get('type', '')}_{result.get('id', '')}"
                        processed_ids.add(unique_id)
                    except json.JSONDecodeError:
                        continue
            elif output_file.endswith(".json"):
                try:
                    results = json.load(f)
                    for result in results:
                        # Create a unique ID from dataset, type and item ID
                        unique_id = f"{result.get('dataset_name', '')}_{result.get('type', '')}_{result.get('id', '')}"
                        processed_ids.add(unique_id)
                except json.JSONDecodeError:
                    pass
    return processed_ids

def main():
    parser = argparse.ArgumentParser(description="Rephrase answers from prediction results")
    parser.add_argument("--config", type=str, default="/path/to/config/config_gpt4o.yaml", help="Path to configuration file")
    parser.add_argument("--input_file", type=str, default="/path/to/prediction_results.json", help="Input JSON file with prediction results")
    parser.add_argument("--output_file", type=str, help="Output file for rephrased results")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching")
    parser.add_argument("--resume", action="store_true", help="Resume processing from last state by loading existing processed IDs")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.input_file:
        config['input_file'] = args.input_file
    if args.output_file:
        config['output_file'] = args.output_file
    if args.resume:
        config['resume'] = True
    
    # Create date-based subdirectory in results - include hours and minutes
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
    date_results_dir = os.path.join(config['results_dir'], current_datetime)
    ensure_dir(date_results_dir)
    
    # Set up output files and logging first
    if 'output_file' not in config:
        input_basename = os.path.basename(config['input_file'])
        input_root, input_ext = os.path.splitext(input_basename)
        
        incremental_output_file = os.path.join(date_results_dir, f"{input_root}_rephrased.jsonl")
        final_output_file = os.path.join(date_results_dir, f"{input_root}_rephrased{input_ext}")
        log_file = os.path.join(date_results_dir, f"{input_root}_log.txt")
        
        config['output_file'] = incremental_output_file
        config['final_output_file'] = final_output_file
        config['log_file'] = log_file
    else:
        config['final_output_file'] = config['output_file']
        config['log_file'] = os.path.join(
            date_results_dir,
            f"{os.path.splitext(os.path.basename(config['output_file']))[0]}_log.txt"
        )
    
    # Set up failed ID recording file
    config['failed_ids_file'] = os.path.join(
        date_results_dir,
        f"{os.path.splitext(os.path.basename(config['input_file']))[0]}_failed_ids.jsonl"
    )
    
    # Ensure output directory exists
    ensure_dir(os.path.dirname(config['output_file']))
    ensure_file(config['output_file'])
    ensure_file(config['log_file'])
    ensure_file(config['failed_ids_file'])  # Ensure failed ID file exists
    
    # Set up logging immediately after creating log file
    class TeeOutput:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'a', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = TeeOutput(config['log_file'])
    
    # Now start the actual processing
    print(f"Starting processing at {current_datetime}")
    print(f"Input file: {config['input_file']}")
    print(f"Output directory: {date_results_dir}")
    
    # Initialize cache if enabled
    if not args.no_cache and config.get('use_cache', True):
        config['cache'] = init_cache(config.get('cache_dir', DEFAULT_CONFIG['cache_dir']))
    else:
        config['cache'] = None
    
    # Load input data
    print("Loading input data...")
    with open(config['input_file'], "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filter out already processed items
    if config['resume']:
        processed_ids = load_processed_ids(config['output_file'])
        remaining_items = []
        for item in data:
            unique_id = f"{item.get('dataset_name', '')}_{item.get('type', '')}_{item.get('id', '')}"
            if unique_id not in processed_ids:
                remaining_items.append(item)
    else:
        remaining_items = data
    
    print(f"Found {len(data)} items in input file")
    print(f"Remaining {len(remaining_items)} items to process")
    print(f"Using similarity threshold: {config['similarity_threshold']}")
    print(f"Using model: {config['model']}")
    print(f"Using API base URL: {config['api_base_url']}")
    
    # Process items in batch, but no need to pass benchset anymore
    processed_results = process_items_batch(
        remaining_items,
        config
    )
    
    # Create a full result set by combining new results with any existing processed items
    full_results = []
    
    # Map for quick lookup of processed items
    processed_items_map = {
        f"{item.get('dataset_name', '')}_{item.get('type', '')}_{item.get('id', '')}": item 
        for item in processed_results
    }
    
    # Add all items with rephrased predictions where available
    for item in data:
        item_id = f"{item.get('dataset_name', '')}_{item.get('type', '')}_{item.get('id', '')}"
        if item_id in processed_items_map:
            # Use the newly processed item
            full_results.append(processed_items_map[item_id])
        elif item_id in processed_ids:
            # Item was processed before but not in this run, read from the output file
            found = False
            with open(config['output_file'], "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        result_id = f"{result.get('dataset_name', '')}_{result.get('type', '')}_{result.get('id', '')}"
                        if result_id == item_id:
                            full_results.append(result)
                            found = True
                            break
                    except json.JSONDecodeError:
                        continue
            
            # If not found, add the original item
            if not found:
                full_results.append(item)
        else:
            # Item was not processed, add the original
            full_results.append(item)
    
    # Write the final combined file
    with open(config['final_output_file'], "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    
    print(f"Processing completed. Results saved to:")
    print(f"- Incremental JSONL: {config['output_file']}")
    print(f"- Combined JSON: {config['final_output_file']}")
    
    # Close the cache properly
    if config['cache'] is not None:
        config['cache'].close()

if __name__ == "__main__":
    main()
