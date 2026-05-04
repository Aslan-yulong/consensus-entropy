def generate_rephrase_prompt(image_path: str, question: str, model_predictions: dict) -> str:
    """
    Generate a prompt for rephrasing the answer based on the image, question and model predictions.
    
    Args:
        image_path: Path to the image
        question: The question asked
        model_predictions: Dictionary of model predictions with their similarity scores
        answers: List of possible answers (can be strings or dictionaries)
        
    Returns:
        str: A prompt for rephrasing
    """
    # Create a combined string of model predictions
    predictions_str = ""
    for model_name, pred_info in model_predictions.items():
        predictions_str += f"- {model_name}: {pred_info['predict']}\n"
    
    return f"""
You are an expert AI assistant tasked with improving answers to visual questions. 
Please look at the image and examine the following question and the current answers from different models.

Question: {question}

Current model predictions:
{predictions_str}

Your task is to synthesize these predictions and create a single improved answer that:
1. Is more accurate based on the visual content
2. Is concise and direct
3. Uses a natural, conversational tone
4. Maintains the core meaning of the original answers if they were correct
5. Improves clarity and precision

Do not invent details not present in the image. Your answer should be grounded in what is actually visible.

Please provide ONLY the improved answer with no explanations or additional text.
"""

def generate_system_prompt() -> str:
    """
    Generate a system prompt for the LLM.
    
    Returns:
        str: System prompt
    """
    return """
You are an assistant specialized in interpreting visual content and improving answers to questions about images.
Your goal is to provide clear, accurate, and concise responses.
Focus on factual information visible in the image and avoid adding speculation or details not present.
"""
