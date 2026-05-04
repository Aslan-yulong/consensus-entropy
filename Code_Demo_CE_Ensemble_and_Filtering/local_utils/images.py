import io
import base64
import os
from PIL import Image

def compress_image_if_needed(image_path: str, max_size_mb = 0.75) -> str:
    """
    Compress image only if it exceeds max_size_mb (default 4MB)
    Returns base64 encoded string
    """
    # First check original file size
    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    
    # If file is smaller than max_size_mb, return original file as base64
    if file_size_mb <= max_size_mb:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    # If file is larger than max_size_mb, compress it
    img = Image.open(image_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Initial quality
    quality = 95
    max_size_bytes = max_size_mb * 1024 * 1024
    
    while True:
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        if len(buffer.getvalue()) <= max_size_bytes or quality <= 5:
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        quality -= 5