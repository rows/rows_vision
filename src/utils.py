import json
import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from text that may contain additional content.
    
    This function searches for JSON objects within text responses from AI models
    that might include explanations or markdown formatting around the actual JSON.
    
    Args:
        text: Input text that may contain a JSON object
        
    Returns:
        Dictionary containing the extracted JSON object, or error information
        
    Examples:
        >>> extract_json_from_text('Here is the data: {"key": "value"}')
        {"key": "value"}
        >>> extract_json_from_text('```json\n{"data": [1,2,3]}\n```')
        {"data": [1,2,3]}
    """
    if not text or not isinstance(text, str):
        logger.warning("Empty or non-string input provided to extract_json_from_text")
        return {"error": "Empty or invalid input text"}
    
    # Remove common markdown code block markers
    text_cleaned = text.strip()
    if text_cleaned.startswith('```json'):
        text_cleaned = text_cleaned[7:]  # Remove ```json
    if text_cleaned.endswith('```'):
        text_cleaned = text_cleaned[:-3]  # Remove ```
    text_cleaned = text_cleaned.strip()
    
    # Try to find JSON object boundaries
    start = text_cleaned.find('{')
    end = text_cleaned.rfind('}') + 1
    
    if start == -1 or end == 0:
        logger.warning("No JSON object boundaries found in text")
        return {"error": "No JSON found in response"}
    
    json_candidate = text_cleaned[start:end]
    
    # Try to parse the JSON
    try:
        parsed_json = json.loads(json_candidate)
        logger.debug("Successfully extracted JSON from text")
        return parsed_json
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {str(e)}")
        
        # Try to fix common JSON issues
        try:
            # Attempt to fix trailing commas and other common issues
            fixed_json = _fix_common_json_issues(json_candidate)
            parsed_json = json.loads(fixed_json)
            logger.debug("Successfully extracted JSON after fixing common issues")
            return parsed_json
        except json.JSONDecodeError:
            logger.error("Could not parse JSON even after attempting fixes")
            return {"error": f"Could not parse JSON from response: {str(e)}"}

def _fix_common_json_issues(json_str: str) -> str:
    """
    Attempt to fix common JSON formatting issues.
    
    Args:
        json_str: JSON string that may have formatting issues
        
    Returns:
        Potentially fixed JSON string
    """
    # Remove trailing commas before closing brackets/braces
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    # Fix single quotes to double quotes (basic attempt)
    json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
    
    # Remove any non-printable characters
    json_str = ''.join(char for char in json_str if char.isprintable() or char.isspace())
    
    return json_str

def validate_extracted_data(data: Dict[str, Any]) -> bool:
    """
    Validate that extracted data has the expected structure.
    
    Args:
        data: Dictionary containing extracted chart/table data
        
    Returns:
        True if data structure is valid, False otherwise
    """
    if not isinstance(data, dict):
        return False
    
    # Check if it's an error response
    if 'error' in data:
        return False
    
    # For chart data, we expect at least one chart with basic structure
    for chart_key, chart_data in data.items():
        if not isinstance(chart_data, dict):
            continue
            
        # Basic validation for chart structure
        if 'dataPoints' in chart_data:
            data_points = chart_data['dataPoints']
            if isinstance(data_points, list) and len(data_points) > 0:
                return True
                
        # Check for direct data structure (from classification)
        if any(key in chart_data for key in ['xAxis', 'yAxis']):
            return True
    
    return False

def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename to remove potentially dangerous characters.
    
    Args:
        filename: Original filename
        max_length: Maximum allowed length for filename
        
    Returns:
        Sanitized filename safe for file system use
    """
    if not filename:
        return "unnamed_file"
    
    # Remove path components
    filename = filename.split('/')[-1].split('\\')[-1]
    
    # Remove dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Limit length
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_length = max_length - len(ext) - 1 if ext else max_length
        filename = name[:max_name_length] + ('.' + ext if ext else '')
    
    return filename or "unnamed_file"

def log_processing_metrics(func_name: str, duration: float, success: bool, **kwargs):
    """
    Log processing metrics for monitoring and debugging.
    
    Args:
        func_name: Name of the function being measured
        duration: Time taken in seconds
        success: Whether the operation was successful
        **kwargs: Additional metrics to log
    """
    status = "SUCCESS" if success else "FAILED"
    metrics = {
        'function': func_name,
        'duration_seconds': round(duration, 3),
        'status': status,
        **kwargs
    }
    
    logger.info(f"Processing metrics: {json.dumps(metrics)}")
    
    if duration > 10:  # Log warning for slow operations
        logger.warning(f"Slow operation detected: {func_name} took {duration:.2f} seconds")