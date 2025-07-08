from flask import Flask, request, jsonify, Response
import os
import sys
import logging
from dotenv import load_dotenv
from src.image_classifier import ImageClassifier
from src.image_analyzer import ImageAnalyzer
from src.rows_vision import RowsVision
from time import time
from io import BytesIO
from src.logging_config import setup_logging
from src.config import AppConfig
import json


load_dotenv()
setup_logging()

app = Flask(__name__)
logger = logging.getLogger(__name__)
app.config['JSON_AS_ASCII'] = False
# Validate required environment variables
required_env_vars = ['API_KEY_ANTHROPIC', 'API_KEY_OPENAI', 'API_KEY_GEMINI', 'API_KEY_GROQ']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {missing_vars}")
    sys.exit(1)

# Get API keys
api_key = os.getenv('API_KEY_ANTHROPIC')
api_key_openai = os.getenv('API_KEY_OPENAI')
api_key_gemini = os.getenv('API_KEY_GEMINI')
api_key_groq = os.getenv('API_KEY_GROQ')

# Initialize components
classifier = ImageClassifier(api_key, api_key_openai, api_key_gemini, api_key_groq)
analyzer = ImageAnalyzer(api_key, api_key_openai, api_key_gemini, api_key_groq)
rows_vision = RowsVision(classifier, analyzer)

config = AppConfig()

def _format_instructions_result(result, include_name=False):
    """
    Format the result from classify_with_instructions to return only data_points array
    or optionally include name if requested.
    
    Args:
        result: Raw result from classifier
        include_name: Whether to include the name in the response
    
    Returns:
        Formatted result according to new output requirements:
        - Default: data_points array only
        - With include_name=True: {"name": "...", "data_points": [...]}
    """
    logger.info(f'Formatting result: {type(result)} - {result}')
    
    # Handle error cases
    if isinstance(result, dict) and 'error' in result:
        return result
    
    # Handle dict result (single chart/table)
    if isinstance(result, dict):
        logger.info(f'Result is dict: {result}')
        
        # Check if it has the expected structure with data_points
        if 'data_points' in result:
            data_points = result['data_points']
            
            if include_name and 'name' in result:
                logger.info('Including name in response')
                # Return only name and data_points, exclude type, sampled_axis, has_data_labels
                return {
                    "name": result['name'],
                    "data_points": data_points
                }
            else:
                logger.info('Returning only data_points')
                # Return only data_points array
                return data_points
        else:
            # If it's a dict but doesn't have data_points, return as-is
            logger.info('Dict without data_points, returning as-is')
            return result
    
    # Handle list of chart results
    elif isinstance(result, list) and len(result) > 0:
        # Get the first chart/table result
        chart_data = result[0]
        logger.info(f'Chart data: {type(chart_data)} - {chart_data}')
        
        # Check if it's a dict with the expected structure
        if isinstance(chart_data, dict) and 'data_points' in chart_data:
            data_points = chart_data['data_points']
            
            if include_name and 'name' in chart_data:
                logger.info('Including name in response')
                # Return only name and data_points, exclude type, sampled_axis, has_data_labels
                return {
                    "name": chart_data['name'],
                    "data_points": data_points
                }
            else:
                logger.info('Returning only data_points')
                # Return only data_points array
                return data_points
        else:
            # If it's not a dict with data_points, return the first element as-is
            logger.info('First element is not a dict with data_points, returning as-is')
            return chart_data
    
    # If result is already a simple array (data_points), return as-is
    if isinstance(result, list):
        logger.info('Result is already a list, returning as-is')
        return result
    
    # Fallback to original result if formatting fails
    logger.warning('Fallback: returning original result')
    return result

@app.route('/api/run', methods=['POST'])
def run_external_api():
    """
    Process image from URL using AI models for classification and data extraction.
    
    Expected JSON payload:
    {
        "image_url": "https://example.com/image.jpg",
        "model_classification": "anthropic",  # optional
        "model_extraction": "anthropic",     # optional
        "time_outputs": false               # optional
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Missing JSON payload'}), 400

        # Validate image_url
        image_url = data.get('image_url')
        if not image_url:
            return jsonify({'error': 'Missing image_url parameter'}), 400
        
        if not image_url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid image URL - must start with http:// or https://'}), 400

        # Get optional parameters with defaults
        model_classification = data.get('model_classification', 'anthropic')
        model_extraction = data.get('model_extraction', 'anthropic')
        time_outputs = data.get('time_outputs', False)

        # Validate model names
        valid_models = ['anthropic', 'openai', 'google', 'groq']
        if model_classification not in valid_models:
            return jsonify({'error': f'Invalid model_classification. Must be one of: {valid_models}'}), 400
        if model_extraction not in valid_models:
            return jsonify({'error': f'Invalid model_extraction. Must be one of: {valid_models}'}), 400

        try:
            file_extension, filename, file_stream = rows_vision.download_image_from_url(image_url)
        except Exception as e:
            logger.error(f"Failed to download image from {image_url}: {str(e)}")
            return jsonify({'error': f'Failed to download image: {str(e)}'}), 400

        if time_outputs:
            start_time = time()
            raw_result = rows_vision.run_image_json(file_extension, filename, file_stream, model_classification, model_extraction)
            total_time = round(time() - start_time, 3)
            
            # Convert dict format back to data_points format for consistency
            if isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], dict):
                # Convert from dict format back to data_points array format
                headers = list(raw_result[0].keys())
                data_points = [headers]
                for row_dict in raw_result:
                    row = [row_dict.get(header, None) for header in headers]
                    data_points.append(row)
                formatted_result = data_points
            else:
                formatted_result = raw_result
            
            response_data = {
                "result": formatted_result,
                "metrics": {
                    "total_time": total_time
                }
            }
            
            # Use Response with ensure_ascii=False
            return Response(
                response=json.dumps(response_data, ensure_ascii=False),
                status=200,
                mimetype='application/json; charset=utf-8'
            )
        else:
            # Get raw result and format it like the instructions endpoint
            raw_result = rows_vision.run_image_json(file_extension, filename, file_stream, model_classification, model_extraction)
            
            # Convert dict format back to data_points format for consistency
            if isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], dict):
                # Convert from dict format back to data_points array format
                headers = list(raw_result[0].keys())
                data_points = [headers]
                for row_dict in raw_result:
                    row = [row_dict.get(header, None) for header in headers]
                    data_points.append(row)
                formatted_result = data_points
            else:
                formatted_result = raw_result
            
            response_data = {"result": formatted_result}
            
            # Use Response with ensure_ascii=False
            return Response(
                response=json.dumps(response_data, ensure_ascii=False),
                status=200,
                mimetype='application/json; charset=utf-8'
            )

    except Exception as e:
        logger.error(f"Unexpected error in run_external_api: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/run-file', methods=['POST'])
def run_external_api_file():
    """
    Process image from URL or local file path using AI models.
    
    Expected JSON payload:
    {
        "image_url": "https://example.com/image.jpg",  # OR
        "file_path": "/path/to/local/image.jpg",       # OR
        "model_classification": "anthropic",           # optional
        "model_extraction": "anthropic",              # optional
        "time_outputs": false                         # optional
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Missing JSON payload'}), 400

        # Get optional parameters with defaults
        model_classification = data.get('model_classification', 'anthropic')
        model_extraction = data.get('model_extraction', 'anthropic')
        time_outputs = data.get('time_outputs', False)

        # Validate model names
        valid_models = ['anthropic', 'openai', 'google', 'groq']
        if model_classification not in valid_models:
            return jsonify({'error': f'Invalid model_classification. Must be one of: {valid_models}'}), 400
        if model_extraction not in valid_models:
            return jsonify({'error': f'Invalid model_extraction. Must be one of: {valid_models}'}), 400

        image_url = data.get('image_url')
        file_path = data.get('file_path')

        if not image_url and not file_path:
            return jsonify({'error': 'Missing image_url or file_path parameter'}), 400

        if image_url and file_path:
            return jsonify({'error': 'Provide either image_url or file_path, not both'}), 400

        try:
            if image_url:
                if not image_url.startswith(('http://', 'https://')):
                    return jsonify({'error': 'Invalid image URL - must start with http:// or https://'}), 400
                file_extension, filename, file_stream = rows_vision.download_image_from_url(image_url)
            elif file_path:
                if not os.path.exists(file_path):
                    return jsonify({'error': f'File not found: {file_path}'}), 400
                file_extension = os.path.splitext(file_path)[1].lstrip(".")
                filename = os.path.basename(file_path)
                with open(file_path, "rb") as f:
                    file_stream = BytesIO(f.read())
        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}")
            return jsonify({'error': f'Failed to load image: {str(e)}'}), 400

        if time_outputs:
            start_time = time()
            raw_result = rows_vision.run_image_json(file_extension, filename, file_stream, model_classification, model_extraction)
            total_time = round(time() - start_time, 3)
            
            # Convert dict format back to data_points format for consistency
            if isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], dict):
                # Convert from dict format back to data_points array format
                headers = list(raw_result[0].keys())
                data_points = [headers]
                for row_dict in raw_result:
                    row = [row_dict.get(header, None) for header in headers]
                    data_points.append(row)
                formatted_result = data_points
            else:
                formatted_result = raw_result
            
            response_data = {
                "result": formatted_result,
                "metrics": {
                    "total_time": total_time
                }
            }
            
            # Use Response with ensure_ascii=False
            return Response(
                response=json.dumps(response_data, ensure_ascii=False),
                status=200,
                mimetype='application/json; charset=utf-8'
            )
        else:
            # Get raw result and format it like the instructions endpoint
            raw_result = rows_vision.run_image_json(file_extension, filename, file_stream, model_classification, model_extraction)
            
            # Convert dict format back to data_points format for consistency
            if isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], dict):
                # Convert from dict format back to data_points array format
                headers = list(raw_result[0].keys())
                data_points = [headers]
                for row_dict in raw_result:
                    row = [row_dict.get(header, None) for header in headers]
                    data_points.append(row)
                formatted_result = data_points
            else:
                formatted_result = raw_result
            
            response_data = {"result": formatted_result}
            
            # Use Response with ensure_ascii=False
            return Response(
                response=json.dumps(response_data, ensure_ascii=False),
                status=200,
                mimetype='application/json; charset=utf-8'
            )

    except Exception as e:
        logger.error(f"Unexpected error in run_external_api_file: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/run-one-shot', methods=['POST'])
def run_external_api_one_shot():
    """
    Process image from URL or local file path using AI models.
    
    Expected JSON payload:
    {
        "image_url": "https://example.com/image.jpg",  # OR
        "file_path": "/path/to/local/image.jpg",       # OR
        "model_classification": "anthropic",           # optional
        "model_extraction": "anthropic",              # optional
        "time_outputs": false                         # optional
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Missing JSON payload'}), 400

        # Get optional parameters with defaults
        model_classification = data.get('model_classification', 'anthropic')
        model_extraction = data.get('model_extraction', 'anthropic')
        time_outputs = data.get('time_outputs', False)

        # Validate model names
        valid_models = ['anthropic', 'openai', 'google', 'groq']
        if model_classification not in valid_models:
            return jsonify({'error': f'Invalid model_classification. Must be one of: {valid_models}'}), 400
        if model_extraction not in valid_models:
            return jsonify({'error': f'Invalid model_extraction. Must be one of: {valid_models}'}), 400

        image_url = data.get('image_url')
        file_path = data.get('file_path')

        if not image_url and not file_path:
            return jsonify({'error': 'Missing image_url or file_path parameter'}), 400

        if image_url and file_path:
            return jsonify({'error': 'Provide either image_url or file_path, not both'}), 400

        try:
            if image_url:
                if not image_url.startswith(('http://', 'https://')):
                    return jsonify({'error': 'Invalid image URL - must start with http:// or https://'}), 400
                file_extension, filename, file_stream = rows_vision.download_image_from_url(image_url)
            elif file_path:
                if not os.path.exists(file_path):
                    return jsonify({'error': f'File not found: {file_path}'}), 400
                file_extension = os.path.splitext(file_path)[1].lstrip(".")
                filename = os.path.basename(file_path)
                with open(file_path, "rb") as f:
                    file_stream = BytesIO(f.read())
        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}")
            return jsonify({'error': f'Failed to load image: {str(e)}'}), 400

        if time_outputs:
            start_time = time()
            raw_result = rows_vision.run_image_json(file_extension, filename, file_stream, model_classification, model_extraction, skip_step=True)
            total_time = round(time() - start_time, 3)
            
            # Convert dict format back to data_points format for consistency
            if isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], dict):
                # Convert from dict format back to data_points array format
                headers = list(raw_result[0].keys())
                data_points = [headers]
                for row_dict in raw_result:
                    row = [row_dict.get(header, None) for header in headers]
                    data_points.append(row)
                formatted_result = data_points
            else:
                formatted_result = raw_result
            
            response_data = {
                "result": formatted_result,
                "metrics": {
                    "total_time": total_time
                }
            }
            
            # Use Response with ensure_ascii=False
            return Response(
                response=json.dumps(response_data, ensure_ascii=False),
                status=200,
                mimetype='application/json; charset=utf-8'
            )
        else:
            # Get raw result and format it like the instructions endpoint
            raw_result = rows_vision.run_image_json(file_extension, filename, file_stream, model_classification, model_extraction, skip_step=True)
            
            # Convert dict format back to data_points format for consistency
            if isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], dict):
                # Convert from dict format back to data_points array format
                headers = list(raw_result[0].keys())
                data_points = [headers]
                for row_dict in raw_result:
                    row = [row_dict.get(header, None) for header in headers]
                    data_points.append(row)
                formatted_result = data_points
            else:
                formatted_result = raw_result
            
            response_data = {"result": formatted_result}
            
            # Use Response with ensure_ascii=False
            return Response(
                response=json.dumps(response_data, ensure_ascii=False),
                status=200,
                mimetype='application/json; charset=utf-8'
            )

    except Exception as e:
        logger.error(f"Unexpected error in run_external_api_one_shot: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/classify-with-instructions', methods=['POST'])
def classify_with_instructions():
    """
    Classify and extract data from image with custom instructions using single model.
    
    Expected JSON payload:
    {
        "image_url": "https://example.com/image.jpg",  # OR
        "file_path": "/path/to/local/image.jpg",       # OR
        "instructions": "Extract only the revenue data from this chart",  # optional - if empty, passes only image
        "model": "google",                            # 'google', 'openai', or 'anthropic'
        "time_outputs": false,                        # optional
        "include_name": false                         # optional - include chart name in response
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Missing JSON payload'}), 400

        # Get instructions parameter (optional - can be empty)
        instructions = data.get('instructions', '')

        model = data.get('model', 'google')
        time_outputs = data.get('time_outputs', False)
        include_name = data.get('include_name', False)

        # Validate model names
        valid_models = ['google', 'openai', 'anthropic']
        if model not in valid_models:
            return jsonify({'error': f'Invalid model. Must be one of: {valid_models}'}), 400

        image_url = data.get('image_url')
        file_path = data.get('file_path')

        if not image_url and not file_path:
            return jsonify({'error': 'Missing image_url or file_path parameter'}), 400

        if image_url and file_path:
            return jsonify({'error': 'Provide either image_url or file_path, not both'}), 400

        try:
            if image_url:
                if not image_url.startswith(('http://', 'https://')):
                    return jsonify({'error': 'Invalid image URL - must start with http:// or https://'}), 400
                file_extension, filename, file_stream = rows_vision.download_image_from_url(image_url)
            elif file_path:
                if not os.path.exists(file_path):
                    return jsonify({'error': f'File not found: {file_path}'}), 400
                file_extension = os.path.splitext(file_path)[1].lstrip(".")
                filename = os.path.basename(file_path)
                with open(file_path, "rb") as f:
                    file_stream = BytesIO(f.read())
        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}")
            return jsonify({'error': f'Failed to load image: {str(e)}'}), 400

        # Validate file extension
        good_image, file_type = classifier.check_file_extension(filename)
        if not good_image:
            return jsonify({'error': f'Unsupported file type: {filename}'}), 400

        if time_outputs:
            start_time = time()
            result = classifier.classify_with_instructions(file_stream, file_type, instructions, model)
            total_time = round(time() - start_time, 3)
            
            # Format result according to new output requirements
            formatted_result = _format_instructions_result(result, include_name)
            
            response_data = {
                "result": formatted_result,
                "metrics": {
                    "total_time": total_time
                }
            }
            
            # Use Response with ensure_ascii=False
            return Response(
                response=json.dumps(response_data, ensure_ascii=False),
                status=200,
                mimetype='application/json; charset=utf-8'
            )
        else:
            result = classifier.classify_with_instructions(file_stream, file_type, instructions, model)
            
            # Format result according to new output requirements
            formatted_result = _format_instructions_result(result, include_name)
            
            response_data = {"result": formatted_result}
            
            # Use Response with ensure_ascii=False
            return Response(
                response=json.dumps(response_data, ensure_ascii=False),
                status=200,
                mimetype='application/json; charset=utf-8'
            )

    except Exception as e:
        logger.error(f"Unexpected error in classify_with_instructions: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
   
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring and Docker."""
    return jsonify({
        'status': 'healthy',
        'service': 'rows_vision',
        'timestamp': time()
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large'}), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405


if __name__ == '__main__':
    logger.info("Starting Image Analysis API server...")
    app.run(host=config.host, port=config.port, debug=config.debug)