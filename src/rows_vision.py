import os 
import logging
import requests
import mimetypes
import json
import uuid
import io
from werkzeug.utils import secure_filename
from collections import OrderedDict
from typing import Dict, Any, List, Tuple, Optional
from io import BytesIO

from src.constants import SupportedModels
from src.config import ModelConfig

logger = logging.getLogger(__name__)

class RowsVision:
    """
    Main orchestrator class that combines image classification and analysis functionality.
    """
    
    def __init__(self, classifier, analyzer):
        """
        Initialize RowsVision with classifier and analyzer components.
        
        Args:
            classifier: ImageClassifier instance
            analyzer: ImageAnalyzer instance
        """
        self.classifier = classifier
        self.analyzer = analyzer
        self.config = ModelConfig()

    def run_image_json(self, file_extension: str, filename: str, file_stream: BytesIO, 
                      model_classification: str, model_extraction: str, skip_step = False) -> List[Dict[str, Any]]:
        """
        Process image file and return extracted data as JSON.
        
        Args:
            file_extension: File extension (e.g., 'jpg', 'png')
            filename: Original filename
            file_stream: Image data as BytesIO stream
            model_classification: AI model to use for classification
            model_extraction: AI model to use for data extraction
            
        Returns:
            List of dictionaries containing extracted data
            
        Raises:
            Exception: If processing fails
        """
        try:
            # Skip PDF files for now
            if file_extension.lower() == 'pdf':
                logger.info("PDF files are not currently supported")
                return []
            
            # Validate file extension and get MIME type
            is_valid, file_type = self.classifier.check_file_extension(filename)
            if not is_valid:
                logger.warning(f"Unsupported file extension: {file_extension}")
                return []
            
            # Classify the image to determine its type
            logger.info(f"Classifying image: {filename}")
            image_type = self.classifier.classify_chart_image(file_stream, file_type, model_classification)
            
            logger.info(f"Classification result type: {type(image_type)}")
            logger.info(f"Classification result: {image_type}")
            
            if isinstance(image_type, dict) and 'error' in image_type:
                logger.error(f"Classification failed: {image_type['error']}")
                return []
            elif isinstance(image_type, list) and len(image_type) > 0 and isinstance(image_type[0], dict) and 'error' in image_type[0]:
                logger.error(f"Classification failed: {image_type[0]['error']}")
                return []
            
            # Check if we can use direct data extraction (for tables with labels)
            result_final = None
            logger.info(f"Checking if can use direct extraction for image_type: {type(image_type)}")
            can_extract = self._can_use_direct_extraction(image_type)
            logger.info(f"Can use direct extraction: {can_extract}, skip_step: {skip_step}")
            
            if can_extract or skip_step:
                logger.info("Using direct data extraction from classification")
                
                # Handle new list format from classification_with_instructions
                if isinstance(image_type, list):
                    logger.info("Processing list format for direct extraction")
                    result_final = self.analyzer.compile_results(image_type)
                elif isinstance(image_type, dict) and 'type' in image_type and 'data_points' in image_type:
                    logger.info("Processing new dict format (single chart) for direct extraction")
                    # Convert single chart dict to list format for compile_results
                    result_final = self.analyzer.compile_results([image_type])
                else:
                    logger.info("Processing old dict format for direct extraction")
                    # Handle old dictionary format (fallback)
                    # Move nested data up one level for processing
                    for chart in image_type.values():
                        if "data" in chart:
                            chart.update(chart["data"])  # Move xAxis, yAxis, dataPoints up
                            del chart["data"]            # Remove the nested data key
                    result_final = self.analyzer.compile_results(image_type)
            else:
                # Use full analysis pipeline
                logger.info("Using full analysis pipeline with model: " + model_extraction)
                result = self.analyzer.analyze_graph(image_type, file_stream, file_type, model_extraction)
                
                if 'error' in result:
                    logger.error(f"Analysis failed: {result['error']}")
                    return []
                    
                result_final = self.analyzer.compile_results(result)
            
            logger.info(f"result_final type: {type(result_final)}")
            logger.info(f"result_final content: {result_final}")
            
            if not result_final or not result_final[0]:
                logger.warning("No data extracted from image")
                return []
            
            # Validate result structure
            logger.info(f"Validating result_final: type={type(result_final)}, len={len(result_final) if result_final else 'None'}")
            if not isinstance(result_final, list) or len(result_final) == 0:
                logger.warning("Invalid result_final structure")
                return []
            
            logger.info(f"result_final[0] type: {type(result_final[0])}, len={len(result_final[0]) if result_final[0] else 'None'}")
            if not isinstance(result_final[0], list) or len(result_final[0]) == 0:
                logger.warning("Invalid result_final[0] structure")
                return []
            
            # Convert to JSON format (simpler version)
            logger.info(f"About to extract headers from result_final[0][0]: {result_final[0][0]}")
            headers = result_final[0][0]
            logger.debug("Extracted headers: %s", headers)

            json_output = []
            for row in result_final[0][1:]:
                row_dict = OrderedDict()
                for col_idx in range(len(headers)):
                    header = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"
                    value = row[col_idx] if col_idx < len(row) else None
                    row_dict[str(header)] = value
                json_output.append(row_dict)
            
            # Fixed: Use ensure_ascii=False to preserve special characters
            logger.debug("Final JSON output: %s", json.dumps(json_output, indent=2, ensure_ascii=False))
            logger.info(f"Successfully processed {filename} and extracted {len(json_output)} rows")
            return json_output
            
        except Exception as e:
            logger.error(f"Error processing image {filename}: {str(e)}")
            raise
        finally:
            # Ensure file stream is closed
            if file_stream and not file_stream.closed:
                file_stream.close()
                logger.debug("File stream closed")

    def _can_use_direct_extraction(self, image_type) -> bool:
        """
        Determine if we can use direct extraction from classification results.
        
        Args:
            image_type: Classification results (can be dict or list)
            
        Returns:
            True if direct extraction is possible, False otherwise
        """
        try:
            logger.info(f"_can_use_direct_extraction called with type: {type(image_type)}")
            logger.info(f"_can_use_direct_extraction data: {image_type}")
            
            # Handle new list format from classification_with_instructions
            if isinstance(image_type, list) and len(image_type) > 0:
                logger.info("Processing list format in _can_use_direct_extraction")
                first_chart = image_type[0]
                logger.info(f"First chart: {first_chart}")
                chart_type = first_chart.get("type")
                has_data_labels = first_chart.get("has_data_labels") == 1
                has_data_points = 'data_points' in first_chart and first_chart['data_points']
                
                logger.info(f"Chart type: {chart_type}, has_data_labels: {has_data_labels}, has_data_points: {has_data_points}")
                
                # Tables, receipts, infographics, and charts with data labels can use direct extraction
                result = has_data_points and (chart_type in {6, 7, 8} or has_data_labels)
                logger.info(f"Direct extraction result for list: {result}")
                return result
            
            # Handle dictionary format (could be new format or old format)
            elif isinstance(image_type, dict):
                logger.info("Processing dict format in _can_use_direct_extraction")
                
                # Check if this is the new format (single chart dict with direct fields)
                if 'type' in image_type and 'data_points' in image_type:
                    logger.info("Dict is in new format (single chart)")
                    chart_type = image_type.get("type")
                    has_data_labels = image_type.get("has_data_labels") == 1
                    has_data_points = 'data_points' in image_type and image_type['data_points']
                    
                    logger.info(f"Chart type (new dict): {chart_type}, has_data_labels: {has_data_labels}, has_data_points: {has_data_points}")
                    
                    # Tables, receipts, infographics, and charts with data labels can use direct extraction
                    result = has_data_points and (chart_type in {6, 7, 8} or has_data_labels)
                    logger.info(f"Direct extraction result for new dict: {result}")
                    return result
                
                # Handle old dictionary format (nested charts)
                else:
                    logger.info("Dict is in old format (nested charts)")
                    first_chart = next(iter(image_type.values()))
                    logger.info(f"First chart (old dict): {first_chart}")
                    chart_data = first_chart.get("data", {})
                    
                    # Check if classification included data extraction
                    has_x_axis = 'xAxis' in chart_data
                    chart_type = first_chart.get("image_type")
                    has_data_labels = first_chart.get("has_data_labels") == 1
                    
                    logger.info(f"Chart type (old dict): {chart_type}, has_data_labels: {has_data_labels}, has_x_axis: {has_x_axis}")
                    
                    # Tables, receipts, and charts with data labels can use direct extraction
                    result = has_x_axis and (chart_type in {6, 7, 8} or has_data_labels)
                    logger.info(f"Direct extraction result for old dict: {result}")
                    return result
            
            logger.info("Unknown format, returning False")
            return False
            
        except (StopIteration, KeyError, AttributeError, TypeError) as e:
            logger.error(f"Error in _can_use_direct_extraction: {str(e)}")
            return False

    def download_image_from_url(self, image_url: str) -> Tuple[str, str, BytesIO]:
        """
        Download image from URL and return file information and stream.
        
        Args:
            image_url: URL of the image to download
            
        Returns:
            Tuple of (file_extension, filename, file_stream)
            
        Raises:
            ValueError: If unable to determine file extension
            Exception: If download fails
        """
        try:
            logger.info(f"Downloading image from: {image_url}")
            response = requests.get(
                image_url, 
                stream=True, 
                timeout=self.config.timeout,
                headers={'User-Agent': 'RowsVision/1.0'}
            )
            response.raise_for_status()
            
            # Determine file extension from content type
            content_type = response.headers.get('content-type', '')
            extension = mimetypes.guess_extension(content_type)
            
            if extension is None:
                # Try to guess from URL
                url_path = image_url.split('/')[-1].split('?')[0]
                if '.' in url_path:
                    extension = '.' + url_path.split('.')[-1].lower()
                else:
                    # Default to .jpg if we can't determine
                    extension = '.jpg'
                    logger.warning(f"Could not determine file extension from content-type: {content_type}, defaulting to .jpg")

            # Generate a unique filename using UUID
            unique_id = str(uuid.uuid4())
            original_name = secure_filename(image_url.split('/')[-1].split('?')[0])
            
            # Ensure original_name has some content
            if not original_name or original_name == '':
                original_name = 'image'
                
            filename = f"{unique_id}_{original_name}{extension}"

            # Load content into memory
            file_stream = BytesIO(response.content)
            file_stream.name = filename  # Set name attribute for reference

            file_extension = extension.lstrip('.')
            
            logger.info(f"Successfully downloaded image: {filename} ({len(response.content)} bytes)")
            return file_extension, filename, file_stream
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading image from {image_url}")
            raise Exception(f"Timeout downloading image from {image_url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image from {image_url}: {str(e)}")
            raise Exception(f"Failed to download image. Status code: {getattr(e.response, 'status_code', 'Unknown')}")
        except Exception as e:
            logger.error(f"Unexpected error downloading image from {image_url}: {str(e)}")
            raise