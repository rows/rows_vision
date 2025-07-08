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
            
            if 'error' in image_type:
                logger.error(f"Classification failed: {image_type['error']}")
                return []
            
            # Check if we can use direct data extraction (for tables with labels)
            result_final = None
            if self._can_use_direct_extraction(image_type) or skip_step:
                logger.info("Using direct data extraction from classification")
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
            
            if not result_final or not result_final[0]:
                logger.warning("No data extracted from image")
                return []
            
            # Convert to JSON format (simpler version)
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

    def _can_use_direct_extraction(self, image_type: Dict[str, Any]) -> bool:
        """
        Determine if we can use direct extraction from classification results.
        
        Args:
            image_type: Classification results
            
        Returns:
            True if direct extraction is possible, False otherwise
        """
        try:
            first_chart = next(iter(image_type.values()))
            logger.info(first_chart)
            chart_data = first_chart.get("data", {})
            
            # Check if classification included data extraction
            has_x_axis = 'xAxis' in chart_data
            chart_type = first_chart.get("image_type")
            has_data_labels = first_chart.get("has_data_labels") == 1
            
            # Tables, receipts, and charts with data labels can use direct extraction
            return has_x_axis and (chart_type in {6, 7, 8} or has_data_labels)
            
        except (StopIteration, KeyError, AttributeError):
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