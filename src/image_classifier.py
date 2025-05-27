import os
import sys
import json
import base64
import logging
import time
from io import BytesIO
from typing import Dict, Any, Tuple, List

from anthropic import Anthropic
from groq import Groq
import google.generativeai as genai
import openai
from PIL import Image as PILImage

from src.utils import extract_json_from_text
from prompts.prompt_loader import load_prompt
from src.constants import SupportedModels, SUPPORTED_IMAGE_EXTENSIONS
from src.config import ModelConfig

logger = logging.getLogger(__name__)

class ImageClassifier:
    """
    Classifies images to determine their type (chart, table, receipt, etc.) using various AI models.
    """
    
    def __init__(self, api_key: str, api_key_openai: str, api_key_gemini: str, api_key_groq: str):
        """
        Initialize the ImageClassifier with API keys for different AI services.
        
        Args:
            api_key: Anthropic API key
            api_key_openai: OpenAI API key
            api_key_gemini: Google Gemini API key
            api_key_groq: Groq API key
        """
        self.anthropic = Anthropic(api_key=api_key)
        self.client_openai = openai.OpenAI(api_key=api_key_openai)
        self.prompt = load_prompt('image_classification_final')
        self.prompt_groq = load_prompt('image_classification_final')
        
        self.anthropic_pdf = Anthropic(
            api_key=api_key, 
            default_headers={"anthropic-beta": "pdfs-2024-09-25"}
        )
        
        # Store API keys
        self.api_key_groq = api_key_groq
        self.api_key_openai = api_key_openai
        self.api_key_gemini = api_key_gemini
        
        # Load model configuration
        config = ModelConfig()
        self.gemini_model = config.gemini_model
        self.openai_model = config.openai_model
        self.anthropic_model = config.anthropic_model
        self.groq_model = config.groq_model
        self.max_tokens = config.max_tokens
        self.timeout = config.timeout
        
    def select_model(self, model: str, encoded_image: str, file_type: str, prompt: str) -> str:
        """
        Execute image classification using the specified AI model.
        
        Args:
            model: Model name ('anthropic', 'openai', 'google', 'groq')
            encoded_image: Base64 encoded image data
            file_type: MIME type of the image
            prompt: Classification prompt
            
        Returns:
            Model response as string
            
        Raises:
            ValueError: If model is not supported
            Exception: For API call failures
        """
        logger.debug('Classification prompt: %s', prompt)
        
        if model == SupportedModels.ANTHROPIC:
            message = self.anthropic.messages.create(
                model=self.anthropic_model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": file_type,
                                    "data": encoded_image
                                }
                            }
                        ]
                    }
                ]
            )
            return message.content[0].text

        elif model == SupportedModels.GROQ:
            logger.debug('Using Groq model for classification')
            client = Groq(api_key=self.api_key_groq)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt_groq},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{file_type};base64,{encoded_image}",
                                },
                            },
                        ],
                    }
                ],
                model=self.groq_model,
            )
            logger.debug('Groq classification response: %s', chat_completion.choices[0].message.content)
            return chat_completion.choices[0].message.content

        elif model == SupportedModels.GOOGLE:
            image_bytes = base64.b64decode(encoded_image)
            image = PILImage.open(BytesIO(image_bytes))
            genai.configure(api_key=self.api_key_gemini)
            gemini_model = genai.GenerativeModel(self.gemini_model)
            response = gemini_model.generate_content([prompt, image])
            return response.text

        elif model == SupportedModels.OPENAI:
            logger.debug('Using OpenAI model for classification')
            response = self.client_openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{file_type};base64,{encoded_image}",
                                },
                            },
                        ]
                    }
                ],
                max_tokens=4000
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Model '{model}' not supported. Supported models: {list(SupportedModels.__dict__.values())}")

    def classify_chart_image(self, image_stream: BytesIO, file_type: str, model: str = SupportedModels.ANTHROPIC) -> Dict[str, Any]:
        """
        Classify chart image type using specified AI model with retry logic.
        
        Args:
            image_stream: Image data as BytesIO stream
            file_type: MIME type of the image
            model: AI model to use ('anthropic', 'openai', 'google', 'groq')
        
        Returns:
            Dictionary containing classification results
            
        Raises:
            ValueError: If model is not supported
        """
        image_stream.seek(0)
        encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')
        image_stream.seek(0)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.select_model(model, encoded_image, file_type, self.prompt)
                logger.debug('Classification response: %s', response)
                
                # Try to parse the response as JSON
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    return extract_json_from_text(response)
                    
            except Exception as e:
                logger.warning(f"Classification attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} classification attempts failed")
                    return {"error": f"Failed after {max_retries} attempts: {str(e)}"}
                time.sleep(2 ** attempt)  # Exponential backoff

    def check_file_extension(self, filename: str) -> Tuple[bool, str]:
        """
        Check if file has a supported image extension and return MIME type.
        
        Args:
            filename: Name of the file to check
            
        Returns:
            Tuple of (is_valid_image, mime_type)
            
        Examples:
            >>> classifier.check_file_extension("image.jpg")
            (True, "image/jpeg")
            >>> classifier.check_file_extension("document.pdf")
            (False, "image/jpeg")
        """
        if '.' not in filename:
            return False, 'image/jpeg'
            
        extension = filename.rsplit('.', 1)[1].lower()
        good_image = extension in SUPPORTED_IMAGE_EXTENSIONS
        
        # Map extensions to MIME types
        mime_type_map = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
            'heic': 'image/heic'
        }
        
        file_type = mime_type_map.get(extension, 'image/jpeg')
        
        return good_image, file_type