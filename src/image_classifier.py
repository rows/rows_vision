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
#import google.generativeai as genai
from google import genai
from google.genai import types
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
        self.prompt = load_prompt('classification_with_instructions')
        self.prompt_groq = load_prompt('classification_with_instructions')
        
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
                system=prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
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
            logger.info('Claude classification response: %s', message.content[0].text)
            
            return message.content[0].text

        elif model == SupportedModels.GROQ:
            logger.debug('Using Groq model for classification')
            client = Groq(api_key=self.api_key_groq)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": [
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
            client = genai.Client(
                api_key=self.api_key_gemini
            )
            gemini_model = self.gemini_model
            generate_content_config = types.GenerateContentConfig(
                thinking_config = types.ThinkingConfig(
                    thinking_budget=0,
                ),
                response_mime_type="text/plain",
                system_instruction=prompt
            )
            response = client.models.generate_content(model = gemini_model, contents = [image], config = generate_content_config)
            return response.text

        elif model == SupportedModels.OPENAI:
            logger.debug('Using OpenAI model for classification')
            response = self.client_openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{file_type};base64,{encoded_image}",
                                },
                            },
                        ]
                    }
                ],
                max_completion_tokens=8000
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

    def classify_with_instructions(self, image_stream: BytesIO, file_type: str, instructions: str, model: str) -> Dict[str, Any]:
        """
        Classify and extract data from image with custom instructions using system/user message structure.
        
        Args:
            image_stream: Image data as BytesIO stream
            file_type: MIME type of the image
            instructions: Custom instructions for extraction
            model: AI model to use ('google' or 'openai')
        
        Returns:
            Dictionary containing classification and extraction results
            
        Raises:
            ValueError: If model is not supported
        """
        # Validate model support
        if model not in [SupportedModels.GOOGLE, SupportedModels.OPENAI, SupportedModels.ANTHROPIC]:
            raise ValueError(f"Model '{model}' not supported for instructions endpoint. Supported models: ['google', 'openai', 'anthropic']")
        
        image_stream.seek(0)
        encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')
        image_stream.seek(0)
        
        # Load system prompt
        system_prompt = load_prompt('classification_with_instructions')
        
        # Handle empty instructions - pass only image
        if not instructions or instructions.strip() == "":
            instructions = ""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if model == SupportedModels.GOOGLE:
                    response = self._classify_with_instructions_gemini(encoded_image, file_type, system_prompt, instructions)
                elif model == SupportedModels.OPENAI:
                    response = self._classify_with_instructions_openai(encoded_image, file_type, system_prompt, instructions)
                elif model == SupportedModels.ANTHROPIC:
                    response = self._classify_with_instructions_anthropic(encoded_image, file_type, system_prompt, instructions)
                
                logger.debug('Classification with instructions response: %s', response)
                
                # Try to parse the response as JSON
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    return extract_json_from_text(response)
                    
            except Exception as e:
                logger.warning(f"Classification with instructions attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} classification with instructions attempts failed")
                    return {"error": f"Failed after {max_retries} attempts: {str(e)}"}
                time.sleep(2 ** attempt)  # Exponential backoff

    def _classify_with_instructions_gemini(self, encoded_image: str, file_type: str, system_prompt: str, instructions: str) -> str:
        """
        Classify with instructions using Gemini model with system/user message structure.
        """
        image_bytes = base64.b64decode(encoded_image)
        image = PILImage.open(BytesIO(image_bytes))
        
        client = genai.Client(api_key=self.api_key_gemini)
        gemini_model = self.gemini_model
        
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="text/plain",
            system_instruction=system_prompt
        )
        
        # If instructions are empty, pass only image
        if instructions.strip() == "":
            contents = [image]
        else:
            contents = [instructions, image]
        
        response = client.models.generate_content(
            model=gemini_model,
            contents=contents,
            config=generate_content_config
        )
        
        return response.text

    def _classify_with_instructions_openai(self, encoded_image: str, file_type: str, system_prompt: str, instructions: str) -> str:
        """
        Classify with instructions using OpenAI model with system/user message structure.
        """
        # Build user content - if instructions are empty, pass only image
        if instructions.strip() == "":
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{file_type};base64,{encoded_image}",
                    },
                }
            ]
        else:
            user_content = [
                {"type": "text", "text": instructions},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{file_type};base64,{encoded_image}",
                    },
                },
            ]
        
        response = self.client_openai.chat.completions.create(
            model=self.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            max_completion_tokens=8000
        )
        
        return response.choices[0].message.content

    def _classify_with_instructions_anthropic(self, encoded_image: str, file_type: str, system_prompt: str, instructions: str) -> str:
        """
        Classify with instructions using Anthropic model with system/user message structure.
        """
        # Build user content - if instructions are empty, pass only image
        if instructions.strip() == "":
            user_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": file_type,
                        "data": encoded_image
                    }
                }
            ]
        else:
            user_content = [
                {
                    "type": "text",
                    "text": instructions
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
        
        message = self.anthropic.messages.create(
            model=self.anthropic_model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        )
        
        return message.content[0].text

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