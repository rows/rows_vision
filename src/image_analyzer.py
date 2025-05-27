import os
import json
import base64
import logging
import time
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional

import requests
from anthropic import Anthropic
from groq import Groq
import google.generativeai as genai
import openai
from PIL import Image as PILImage

from src.utils import extract_json_from_text
from prompts.prompt_loader import load_prompt
from src.constants import ImageTypes, SupportedModels
from src.config import ModelConfig

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """
    Analyzes images using various AI models to extract structured data from charts, tables, and receipts.
    """
    
    def __init__(self, api_key: str, api_key_openai: str, api_key_gemini: str, api_key_groq: str):
        """
        Initialize the ImageAnalyzer with API keys for different AI services.
        
        Args:
            api_key: Anthropic API key
            api_key_openai: OpenAI API key
            api_key_gemini: Google Gemini API key
            api_key_groq: Groq API key
        """
        self.anthropic = Anthropic(api_key=api_key)
        self.client_openai = openai.OpenAI(api_key=api_key_openai)
        self.api_key_openai = api_key_openai
        self.api_key_groq = api_key_groq
        self.api_key_gemini = api_key_gemini
        
        # Load model configuration
        config = ModelConfig()
        self.gemini_model = config.gemini_model
        self.openai_model = config.openai_model
        self.anthropic_model = config.anthropic_model
        self.groq_model = config.groq_model
        self.max_tokens = config.max_tokens
        self.timeout = config.timeout

    def get_all_values_axis(self, image_info: Dict[str, Any], image_stream: BytesIO) -> Dict[str, Dict[str, List]]:
        """
        Extract axis values from sampled charts using OpenAI's vision model.
        
        Args:
            image_info: Dictionary containing chart information with sampling status
            image_stream: Image data as BytesIO stream
            
        Returns:
            Dictionary mapping chart names to their x-axis values
        """
        image_stream.seek(0)
        base64_image = base64.b64encode(image_stream.read()).decode('utf-8')
        image_stream.seek(0)
        
        sampled_list = []
        for chart in image_info:
            if image_info[chart]['sampled_axis'] == 1:
                sampled_list.append('True')
            else:
                sampled_list.append('False')
                
        if all(x == 'False' for x in sampled_list):
            actual_list_x = {}
            for chart in image_info:
                actual_list_x[chart] = {'x_axis_values': []}
            return actual_list_x

        prompt = load_prompt('unsample_axis') + str(sampled_list)
        logger.debug("Axis extraction prompt: %s", prompt)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key_openai}"
        }

        payload_x = {
            "model": self.openai_model,
            "messages": [
                {
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
                }
            ],
            "max_tokens": 4000,
            "response_format": {"type": "json_object"}
        }

        try:
            response_x = requests.post(
                "https://api.openai.com/v1/chat/completions", 
                headers=headers, 
                json=payload_x,
                timeout=self.timeout
            )
            response_x.raise_for_status()
            
            logger.debug("OpenAI axis response: %s", response_x.json())
            actual_list_x = json.loads(
                response_x.json()['choices'][0]['message']['content'].strip('```json\n```').rstrip('')
            )
            logger.debug('Result of unsampling: %s', actual_list_x)
            return actual_list_x
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API request failed: {str(e)}")
            return {}
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse OpenAI response: {str(e)}")
            return {}

    def select_prompt(self, image_type: int) -> str:
        """
        Select appropriate prompt based on image type.
        
        Args:
            image_type: Integer representing the type of chart/image
            
        Returns:
            Loaded prompt string for the given image type
        """
        prompt_map = {
            ImageTypes.LINE_CHART: "chart_one_line",
            ImageTypes.MULTI_LINE_CHART: "chart_multi_line",
            ImageTypes.BAR_CHART: "chart_bar",
            ImageTypes.SCATTER_CHART: "chart_scatter",
            ImageTypes.PIE_CHART: "chart_pie",
            ImageTypes.TABLE: "chart_table",
            ImageTypes.RECEIPT: "chart_receipt",
            ImageTypes.TABLE_ALT: "chart_table"
        }

        prompt_name = prompt_map.get(image_type, "chart_table")
        prompt_content = load_prompt(prompt_name)
        logger.debug('Selected prompt: %s', prompt_name)
        return prompt_content
  
    def select_model(self, model: str, encoded_image: str, file_type: str, prompt: str) -> str:
        """
        Execute image analysis using the specified AI model.
        
        Args:
            model: Model name ('anthropic', 'openai', 'google', 'groq')
            encoded_image: Base64 encoded image data
            file_type: MIME type of the image
            prompt: Analysis prompt
            
        Returns:
            Model response as string
            
        Raises:
            ValueError: If model is not supported
        """
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
            logger.debug('Using Groq model for analysis')
            client = Groq(api_key=self.api_key_groq)
            chat_completion = client.chat.completions.create(
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
                        ],
                    }
                ],
                model=self.groq_model,
            )
            logger.debug('Groq response: %s', chat_completion.choices[0].message.content)
            return chat_completion.choices[0].message.content

        elif model == SupportedModels.GOOGLE:
            image_bytes = base64.b64decode(encoded_image)
            image = PILImage.open(BytesIO(image_bytes))
            genai.configure(api_key=self.api_key_gemini)
            gemini_model = genai.GenerativeModel(self.gemini_model)
            response = gemini_model.generate_content([prompt, image])
            return response.text

        elif model == SupportedModels.OPENAI:
            logger.debug('Using OpenAI model for analysis')
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

    def analyze_chart_image(self, encoded_image: str, file_type: str, prompt: str, model: str = SupportedModels.ANTHROPIC) -> Dict[str, Any]:
        """
        Analyze chart image with retry logic and error handling.
        
        Args:
            encoded_image: Base64 encoded image data
            file_type: MIME type of the image
            prompt: Analysis prompt
            model: AI model to use
            
        Returns:
            Dictionary containing extracted data or error information
        """
        logger.debug('Starting chart image analysis')
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.select_model(model, encoded_image, file_type, prompt)
                logger.debug('Model response: %s', response)
                
                # Try to parse as JSON
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    return extract_json_from_text(response)
                    
            except Exception as e:
                logger.warning(f"Analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} analysis attempts failed")
                    return {"error": f"Failed after {max_retries} attempts: {str(e)}"}
                time.sleep(2 ** attempt)  # Exponential backoff

    def analyze_graph(self, image_info: Dict[str, Any], image_stream: BytesIO, file_type: str, model: str = SupportedModels.ANTHROPIC) -> Dict[str, Any]:
        """
        Analyze multiple charts/graphs in a single image.
        
        Args:
            image_info: Dictionary containing information about charts in the image
            image_stream: Image data as BytesIO stream
            file_type: MIME type of the image
            model: AI model to use for analysis
            
        Returns:
            Dictionary containing analysis results for all charts
        """
        image_stream.seek(0)
        encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')
        image_stream.seek(0)
        
        prompt = '''You will have to analyze an image that may have 1 or more charts/tables/items in it. 
                I will give you instructions for you to perform operations, one by one, by the order they appear in the image (from left to right, top to bottom).
                Answer with a json with the following structure {"chart_1": json from chart 1, "chart_2": ...}
                '''

        x_axis_values = self.get_all_values_axis(image_info, image_stream)

        for i, chart in enumerate(x_axis_values):
            prompt += f'Chart {i+1}:\n'
            selected_prompt = self.select_prompt(image_info[chart]['image_type'])
            x_axis = x_axis_values[chart]['x_axis_values']
            prompt += selected_prompt + '''
            This dictionary represents the x values that should have values for each chart. 
            The ones that are empty are unsampled, so you can extract them directly from the image.
            x-axis points: 
            ''' + str(x_axis) + '''; 
            End of image.
            
            '''
            
        prompt += '''Remember, please return ONLY a valid JSON object and nothing else. Do not include explanations, markdown, or extra text.'''
        logger.debug('Complete analysis prompt: %s', prompt)
        return self.analyze_chart_image(encoded_image, file_type, prompt, model)

    def compile_results(self, base_data: Dict[str, Any]) -> List[List[List]]:
        """
        Compile analysis results into a structured format for output.
        
        Args:
            base_data: Dictionary containing chart analysis results
            
        Returns:
            List of compiled results, where each element represents a chart's data
        """
        results_list = []
        
        for chart in base_data:
            data = base_data[chart]
            final_results = []

            if 'dataPoints' not in data or not data['dataPoints']:
                logger.warning(f"No dataPoints found in chart {chart}")
                results_list.append([])
                continue

            n_points_series = len(data['dataPoints'][0]) if data['dataPoints'] else 0
            
            # Handle x-axis title
            if 'title' in data.get('xAxis', {}):
                title_array = [data['xAxis']['title']]
            else:
                title_array = ['X']
            # Handle y-axis series/titles
            if 'series' in data.get('yAxis', {}):
                series = data['yAxis']['series']
                if len(series) == n_points_series:
                    title_array[0] = series[0]
                    title_array.extend(series[1:])
                elif len(series) >= 1:
                    # Use first series name for additional columns
                    for a in range(min(len(series), n_points_series - 1)):
                        title_array.append(series[a])
                else:
                    # Fallback to series names
                    for _ in range(n_points_series - 1):
                        title_array.append(series[0])
                    
                        
            elif 'title' in data.get('yAxis', {}):
                titles = data['yAxis']['title']
                if isinstance(titles, list):
                    if len(titles) == n_points_series:
                        title_array[0] = titles[0]
                        title_array.extend(titles[1:])
                    elif len(titles) >= 1:
                        for _ in range(n_points_series - 1):
                            title_array.append(titles[0])
                    else:
                        for a in range(min(len(titles), n_points_series - 1)):
                            title_array.append(titles[a])
                else:
                    # Single title string
                    for _ in range(n_points_series - 1):
                        title_array.append(titles)
            else: 
                # No y-axis information, use default
                for _ in range(n_points_series - 1):
                    title_array.append('Values')

            # Ensure title_array has correct length
            while len(title_array) < n_points_series:
                title_array.append('Values')
                
            final_results.append(title_array)

            # Add data points
            for data_point in data['dataPoints']:
                row = []
                for key, value in data_point.items():
                    row.append(value)
                final_results.append(row)

            results_list.append(final_results)
            
        return results_list