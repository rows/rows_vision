"""
Configuration settings for the Image Analysis API.
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for AI model settings."""
    
    # Model versions
    anthropic_model: str = "claude-3-7-sonnet-20250219"
    openai_model: str = "gpt-4o"
    gemini_model: str = "gemini-2.0-flash"
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    # Request settings
    max_tokens: int = 8000
    timeout: int = 30  # seconds
    
    # Retry settings
    max_retries: int = 3
    retry_delay_base: int = 2  # seconds, for exponential backoff
    
    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Create ModelConfig from environment variables."""
        return cls(
            anthropic_model=os.getenv('ANTHROPIC_MODEL', cls.anthropic_model),
            openai_model=os.getenv('OPENAI_MODEL', cls.openai_model),
            gemini_model=os.getenv('GEMINI_MODEL', cls.gemini_model),
            groq_model=os.getenv('GROQ_MODEL', cls.groq_model),
            max_tokens=int(os.getenv('MAX_TOKENS', cls.max_tokens)),
            timeout=int(os.getenv('REQUEST_TIMEOUT', cls.timeout)),
            max_retries=int(os.getenv('MAX_RETRIES', cls.max_retries)),
        )

@dataclass
class AppConfig:
    """Configuration for Flask application settings."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    
    # Security settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_origins: list = None
    
    # Processing settings
    enable_timing_metrics: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Initialize default values after creation."""
        if self.allowed_origins is None:
            self.allowed_origins = ['*']  # Allow all origins by default
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create AppConfig from environment variables."""
        return cls(
            host=os.getenv('HOST', cls.host),
            port=int(os.getenv('PORT', cls.port)),
            debug=os.getenv('DEBUG', 'false').lower() in ('true', '1', 'yes'),
            max_file_size=int(os.getenv('MAX_FILE_SIZE', cls.max_file_size)),
            enable_timing_metrics=os.getenv('ENABLE_TIMING', 'true').lower() in ('true', '1', 'yes'),
            log_level=os.getenv('LOG_LEVEL', cls.log_level).upper(),
        )

@dataclass
class APIConfig:
    """Configuration for API behavior."""
    
    # Default models
    default_classification_model: str = "anthropic"
    default_extraction_model: str = "anthropic"
    
    # Processing limits
    max_charts_per_image: int = 10
    max_data_points_per_chart: int = 1000
    
    # Feature flags
    enable_pdf_processing: bool = False
    enable_batch_processing: bool = False
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Create APIConfig from environment variables."""
        return cls(
            default_classification_model=os.getenv('DEFAULT_CLASSIFICATION_MODEL', cls.default_classification_model),
            default_extraction_model=os.getenv('DEFAULT_EXTRACTION_MODEL', cls.default_extraction_model),
            max_charts_per_image=int(os.getenv('MAX_CHARTS_PER_IMAGE', cls.max_charts_per_image)),
            max_data_points_per_chart=int(os.getenv('MAX_DATA_POINTS_PER_CHART', cls.max_data_points_per_chart)),
            enable_pdf_processing=os.getenv('ENABLE_PDF', 'false').lower() in ('true', '1', 'yes'),
            enable_batch_processing=os.getenv('ENABLE_BATCH', 'false').lower() in ('true', '1', 'yes'),
        )

# Global configuration instances
model_config = ModelConfig.from_env()
app_config = AppConfig.from_env()
api_config = APIConfig.from_env()