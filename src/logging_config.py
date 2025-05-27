"""
Logging configuration for the Image Analysis API.
"""
import logging
import logging.handlers
import sys
import os
from typing import Optional

def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None, 
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> None:
    """
    Set up comprehensive logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, logs only to console
        max_file_size: Maximum size of log file before rotation (bytes)
        backup_count: Number of backup log files to keep
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        try:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Use rotating file handler to prevent large log files
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            root_logger.info(f"Logging to file: {log_file}")
            
        except Exception as e:
            root_logger.warning(f"Failed to set up file logging: {str(e)}")
    
    # Set specific logger levels to reduce noise
    _configure_third_party_loggers()
    
    root_logger.info(f"Logging configured at {level} level")

def _configure_third_party_loggers():
    """Configure logging levels for third-party libraries to reduce noise."""
    
    # Reduce noise from common third-party libraries
    noisy_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3',
        'PIL.PngImagePlugin',
        'PIL.Image',
        'matplotlib',
        'anthropic',
        'openai',
        'google.generativeai',
        'groq',
    ]
    
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name, typically __name__ of the calling module
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class APILogger:
    """
    Specialized logger class for API operations with structured logging.
    """
    
    def __init__(self, name: str):
        """
        Initialize the API logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
    
    def log_request(self, endpoint: str, method: str, params: dict = None, ip: str = None):
        """Log incoming API request."""
        self.logger.info(
            f"API Request: {method} {endpoint}",
            extra={
                'endpoint': endpoint,
                'method': method,
                'params': params or {},
                'client_ip': ip or 'unknown'
            }
        )
    
    def log_response(self, endpoint: str, status_code: int, duration: float, success: bool):
        """Log API response."""
        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            f"API Response: {endpoint} - {status_code} ({duration:.3f}s)",
            extra={
                'endpoint': endpoint,
                'status_code': status_code,
                'duration_seconds': duration,
                'success': success
            }
        )
    
    def log_processing_step(self, step: str, duration: float, success: bool, **metadata):
        """Log processing step with timing."""
        level = logging.INFO if success else logging.ERROR
        status = "SUCCESS" if success else "FAILED"
        
        self.logger.log(
            level,
            f"Processing step: {step} - {status} ({duration:.3f}s)",
            extra={
                'step': step,
                'duration_seconds': duration,
                'success': success,
                **metadata
            }
        )
    
    def log_model_usage(self, model: str, operation: str, tokens_used: int = None, cost: float = None):
        """Log AI model usage for monitoring and cost tracking."""
        self.logger.info(
            f"Model usage: {model} for {operation}",
            extra={
                'model': model,
                'operation': operation,
                'tokens_used': tokens_used,
                'estimated_cost': cost
            }
        )
    
    def log_error(self, error: Exception, context: dict = None):
        """Log error with context information."""
        self.logger.error(
            f"Error occurred: {str(error)}",
            extra={
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context or {}
            },
            exc_info=True
        )