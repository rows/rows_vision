"""
Constants used throughout the Image Analysis API.
"""

class ImageTypes:
    """Constants for different types of images/charts that can be processed."""
    
    LINE_CHART = 1
    MULTI_LINE_CHART = 2
    BAR_CHART = 3
    SCATTER_CHART = 4
    PIE_CHART = 5
    TABLE = 6
    RECEIPT = 7
    TABLE_ALT = 8

class SupportedModels:
    """Constants for supported AI models."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"
    
    # List of all supported models
    ALL = [ANTHROPIC, OPENAI, GOOGLE, GROQ]
    
    @classmethod
    def is_valid(cls, model: str) -> bool:
        """Check if a model name is valid."""
        return model in cls.ALL

class FileExtensions:
    """Constants for supported file extensions."""
    
    # Image extensions
    JPG = 'jpg'
    JPEG = 'jpeg'
    PNG = 'png'
    GIF = 'gif'
    WEBP = 'webp'
    HEIC = 'heic'
    
    # Document extensions
    PDF = 'pdf'

class MimeTypes:
    """Constants for MIME types."""
    
    JPEG = 'image/jpeg'
    PNG = 'image/png'
    GIF = 'image/gif'
    WEBP = 'image/webp'
    HEIC = 'image/heic'
    PDF = 'application/pdf'

class ProcessingStatus:
    """Constants for processing status."""
    
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_FORMAT = "unsupported_format"

class ErrorCodes:
    """Constants for error codes."""
    
    # Input validation errors
    MISSING_IMAGE_URL = "missing_image_url"
    INVALID_IMAGE_URL = "invalid_image_url"
    MISSING_PAYLOAD = "missing_payload"
    INVALID_MODEL = "invalid_model"
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    FILE_TOO_LARGE = "file_too_large"
    FILE_NOT_FOUND = "file_not_found"
    
    # Processing errors
    DOWNLOAD_FAILED = "download_failed"
    CLASSIFICATION_FAILED = "classification_failed"
    EXTRACTION_FAILED = "extraction_failed"
    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    
    # System errors
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"

# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS = {
    FileExtensions.PNG,
    FileExtensions.JPG,
    FileExtensions.JPEG,
    FileExtensions.GIF,
    FileExtensions.WEBP,
    FileExtensions.HEIC
}

# Mapping of file extensions to MIME types
EXTENSION_TO_MIME_TYPE = {
    FileExtensions.JPG: MimeTypes.JPEG,
    FileExtensions.JPEG: MimeTypes.JPEG,
    FileExtensions.PNG: MimeTypes.PNG,
    FileExtensions.GIF: MimeTypes.GIF,
    FileExtensions.WEBP: MimeTypes.WEBP,
    FileExtensions.HEIC: MimeTypes.HEIC,
    FileExtensions.PDF: MimeTypes.PDF,
}

# Chart type descriptions for logging and debugging
CHART_TYPE_DESCRIPTIONS = {
    ImageTypes.LINE_CHART: "Single Line Chart",
    ImageTypes.MULTI_LINE_CHART: "Multi-Line Chart",
    ImageTypes.BAR_CHART: "Bar Chart",
    ImageTypes.SCATTER_CHART: "Scatter Plot",
    ImageTypes.PIE_CHART: "Pie Chart",
    ImageTypes.TABLE: "Data Table",
    ImageTypes.RECEIPT: "Receipt/Invoice",
    ImageTypes.TABLE_ALT: "Alternative Table Format",
}

# Default prompt names for different image types
PROMPT_NAMES = {
    ImageTypes.LINE_CHART: "chart_one_line",
    ImageTypes.MULTI_LINE_CHART: "chart_multi_line",
    ImageTypes.BAR_CHART: "chart_bar",
    ImageTypes.SCATTER_CHART: "chart_scatter",
    ImageTypes.PIE_CHART: "chart_pie",
    ImageTypes.TABLE: "chart_table",
    ImageTypes.RECEIPT: "chart_receipt",
    ImageTypes.TABLE_ALT: "chart_table",
}

# API response field names
class ResponseFields:
    """Constants for API response field names."""
    
    RESULT = "result"
    ERROR = "error"
    METRICS = "metrics"
    TOTAL_TIME = "total_time"
    STATUS = "status"
    MESSAGE = "message"
    CODE = "code"

# HTTP status codes
class HTTPStatus:
    """Constants for HTTP status codes."""
    
    OK = 200
    BAD_REQUEST = 400
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    REQUEST_ENTITY_TOO_LARGE = 413
    INTERNAL_SERVER_ERROR = 500
    SERVICE_UNAVAILABLE = 503

# Logging constants
class LogLevels:
    """Constants for logging levels."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"