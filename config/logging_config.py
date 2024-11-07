import logging
import logging.handlers
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Optional


class CustomJSONFormatter(logging.Formatter):
    """Custom JSON formatter for logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread
        }

        if hasattr(record, "extra"):
            log_data.update(record.extra)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(
        log_dir: Path,
        level: int = logging.INFO,
        retention: int = 30,
        max_size: int = 10 * 1024 * 1024,  # 10MB
        console_output: bool = True
) -> None:
    """Configure logging settings"""

    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handlers
    handlers_config = {
        "error": {
            "level": logging.ERROR,
            "filename": "error.log"
        },
        "info": {
            "level": logging.INFO,
            "filename": "info.log"
        },
        "debug": {
            "level": logging.DEBUG,
            "filename": "debug.log"
        }
    }

    for handler_name, config in handlers_config.items():
        handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / config["filename"],
            maxBytes=max_size,
            backupCount=retention
        )
        handler.setLevel(config["level"])
        handler.setFormatter(CustomJSONFormatter())
        root_logger.addHandler(handler)


class LoggerManager:
    """Manager class for handling loggers"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def log_with_context(self,
                         level: int,
                         message: str,
                         context: Optional[Dict[str, Any]] = None) -> None:
        """Log message with additional context"""
        extra = {"extra": context} if context else {}
        self.logger.log(level, message, extra=extra)

    def log_model_metrics(self,
                          model_name: str,
                          metrics: Dict[str, Any]) -> None:
        """Log model performance metrics"""
        self.log_with_context(
            logging.INFO,
            f"Model metrics for {model_name}",
            metrics
        )

    def log_api_request(self,
                        endpoint: str,
                        method: str,
                        response_time: float,
                        status_code: int) -> None:
        """Log API request details"""
        context = {
            "endpoint": endpoint,
            "method": method,
            "response_time": response_time,
            "status_code": status_code
        }
        self.log_with_context(logging.INFO, "API Request", context)

    def log_error(self,
                  error: Exception,
                  context: Optional[Dict[str, Any]] = None) -> None:
        """Log error with context"""
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        if context:
            error_context.update(context)
        self.log_with_context(logging.ERROR, "Error occurred", error_context)