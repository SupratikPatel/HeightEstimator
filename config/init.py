from .settings import config
from .logging_config import setup_logging, LoggerManager
from pathlib import Path

# Initialize logging
setup_logging(Path(__file__).parent.parent / "logs")

# Create logger manager instance
logger = LoggerManager(__name__)

__all__ = ['config', 'setup_logging', 'LoggerManager', 'logger']