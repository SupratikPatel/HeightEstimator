import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
import yaml

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
CONFIG_DIR = BASE_DIR / "config"

# Create necessary directories
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """Application configuration settings"""
    DEBUG: bool = False
    TESTING: bool = False
    ENV: str = os.getenv("APP_ENV", "development")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    VERSION: str = "1.0.0"
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB


@dataclass
class ModelConfig:
    """Model configuration settings"""
    POSE_MODEL_NAME: str = "mediapipe_pose"
    DEPTH_MODEL_NAME: str = "depth_anything"
    USE_GPU: bool = True
    BATCH_SIZE: int = 1
    CONFIDENCE_THRESHOLD: float = 0.5
    MAX_DETECTION_INSTANCES: int = 10
    MODEL_PRECISION: str = "float32"
    ENABLE_OPTIMIZATION: bool = True


@dataclass
class ProcessingConfig:
    """Image processing configuration"""
    MAX_IMAGE_SIZE: tuple = (1920, 1080)
    MIN_IMAGE_SIZE: tuple = (640, 480)
    SUPPORTED_FORMATS: list = None
    COMPRESSION_QUALITY: int = 95
    COLOR_FORMAT: str = "RGB"
    NORMALIZE_PIXELS: bool = True
    ENABLE_AUGMENTATION: bool = False

    def __post_init__(self):
        if self.SUPPORTED_FORMATS is None:
            self.SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']


@dataclass
class APIConfig:
    """API configuration settings"""
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    TIMEOUT: int = 60
    RATE_LIMIT: int = 100
    ENABLE_CORS: bool = True
    ALLOWED_ORIGINS: list = None

    def __post_init__(self):
        if self.ALLOWED_ORIGINS is None:
            self.ALLOWED_ORIGINS = ["*"]


class Config:
    """Main configuration class"""

    def __init__(self):
        self.app = AppConfig()
        self.model = ModelConfig()
        self.processing = ProcessingConfig()
        self.api = APIConfig()

        # Load environment-specific settings
        self._load_env_settings()

        # Load custom configurations
        self._load_custom_config()

    def _load_env_settings(self):
        """Load environment-specific settings"""
        env = os.getenv("APP_ENV", "development")

        if env == "production":
            self.app.DEBUG = False
            self.app.TESTING = False
            self.model.ENABLE_OPTIMIZATION = True
        elif env == "testing":
            self.app.DEBUG = True
            self.app.TESTING = True
            self.model.ENABLE_OPTIMIZATION = False
        else:  # development
            self.app.DEBUG = True
            self.app.TESTING = False
            self.model.ENABLE_OPTIMIZATION = False

    def _load_custom_config(self):
        """Load custom configuration from YAML file"""
        config_file = CONFIG_DIR / "config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                custom_config = yaml.safe_load(f)
                self._update_config(custom_config)

    def _update_config(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in config_dict.items():
            if hasattr(self, section.lower()):
                section_obj = getattr(self, section.lower())
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "app": self.app.__dict__,
            "model": self.model.__dict__,
            "processing": self.processing.__dict__,
            "api": self.api.__dict__
        }


# Create global config instance
config = Config()