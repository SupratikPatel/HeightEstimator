from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import yaml
import torch
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    POSE = "pose"
    DEPTH = "depth"
    COMBINED = "combined"


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # For Apple Silicon


@dataclass
class ModelPaths:
    """Paths configuration for model files"""
    base_dir: Path
    weights_dir: Path = field(init=False)
    config_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)

    def __post_init__(self):
        self.weights_dir = self.base_dir / "weights"
        self.config_dir = self.base_dir / "configs"
        self.cache_dir = self.base_dir / "cache"

        # Create directories
        for directory in [self.weights_dir, self.config_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class PoseModelConfig:
    """Configuration for pose estimation model"""
    model_complexity: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False
    enable_segmentation: bool = True
    smooth_landmarks: bool = True
    smooth_segmentation: bool = True
    refinement_steps: int = 1
    detection_threshold: float = 0.7
    tracking_threshold: float = 0.5
    fps_optimization: bool = True
    enable_face_detection: bool = False
    enable_hand_detection: bool = False
    model_assets: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_complexity": self.model_complexity,
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "static_image_mode": self.static_image_mode,
            "enable_segmentation": self.enable_segmentation,
            "smooth_landmarks": self.smooth_landmarks,
            "smooth_segmentation": self.smooth_segmentation,
            "refinement_steps": self.refinement_steps,
            "detection_threshold": self.detection_threshold,
            "tracking_threshold": self.tracking_threshold,
            "fps_optimization": self.fps_optimization,
            "enable_face_detection": self.enable_face_detection,
            "enable_hand_detection": self.enable_hand_detection,
            "model_assets": self.model_assets
        }


@dataclass
class DepthModelConfig:
    """Configuration for depth estimation model"""
    model_name: str = "LiheYoung/depth-anything-base"
    revision: str = "main"
    trust_remote_code: bool = True
    use_cache: bool = True
    max_memory: Dict[str, str] = field(default_factory=dict)
    precision: str = "float32"
    enable_optimization: bool = True
    batch_size: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    model_parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.max_memory:
            self.max_memory = {
                "cuda:0": "4GiB",
                "cpu": "4GiB"
            }
        if not self.model_parameters:
            self.model_parameters = {
                "pretrained": True,
                "freeze_backbone": True
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "use_cache": self.use_cache,
            "max_memory": self.max_memory,
            "precision": self.precision,
            "enable_optimization": self.enable_optimization,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "model_parameters": self.model_parameters
        }


class ModelConfigManager:
    """Manager class for model configurations"""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.model_paths = ModelPaths(base_path)
        self.pose_config = PoseModelConfig()
        self.depth_config = DepthModelConfig()
        self.device = self._get_optimal_device()

        # Load configurations if they exist
        self._load_configurations()

    def _get_optimal_device(self) -> str:
        """Determine the optimal device for model inference"""
        if torch.cuda.is_available():
            return DeviceType.CUDA.value
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return DeviceType.MPS.value
        return DeviceType.CPU.value

    def _load_configurations(self):
        """Load existing configurations from files"""
        try:
            pose_config_path = self.model_paths.config_dir / "pose_config.yaml"
            depth_config_path = self.model_paths.config_dir / "depth_config.yaml"

            if pose_config_path.exists():
                with open(pose_config_path, 'r') as f:
                    pose_config = yaml.safe_load(f)
                    self._update_pose_config(pose_config)

            if depth_config_path.exists():
                with open(depth_config_path, 'r') as f:
                    depth_config = yaml.safe_load(f)
                    self._update_depth_config(depth_config)

        except Exception as e:
            logger.error(f"Error loading configurations: {str(e)}")

    def _update_pose_config(self, config_dict: Dict[str, Any]):
        """Update pose model configuration"""
        for key, value in config_dict.items():
            if hasattr(self.pose_config, key):
                setattr(self.pose_config, key, value)

    def _update_depth_config(self, config_dict: Dict[str, Any]):
        """Update depth model configuration"""
        for key, value in config_dict.items():
            if hasattr(self.depth_config, key):
                setattr(self.depth_config, key, value)

    def save_configurations(self):
        """Save current configurations to files"""
        try:
            pose_config_path = self.model_paths.config_dir / "pose_config.yaml"
            depth_config_path = self.model_paths.config_dir / "depth_config.yaml"

            with open(pose_config_path, 'w') as f:
                yaml.dump(self.pose_config.to_dict(), f)

            with open(depth_config_path, 'w') as f:
                yaml.dump(self.depth_config.to_dict(), f)

        except Exception as e:
            logger.error(f"Error saving configurations: {str(e)}")

    def get_model_path(self, model_type: ModelType) -> Path:
        """Get path for model weights"""
        return self.model_paths.weights_dir / f"{model_type.value}_model.pth"

    def get_device_config(self, available_memory: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get device configuration based on available memory"""
        if available_memory:
            return available_memory
        return self.depth_config.max_memory

    def optimize_for_device(self):
        """Optimize configurations for current device"""
        if self.device == DeviceType.CPU.value:
            # Optimize for CPU
            self.pose_config.model_complexity = 1
            self.pose_config.fps_optimization = True
            self.depth_config.batch_size = 1
            self.depth_config.num_workers = 1
            self.depth_config.precision = "float32"
        else:
            # Optimize for GPU/MPS
            self.pose_config.model_complexity = 2
            self.depth_config.batch_size = 4
            self.depth_config.num_workers = 2
            self.depth_config.precision = "float16"

    def get_model_config(self, model_type: ModelType) -> Dict[str, Any]:
        """Get configuration for specific model type"""
        if model_type == ModelType.POSE:
            return self.pose_config.to_dict()
        elif model_type == ModelType.DEPTH:
            return self.depth_config.to_dict()
        else:
            return {
                "pose": self.pose_config.to_dict(),
                "depth": self.depth_config.to_dict()
            }

    def validate_configurations(self) -> List[str]:
        """Validate current configurations"""
        validation_errors = []

        # Validate pose config
        if self.pose_config.model_complexity not in [0, 1, 2]:
            validation_errors.append("Invalid model complexity")
        if not 0 <= self.pose_config.min_detection_confidence <= 1:
            validation_errors.append("Invalid detection confidence range")

        # Validate depth config
        if self.depth_config.batch_size < 1:
            validation_errors.append("Invalid batch size")
        if self.depth_config.num_workers < 0:
            validation_errors.append("Invalid number of workers")

        return validation_errors

    def get_performance_profile(self) -> Dict[str, Any]:
        """Get performance profile for current configuration"""
        return {
            "device": self.device,
            "pose_model": {
                "complexity": self.pose_config.model_complexity,
                "optimization": self.pose_config.fps_optimization
            },
            "depth_model": {
                "batch_size": self.depth_config.batch_size,
                "precision": self.depth_config.precision,
                "optimization": self.depth_config.enable_optimization
            }
        }