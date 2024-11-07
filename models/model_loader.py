import torch
from pathlib import Path
import mediapipe as mp
from transformers import Pipeline, pipeline
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name: str
    model_type: str
    model_path: Optional[Path] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    config: Dict[str, Any] = None


class ModelLoader:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("models/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    def load_pose_model(self, model_config: Optional[ModelConfig] = None) -> mp.solutions.pose.Pose:
        """Load MediaPipe Pose model"""
        try:
            if not model_config:
                model_config = ModelConfig(
                    model_name="mediapipe_pose",
                    model_type="pose",
                    config={
                        "static_image_mode": False,
                        "model_complexity": 2,
                        "enable_segmentation": True,
                        "min_detection_confidence": 0.5,
                        "min_tracking_confidence": 0.5
                    }
                )

            if model_config.model_name not in self.models:
                self.models[model_config.model_name] = self.mp_pose.Pose(
                    **model_config.config
                )
                logger.info(f"Loaded pose model: {model_config.model_name}")

            return self.models[model_config.model_name]
        except Exception as e:
            logger.error(f"Error loading pose model: {str(e)}")
            raise

    def load_depth_model(self, model_config: Optional[ModelConfig] = None) -> Pipeline:
        """Load depth estimation model"""
        try:
            if not model_config:
                model_config = ModelConfig(
                    model_name="depth_anything",
                    model_type="depth",
                    config={
                        "model_id": "LiheYoung/depth-anything-base",
                        "device": 0 if torch.cuda.is_available() else -1
                    }
                )

            if model_config.model_name not in self.models:
                self.models[model_config.model_name] = pipeline(
                    task="depth-estimation",
                    model=model_config.config["model_id"],
                    device=model_config.config["device"]
                )
                logger.info(f"Loaded depth model: {model_config.model_name}")

            return self.models[model_config.model_name]
        except Exception as e:
            logger.error(f"Error loading depth model: {str(e)}")
            raise


class ModelManager:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.pose_model = None
        self.depth_model = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all required models"""
        try:
            self.pose_model = self.model_loader.load_pose_model()
            self.depth_model = self.model_loader.load_depth_model()
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def get_pose_model(self) -> mp.solutions.pose.Pose:
        """Get pose estimation model"""
        if not self.pose_model:
            self.pose_model = self.model_loader.load_pose_model()
        return self.pose_model

    def get_depth_model(self) -> Pipeline:
        """Get depth estimation model"""
        if not self.depth_model:
            self.depth_model = self.model_loader.load_depth_model()
        return self.depth_model