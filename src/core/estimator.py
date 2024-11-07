import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
from models.pose_model import PoseEstimator, PoseResult
from models.depth_model import DepthEstimator, DepthResult
from utils.image_processing import ImageProcessor
from utils.calibration import Calibrator
from config.model_config import ModelConfigManager

logger = logging.getLogger(__name__)


@dataclass
class HeightEstimationResult:
    height: float
    confidence: float
    pose_result: PoseResult
    depth_result: Optional[DepthResult] = None
    metadata: Optional[Dict[str, Any]] = None


class HeightEstimator:
    def __init__(self, model_config: ModelConfigManager):
        """Initialize height estimator with models and utilities"""
        self.model_config = model_config
        self.image_processor = ImageProcessor()
        self.calibrator = Calibrator()

        # Initialize models
        self.pose_estimator = PoseEstimator(model_config.pose_config)
        self.depth_estimator = DepthEstimator(model_config.depth_config)

        # Calibration parameters
        self.is_calibrated = False
        self.calibration_factor = None

    def calibrate(self,
                  reference_image: np.ndarray,
                  known_height: float,
                  known_distance: float) -> bool:
        """Calibrate the system using a reference image"""
        try:
            # Process reference image
            pose_result = self.pose_estimator.process_image(reference_image)
            if not pose_result:
                raise ValueError("No pose detected in reference image")

            # Calculate calibration factor
            pixel_height = self._calculate_pixel_height(pose_result.landmarks)
            self.calibration_factor = known_height / pixel_height

            # Store calibration data
            self.calibrator.calibrate(
                reference_height=known_height,
                reference_distance=known_distance,
                pixel_height=pixel_height
            )

            self.is_calibrated = True
            logger.info("Calibration completed successfully")
            return True

        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            return False

    def estimate_height(self,
                        image: np.ndarray,
                        distance_from_camera: Optional[float] = None) -> Optional[HeightEstimationResult]:
        """Estimate height from image"""
        try:
            if not self.is_calibrated:
                raise ValueError("System not calibrated")

            # Preprocess image
            processed_image = self.image_processor.preprocess_image(image)

            # Get pose estimation
            pose_result = self.pose_estimator.process_image(processed_image)
            if not pose_result:
                raise ValueError("No pose detected in image")

            # Get depth estimation if distance not provided
            depth_result = None
            if distance_from_camera is None:
                depth_result = self.depth_estimator.estimate_depth(processed_image)
                distance_from_camera = self._estimate_distance(depth_result)

            # Calculate height
            pixel_height = self._calculate_pixel_height(pose_result.landmarks)
            estimated_height = self._calculate_real_height(
                pixel_height, distance_from_camera
            )

            # Calculate confidence
            confidence = self._calculate_confidence(pose_result, depth_result)

            return HeightEstimationResult(
                height=estimated_height,
                confidence=confidence,
                pose_result=pose_result,
                depth_result=depth_result,
                metadata={
                    "distance": distance_from_camera,
                    "pixel_height": pixel_height
                }
            )

        except Exception as e:
            logger.error(f"Height estimation failed: {str(e)}")
            return None

    def _calculate_pixel_height(self, landmarks: list) -> float:
        """Calculate height in pixels from landmarks"""
        try:
            # Get ankle and head landmarks
            left_ankle = landmarks[self.pose_estimator.landmark_names.index('left_ankle')]
            right_ankle = landmarks[self.pose_estimator.landmark_names.index('right_ankle')]
            nose = landmarks[self.pose_estimator.landmark_names.index('nose')]

            # Use the higher ankle point
            ankle_y = min(left_ankle.y, right_ankle.y)

            # Calculate height in pixels
            return abs(ankle_y - nose.y)

        except Exception as e:
            logger.error(f"Error calculating pixel height: {str(e)}")
            raise

    def _estimate_distance(self, depth_result: DepthResult) -> float:
        """Estimate distance from depth map"""
        try:
            # Calculate average depth in person region
            person_depth = np.mean(depth_result.depth_map)
            return float(person_depth)

        except Exception as e:
            logger.error(f"Error estimating distance: {str(e)}")
            raise

    def _calculate_real_height(self,
                               pixel_height: float,
                               distance: float) -> float:
        """Calculate real height using calibration data"""
        try:
            return self.calibrator.get_real_height(pixel_height, distance)

        except Exception as e:
            logger.error(f"Error calculating real height: {str(e)}")
            raise

    def _calculate_confidence(self,
                              pose_result: PoseResult,
                              depth_result: Optional[DepthResult]) -> float:
        """Calculate overall confidence score"""
        try:
            confidence_scores = [pose_result.confidence]

            if depth_result:
                confidence_scores.append(depth_result.confidence)

            return sum(confidence_scores) / len(confidence_scores)

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0