import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PoseLandmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PoseResult:
    landmarks: List[PoseLandmark]
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None


class PoseEstimator:
    def __init__(self, model):
        self.model = model
        self.landmark_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

    def process_image(self, image: np.ndarray) -> Optional[PoseResult]:
        """Process single image and return pose landmarks"""
        try:
            results = self.model.process(image)

            if not results.pose_landmarks:
                return None

            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append(PoseLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=landmark.visibility
                ))

            confidence = self._calculate_confidence(landmarks)
            bbox = self._calculate_bbox(landmarks)

            return PoseResult(
                landmarks=landmarks,
                confidence=confidence,
                bbox=bbox
            )
        except Exception as e:
            logger.error(f"Error processing image for pose estimation: {str(e)}")
            return None

    def _calculate_confidence(self, landmarks: List[PoseLandmark]) -> float:
        """Calculate overall pose confidence score"""
        try:
            visibilities = [lm.visibility for lm in landmarks]
            return sum(visibilities) / len(visibilities)
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0

    def _calculate_bbox(self, landmarks: List[PoseLandmark]) -> Tuple[int, int, int, int]:
        """Calculate bounding box for pose"""
        try:
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]

            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)

            return (x_min, y_min, x_max, y_max)
        except Exception as e:
            logger.error(f"Error calculating bounding box: {str(e)}")
            return (0, 0, 1, 1)

    def get_landmark_coordinates(self,
                                 landmarks: List[PoseLandmark],
                                 landmark_name: str) -> Optional[PoseLandmark]:
        """Get specific landmark coordinates by name"""
        try:
            if landmark_name not in self.landmark_names:
                raise ValueError(f"Invalid landmark name: {landmark_name}")

            idx = self.landmark_names.index(landmark_name)
            return landmarks[idx]
        except Exception as e:
            logger.error(f"Error getting landmark coordinates: {str(e)}")
            return None