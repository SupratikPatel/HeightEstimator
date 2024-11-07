import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    reference_height: float
    reference_distance: float
    calibration_factor: float
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray


class Calibrator:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/calibration.json")
        self.calibration_data = None
        self._load_calibration()

    def _load_calibration(self) -> None:
        """Load calibration data from config file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                self.calibration_data = CalibrationData(
                    reference_height=data['reference_height'],
                    reference_distance=data['reference_distance'],
                    calibration_factor=data['calibration_factor'],
                    camera_matrix=np.array(data['camera_matrix']),
                    distortion_coeffs=np.array(data['distortion_coeffs'])
                )
        except Exception as e:
            logger.error(f"Error loading calibration data: {str(e)}")

    def save_calibration(self) -> None:
        """Save calibration data to config file"""
        try:
            if self.calibration_data:
                data = {
                    'reference_height': self.calibration_data.reference_height,
                    'reference_distance': self.calibration_data.reference_distance,
                    'calibration_factor': self.calibration_data.calibration_factor,
                    'camera_matrix': self.calibration_data.camera_matrix.tolist(),
                    'distortion_coeffs': self.calibration_data.distortion_coeffs.tolist()
                }
                with open(self.config_path, 'w') as f:
                    json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving calibration data: {str(e)}")

    def calibrate(self,
                  reference_height: float,
                  reference_distance: float,
                  pixel_height: float) -> float:
        """Calculate calibration factor"""
        try:
            calibration_factor = reference_height / pixel_height
            self.calibration_data = CalibrationData(
                reference_height=reference_height,
                reference_distance=reference_distance,
                calibration_factor=calibration_factor,
                camera_matrix=np.eye(3),
                distortion_coeffs=np.zeros(5)
            )
            self.save_calibration()
            return calibration_factor
        except Exception as e:
            logger.error(f"Error during calibration: {str(e)}")
            raise

    def get_real_height(self, pixel_height: float, distance: float) -> float:
        """Calculate real height using calibration data"""
        try:
            if not self.calibration_data:
                raise ValueError("Calibration data not available")

            distance_ratio = distance / self.calibration_data.reference_distance
            real_height = pixel_height * self.calibration_data.calibration_factor * distance_ratio
            return real_height
        except Exception as e:
            logger.error(f"Error calculating real height: {str(e)}")
            raise