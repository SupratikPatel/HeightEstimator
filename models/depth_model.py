import numpy as np
from typing import Optional, Tuple, Dict
import logging
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    depth_map: np.ndarray
    confidence: float
    min_depth: float
    max_depth: float


class DepthEstimator:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def estimate_depth(self, image: np.ndarray) -> Optional[DepthResult]:
        """Estimate depth from single image"""
        try:
            # Get depth estimation from model
            depth_output = self.model(image)
            depth_map = depth_output["depth"]

            # Calculate confidence and depth range
            confidence = self._calculate_confidence(depth_map)
            min_depth = float(depth_map.min())
            max_depth = float(depth_map.max())

            return DepthResult(
                depth_map=depth_map,
                confidence=confidence,
                min_depth=min_depth,
                max_depth=max_depth
            )
        except Exception as e:
            logger.error(f"Error estimating depth: {str(e)}")
            return None

    def _calculate_confidence(self, depth_map: np.ndarray) -> float:
        """Calculate confidence score for depth estimation"""
        try:
            # Calculate confidence based on depth consistency
            gradient_x = np.gradient(depth_map, axis=1)
            gradient_y = np.gradient(depth_map, axis=0)

            consistency = 1.0 - np.mean(np.abs(gradient_x) + np.abs(gradient_y))
            return float(consistency)
        except Exception as e:
            logger.error(f"Error calculating depth confidence: {str(e)}")
            return 0.0

    def get_depth_at_point(self,
                           depth_result: DepthResult,
                           point: Tuple[int, int]) -> Optional[float]:
        """Get depth value at specific point"""
        try:
            x, y = point
            if x < 0 or y < 0 or x >= depth_result.depth_map.shape[1] or y >= depth_result.depth_map.shape[0]:
                raise ValueError("Point coordinates out of bounds")

            return float(depth_result.depth_map[y, x])
        except Exception as e:
            logger.error(f"Error getting depth at point: {str(e)}")
            return None

    def normalize_depth_map(self, depth_result: DepthResult) -> np.ndarray:
        """Normalize depth map for visualization"""
        try:
            depth_map = depth_result.depth_map
            normalized = (depth_map - depth_result.min_depth) / (depth_result.max_depth - depth_result.min_depth)
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing depth map: {str(e)}")
            return np.zeros_like(depth_result.depth_map)