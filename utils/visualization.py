import cv2
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(self):
        self.colors = {
            'skeleton': (0, 255, 0),
            'text': (255, 255, 255),
            'bbox': (0, 0, 255),
            'height_line': (255, 0, 0)
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.line_thickness = 2
        self.font_scale = 0.6

    def draw_results(self,
                     image: np.ndarray,
                     height: float,
                     landmarks: list,
                     confidence: float) -> np.ndarray:
        """Draw height estimation results on image"""
        try:
            result_image = image.copy()

            # Draw skeleton
            result_image = self.draw_skeleton(result_image, landmarks)

            # Draw height measurement
            result_image = self.draw_height_measurement(
                result_image, height, landmarks[0], landmarks[-1]
            )

            # Draw confidence
            self.draw_confidence(result_image, confidence)

            return result_image
        except Exception as e:
            logger.error(f"Error drawing results: {str(e)}")
            return image

    def draw_skeleton(self,
                      image: np.ndarray,
                      landmarks: list) -> np.ndarray:
        """Draw skeleton connections on image"""
        try:
            h, w = image.shape[:2]
            for landmark in landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(image, (x, y), 5, self.colors['skeleton'], -1)
            return image
        except Exception as e:
            logger.error(f"Error drawing skeleton: {str(e)}")
            return image

    def draw_height_measurement(self,
                                image: np.ndarray,
                                height: float,
                                top_point: tuple,
                                bottom_point: tuple) -> np.ndarray:
        """Draw height measurement line and value"""
        try:
            h, w = image.shape[:2]
            start_point = (int(top_point.x * w), int(top_point.y * h))
            end_point = (int(bottom_point.x * w), int(bottom_point.y * h))

            # Draw vertical line
            cv2.line(image, start_point, end_point,
                     self.colors['height_line'], self.line_thickness)

            # Draw height value
            text_pos = (end_point[0] + 10, (start_point[1] + end_point[1]) // 2)
            cv2.putText(image, f'{height:.2f}m', text_pos,
                        self.font, self.font_scale, self.colors['text'], 2)

            return image
        except Exception as e:
            logger.error(f"Error drawing height measurement: {str(e)}")
            return image

    def draw_confidence(self,
                        image: np.ndarray,
                        confidence: float) -> np.ndarray:
        """Draw confidence score"""
        try:
            text = f'Confidence: {confidence:.2f}'
            cv2.putText(image, text, (10, 30), self.font,
                        self.font_scale, self.colors['text'], 2)
            return image
        except Exception as e:
            logger.error(f"Error drawing confidence: {str(e)}")
            return image

    def save_visualization(self,
                           image: np.ndarray,
                           output_path: Path,
                           dpi: int = 300) -> None:
        """Save visualization to file"""
        try:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
            raise

    @staticmethod
    def create_progress_bar(progress: float,
                            width: int = 200,
                            height: int = 20) -> np.ndarray:
        """Create a progress bar visualization"""
        try:
            bar = np.zeros((height, width, 3), dtype=np.uint8)
            filled_width = int(width * progress)

            # Background (gray)
            bar[:, :, :] = [128, 128, 128]

            # Filled portion (green)
            bar[:, :filled_width, :] = [0, 255, 0]

            # Border
            cv2.rectangle(bar, (0, 0), (width - 1, height - 1), (0, 0, 0), 1)

            return bar
        except Exception as e:
            logger.error(f"Error creating progress bar: {str(e)}")
            return np.zeros((height, width, 3), dtype=np.uint8)