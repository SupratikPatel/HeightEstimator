import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']

    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load and validate image from path"""
        try:
            if not isinstance(image_path, Path):
                image_path = Path(image_path)

            if not image_path.suffix.lower() in self.supported_formats:
                raise ValueError(f"Unsupported image format: {image_path.suffix}")

            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            return image
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None

    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """Preprocess image for model input"""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize image
            image_resized = cv2.resize(image_rgb, target_size)

            # Normalize pixel values
            image_normalized = image_resized.astype(np.float32) / 255.0

            return image_normalized
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better detection"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

            # Split channels
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            # Merge channels
            limg = cv2.merge((cl, a, b))

            # Convert back to BGR
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            return enhanced
        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            return image

    def detect_blur(self, image: np.ndarray, threshold: float = 100.0) -> bool:
        """Detect if image is blurry"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            return fm < threshold
        except Exception as e:
            logger.error(f"Error detecting blur: {str(e)}")
            return False

    @staticmethod
    def draw_skeleton(image: np.ndarray, landmarks: list) -> np.ndarray:
        """Draw skeleton on image using landmarks"""
        try:
            image_copy = image.copy()
            height, width = image.shape[:2]

            # Draw points
            for landmark in landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)

            return image_copy
        except Exception as e:
            logger.error(f"Error drawing skeleton: {str(e)}")
            return image