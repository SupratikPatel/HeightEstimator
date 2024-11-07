from pathlib import Path
from typing import Optional, Dict, Any
import logging
from estimator import HeightEstimator
from processor import VideoProcessor
from config.model_config import ModelConfigManager
from utils.calibration import Calibrator

logger = logging.getLogger(__name__)


class HeightEstimationPipeline:
    def __init__(self, config: Dict[str, Any]):
        """Initialize height estimation pipeline"""
        self.config = config
        self.model_config = ModelConfigManager(Path(config.get("model_dir", "models")))

        # Initialize components
        self.height_estimator = HeightEstimator(self.model_config)
        self.video_processor = VideoProcessor(self.height_estimator)
        self.calibrator = Calibrator()

        # Load calibration if available
        self._load_calibration()

    def _load_calibration(self) -> None:
        """Load calibration data if available"""
        try:
            if self.calibrator.calibration_data:
                self.height_estimator.calibration_factor = (
                    self.calibrator.calibration_data.calibration_factor
                )
                self.height_estimator.is_calibrated = True
                logger.info("Calibration data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading calibration: {str(e)}")

    def calibrate(self,
                  reference_image_path: str,
                  known_height: float,
                  known_distance: float) -> bool:
        """Calibrate the pipeline"""
        try:
            # Load reference image
            reference_image = cv2.imread(reference_image_path)
            if reference_image is None:
                raise ValueError(f"Could not load reference image: {reference_image_path}")

            # Perform calibration
            return self.height_estimator.calibrate(
                reference_image,
                known_height,
                known_distance
            )
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            return False

    def process_image(self,
                      image_path: str,
                      distance_from_camera: Optional[float] = None) -> Dict[str, Any]:
        """Process single image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Estimate height
            result = self.height_estimator.estimate_height(
                image,
                distance_from_camera
            )

            if result:
                return {
                    "height": result.height,
                    "confidence": result.confidence,
                    "metadata": result.metadata
                }

            return {"error": "Height estimation failed"}

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {"error": str(e)}

    def process_video(self,
                      video_source: str,
                      output_path: Optional[str] = None,
                      display: bool = True) -> Dict[str, Any]:
        """Process video stream"""
        try:
            self.video_processor.process_video(
                video_source,
                output_path,
                display
            )

            return {
                "frames_processed": self.video_processor.frame_count,
                "processing_history": len(self.video_processor.processing_history)
            }

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return {"error": str(e)}

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        try:
            history = self.video_processor.processing_history

            if not history:
                return {"error": "No processing history available"}

            successful_frames = sum(
                1 for result in history
                if result.height_estimation is not None
            )

            return {
                "total_frames": len(history),
                "successful_frames": successful_frames,
                "success_rate": successful_frames / len(history),
                "average_confidence": sum(
                    result.height_estimation.confidence
                    for result in history
                    if result.height_estimation is not None
                ) / successful_frames if successful_frames > 0 else 0
            }

        except Exception as e:
            logger.error(f"Error getting processing stats: {str(e)}")
            return {"error": str(e)}