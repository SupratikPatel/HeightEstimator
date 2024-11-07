import cv2
import numpy as np
from typing import Optional, List, Tuple
import logging
from dataclasses import dataclass
from utils.image_processing import ImageProcessor
from estimator import HeightEstimator, HeightEstimationResult

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    frame: np.ndarray
    height_estimation: Optional[HeightEstimationResult]
    timestamp: float
    frame_number: int
    metadata: dict


class VideoProcessor:
    def __init__(self, height_estimator: HeightEstimator):
        """Initialize video processor"""
        self.height_estimator = height_estimator
        self.image_processor = ImageProcessor()
        self.frame_count = 0
        self.processing_history: List[ProcessingResult] = []

    def process_video(self,
                      video_source: int = 0,
                      output_path: Optional[str] = None,
                      display: bool = True) -> None:
        """Process video stream for height estimation"""
        try:
            cap = cv2.VideoCapture(video_source)

            # Initialize video writer if output path is provided
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                result = self.process_frame(frame)

                if result:
                    # Draw results on frame
                    annotated_frame = self._draw_results(result)

                    # Write frame if output path is provided
                    if writer:
                        writer.write(annotated_frame)

                    # Display frame if requested
                    if display:
                        cv2.imshow('Height Estimation', annotated_frame)

                    # Store processing result
                    self.processing_history.append(result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

    def process_frame(self, frame: np.ndarray) -> Optional[ProcessingResult]:
        """Process single frame"""
        try:
            # Increment frame counter
            self.frame_count += 1

            # Estimate height
            height_result = self.height_estimator.estimate_height(frame)

            if height_result:
                return ProcessingResult(
                    frame=frame,
                    height_estimation=height_result,
                    timestamp=cv2.getTickCount() / cv2.getTickFrequency(),
                    frame_number=self.frame_count,
                    metadata={
                        "processed": True,
                        "success": True
                    }
                )

            return ProcessingResult(
                frame=frame,
                height_estimation=None,
                timestamp=cv2.getTickCount() / cv2.getTickFrequency(),
                frame_number=self.frame_count,
                metadata={
                    "processed": True,
                    "success": False
                }
            )

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return None

    def _draw_results(self, result: ProcessingResult) -> np.ndarray:
        """Draw height estimation results on frame"""
        try:
            frame = result.frame.copy()

            if result.height_estimation:
                # Draw skeleton
                if result.height_estimation.pose_result:
                    frame = self.image_processor.draw_skeleton(
                        frame,
                        result.height_estimation.pose_result.landmarks
                    )

                # Draw height measurement
                height = result.height_estimation.height
                conf = result.height_estimation.confidence

                cv2.putText(
                    frame,
                    f"Height: {height:.2f}m (Conf: {conf:.2f})",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            return frame

        except Exception as e:
            logger.error(f"Error drawing results: {str(e)}")
            return result.frame