import cv2
import numpy as np
from typing import List, Tuple
import os

class VideoProcessor:
    def __init__(self, interval_seconds: int = 5):
        """Initialize the video processor.
        
        Args:
            interval_seconds (int): Interval in seconds between keyframes
        """
        self.interval_seconds = interval_seconds

    def extract_keyframes(self, video_path: str) -> List[Tuple[float, np.ndarray]]:
        """Extract keyframes from video at fixed intervals.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            List[Tuple[float, np.ndarray]]: List of (timestamp, frame) tuples
        """
        keyframes = []
        
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * self.interval_seconds)
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at interval
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    keyframes.append((timestamp, frame_rgb))
                
                frame_count += 1
            
            cap.release()
            return keyframes
            
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")

    def save_keyframes(self, keyframes: List[Tuple[float, np.ndarray]], output_dir: str):
        """Save keyframes as images.
        
        Args:
            keyframes (List[Tuple[float, np.ndarray]]): List of (timestamp, frame) tuples
            output_dir (str): Directory to save keyframe images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for timestamp, frame in keyframes:
            timestamp_str = f"{timestamp:.2f}".replace(".", "_")
            output_path = os.path.join(output_dir, f"frame_{timestamp_str}.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) 