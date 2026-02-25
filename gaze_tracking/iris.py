import numpy as np
from typing import Tuple

class IrisTracker:
    def __init__(self):
        # Thresholds can be tuned
        self.EAR_THRESHOLD = 0.20
        
    def calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        Calculate the Eye Aspect Ratio (EAR) to detect blinks.
        Typically eye_points contains points outlining the eye polygon.
        We approximate using width vs height.
        """
        # A simple approximation for EAR:
        # vertical distances / horizontal distance
        # Using bounding box width/height as a quick proxy if specific indices aren't strictly ordered,
        # but MediaPipe gives them in contour order. So we can use max width vs max height.
        min_x = np.min(eye_points[:, 0])
        max_x = np.max(eye_points[:, 0])
        min_y = np.min(eye_points[:, 1])
        max_y = np.max(eye_points[:, 1])
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0:
            return 0.0
            
        return height / width

    def is_blinking(self, left_eye: np.ndarray, right_eye: np.ndarray) -> bool:
        """
        Returns True if the user is currently blinking based on EAR.
        """
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear < self.EAR_THRESHOLD

    def get_iris_center(self, iris_points: np.ndarray) -> Tuple[float, float]:
        """
        Returns the (x, y) center of the iris by averaging the landmarks.
        """
        center = np.mean(iris_points, axis=0)
        return center[0], center[1]

    def normalize_iris_position(self, eye_points: np.ndarray, iris_center: Tuple[float, float]) -> Tuple[float, float]:
        """
        Normalizes the iris center relative to the eye bounding box, resulting in a value between [0, 1].
        """
        cx, cy = iris_center
        min_x = np.min(eye_points[:, 0])
        max_x = np.max(eye_points[:, 0])
        min_y = np.min(eye_points[:, 1])
        max_y = np.max(eye_points[:, 1])
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 or height == 0:
            return 0.5, 0.5 # Default center
            
        nx = (cx - min_x) / width
        ny = (cy - min_y) / height
        
        return nx, ny

    def extract_features(self, eye_features: dict) -> Tuple[np.ndarray, bool]:
        """
        High level method to extract the final normalized iris vector + blink status.
        Output vector dim: 4 (left_nx, left_ny, right_nx, right_ny)
        """
        left_eye = eye_features['left_eye']
        right_eye = eye_features['right_eye']
        left_iris = eye_features['left_iris']
        right_iris = eye_features['right_iris']
        
        blink = self.is_blinking(left_eye, right_eye)
        
        left_center = self.get_iris_center(left_iris)
        right_center = self.get_iris_center(right_iris)
        
        left_nx, left_ny = self.normalize_iris_position(left_eye, left_center)
        right_nx, right_ny = self.normalize_iris_position(right_eye, right_center)
        
        # Combine into feature vector
        features = np.array([left_nx, left_ny, right_nx, right_ny])
        return features, blink
