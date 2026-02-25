import cv2
import numpy as np
from typing import Tuple

class HeadPoseEstimator:
    def __init__(self, img_w: int, img_h: int):
        self.img_w = img_w
        self.img_h = img_h
        
        # 3D generic face model points (Nose, Chin, Left Eye, Right Eye, Left Mouth, Right Mouth)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # Corresponding MediaPipe landmark indices
        self.landmark_indices = [1, 152, 226, 446, 57, 287]
        
        # Camera matrix approximation
        focal_length = self.img_w
        self.camera_matrix = np.array([
            [focal_length, 0, self.img_w / 2],
            [0, focal_length, self.img_h / 2],
            [0, 0, 1]
        ], dtype="double")
        
        self.dist_coeffs = np.zeros((4, 1))
        
        # EMA for smoothing the rotation vector to reduce head jitter
        self.ema_rotation = None
        self.ema_alpha = 0.5

    def estimate_pose(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Estimates the head pose rotation vector (pitch, yaw, roll approximate encoded).
        Returns a smoothed 3-element vector.
        """
        # Extract the 6 2D points from landmarks
        image_points = landmarks[self.landmark_indices]
        
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, 
            image_points, 
            self.camera_matrix, 
            self.dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # We only really care about the rotation vector for gaze compensation
            rot_vec = rotation_vector.flatten()
            
            # Smooth out head pose jitter with EMA
            if self.ema_rotation is None:
                self.ema_rotation = rot_vec
            else:
                self.ema_rotation = self.ema_alpha * rot_vec + (1 - self.ema_alpha) * self.ema_rotation
                
            return self.ema_rotation
        else:
            if self.ema_rotation is not None:
                return self.ema_rotation
            return np.zeros(3)
