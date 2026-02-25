import numpy as np
from scipy.spatial.transform import Rotation as Rscipy

class HeadPoseEstimator:
    def __init__(self, img_w: int, img_h: int):
        self.img_w = img_w
        self.img_h = img_h
        
        # Nose-only landmark indices for highly stable PCA tracking (avoids mouth/jaw deformation)
        self.nose_indices = [
            4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
            461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
            3, 248
        ]
        
        # Reference matrix to fix coordinate flipping issue
        self.R_ref_nose = [None]
        
    def estimate_pose(self, landmarks_3d: np.ndarray) -> tuple:
        """
        Estimates the head pose rotation matrix and head center using PCA on nose landmarks.
        Returns:
            center (np.ndarray): 3D center of the head/nose points
            R_final (np.ndarray): 3x3 stable Rotation matrix
            points_3d (np.ndarray): The specific 3D points used for PCA, useful for scaling
        """
        # Extract 3D positions of selected landmarks
        points_3d = landmarks_3d[self.nose_indices]
        
        # Compute the average position as the center of this substructure
        center = np.mean(points_3d, axis=0)
        
        # PCA-based orientation: Compute eigenvectors of the covariance matrix
        centered = points_3d - center
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvecs = eigvecs[:, np.argsort(-eigvals)]  # Sort by descending eigenvalue (major axes)

        # Ensure the orientation matrix is right-handed
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 2] *= -1

        # Convert to Euler angles and re-construct rotation matrix 
        r = Rscipy.from_matrix(eigvecs)
        roll, pitch, yaw = r.as_euler('zyx', degrees=False)
        
        # Construct final R matrix
        R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

        # Stabilize rotation with reference matrix to avoid flipping during eigenvector sign change
        if self.R_ref_nose[0] is None:
            self.R_ref_nose[0] = R_final.copy()
        else:
            R_ref = self.R_ref_nose[0]
            for i in range(3):
                if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                    R_final[:, i] *= -1

        return center, R_final, points_3d
