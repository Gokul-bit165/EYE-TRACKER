import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
import urllib.request
from typing import Optional, Tuple, List, Dict, Any

class LandmarkExtractor:
    def __init__(self, max_num_faces=1, refine_landmarks=True):
        # MediaPipe modern API requires the model bundle file
        model_path = 'face_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading face_landmarker.task model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Define indices for left and right eye, and iris
        self.LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_IRIS_INDICES = [474, 475, 476, 477]
        self.RIGHT_IRIS_INDICES = [469, 470, 471, 472]
        
    def process_frame(self, frame: np.ndarray) -> Optional[Any]:
        """
        Processes a BGR image and returns the face mesh results.
        Includes low-light enhancement (CLAHE) to reduce jitter under poor lighting.
        """
        # Low light enhancement using CLAHE on L channel of LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to mediapipe image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = self.landmarker.detect(mp_image)
        return results

    def extract_landmarks(self, results, img_w: int, img_h: int) -> Optional[np.ndarray]:
        """
        Extracts 2D landmarks from the results.
        """
        if not results or not results.face_landmarks:
            return None
        
        # Scale normalized landmarks back to image dimensions
        mesh_points = np.array(
            [[p.x * img_w, p.y * img_h] for p in results.face_landmarks[0]]
        )
        return mesh_points

    def get_eye_features(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Returns dictionaries of left/right eye and iris coordinates.
        """
        return {
            'left_eye': landmarks[self.LEFT_EYE_INDICES],
            'right_eye': landmarks[self.RIGHT_EYE_INDICES],
            'left_iris': landmarks[self.LEFT_IRIS_INDICES],
            'right_iris': landmarks[self.RIGHT_IRIS_INDICES]
        }
