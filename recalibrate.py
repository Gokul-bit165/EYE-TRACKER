import sys
import cv2
import numpy as np
import time
from screeninfo import get_monitors

from gaze_tracking.landmarks import LandmarkExtractor
from gaze_tracking.iris import IrisTracker
from gaze_tracking.head_pose import HeadPoseEstimator
from gaze_tracking.regression_model import GazeRegressionModel
from gaze_tracking.calibration import CalibrationManager
from ui.calibration_ui import CalibrationUI

def main():
    print("Initializing Standalone Calibration Tool...")

    # Handle screen resolution via screeninfo
    try:
        monitor = get_monitors()[0]
        SCREEN_W = monitor.width
        SCREEN_H = monitor.height
    except Exception as e:
        print(f"Could not fetch monitor resolution automatically. Defaulting to 1920x1080. Error: {e}")
        SCREEN_W = 1920
        SCREEN_H = 1080

    # Initialize modules
    extractor = LandmarkExtractor(max_num_faces=1, refine_landmarks=True)
    iris_tracker = IrisTracker()
    head_pose_est = None
    
    regression_model = GazeRegressionModel()
    
    # Use 16-point mode for maximum accuracy
    calibration_mgr = CalibrationManager(mode='16-point', screen_w=SCREEN_W, screen_h=SCREEN_H)
    calib_ui = CalibrationUI(screen_w=SCREEN_W, screen_h=SCREEN_H)
    
    print(f"Resolution set to {SCREEN_W}x{SCREEN_H}")

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create window
    window_name = "Standalone Gaze Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    profile_path = "gaze_profile.pkl"
    calibration_mgr.start_point()
    state = "CALIBRATION"
    validation_start = 0
    training_error = 0.0

    print("Look at the dots to calibrate.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if head_pose_est is None:
            head_pose_est = HeadPoseEstimator(img_w=w, img_h=h)

        results = extractor.process_frame(frame)
        landmarks = extractor.extract_landmarks(results, w, h)
        
        features = None
        blink = False

        if landmarks is not None:
            eye_features = extractor.get_eye_features(landmarks)
            iris_features, blink = iris_tracker.extract_features(eye_features)
            pose_features = head_pose_est.estimate_pose(landmarks)
            features = np.concatenate([iris_features, pose_features])

        if state == "CALIBRATION":
            point = calibration_mgr.get_current_point()
            
            if features is not None and not blink:
                calib_state = calibration_mgr.add_frame_data(features)
                if calib_state == "DONE_POINT":
                    more_points = calibration_mgr.next_point()
                    if not more_points:
                        state = "TRAINING"
                        
            progress = calibration_mgr.get_progress()
            
            screen_frame = calib_ui.draw_calibration_screen(point, calibration_mgr.state, progress)
            cv2.imshow(window_name, screen_frame)

        elif state == "TRAINING":
            msg = "Training Neural Network. Please Wait..."
            screen_frame = np.full((SCREEN_H, SCREEN_W, 3), (15, 15, 15), dtype=np.uint8)
            cv2.putText(screen_frame, msg, (SCREEN_W // 2 - 300, SCREEN_H // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
            cv2.imshow(window_name, screen_frame)
            cv2.waitKey(10)

            X, Y = calibration_mgr.get_dataset()
            print(f"Collected {len(X)} stable points for training.")
            
            success = regression_model.train(X, Y)
            if success:
                regression_model.save_model(profile_path)
                training_error = calibration_mgr.calculate_error(regression_model)
                print(f"Calibration Avg Error: {training_error:.2f} pixels")
                state = "VALIDATING"
                validation_start = time.time()
            else:
                print("Training failed. Restarting.")
                calibration_mgr.reset()
                state = "CALIBRATION"
                
        elif state == "VALIDATING":
            screen_frame = np.full((SCREEN_H, SCREEN_W, 3), (15, 15, 15), dtype=np.uint8)
            cv2.putText(screen_frame, "Calibration Saved!", (SCREEN_W // 2 - 200, SCREEN_H // 2 - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        
            color = (0, 255, 0) if training_error < 150 else (0, 150, 255)
            cv2.putText(screen_frame, f"Average Error: {int(training_error)} px", (SCREEN_W // 2 - 180, SCREEN_H // 2 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                        
            cv2.putText(screen_frame, "You can now run 'python main.py'", (SCREEN_W // 2 - 250, SCREEN_H // 2 + 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                            
            cv2.imshow(window_name, screen_frame)
            
            if time.time() - validation_start > 4.0:
                print("Calibration complete. Exiting standalone tool.")
                break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Calibration aborted.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
