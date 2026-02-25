import cv2
import numpy as np
import time
from screeninfo import get_monitors

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, Qt
from gaze_tracking.app_overlay import GazeOverlay
import time
from screeninfo import get_monitors

from gaze_tracking.landmarks import LandmarkExtractor
from gaze_tracking.iris import IrisTracker
from gaze_tracking.head_pose import HeadPoseEstimator
from gaze_tracking.smoothing import KalmanFilter2D
from gaze_tracking.regression_model import GazeRegressionModel
from gaze_tracking.calibration import CalibrationManager
from ui.calibration_ui import CalibrationUI

class TrackingSession:
    def __init__(self, screen_w, screen_h):
        self.SCREEN_W = screen_w
        self.SCREEN_H = screen_h
        
        # Initialize modules
        self.extractor = LandmarkExtractor(max_num_faces=1, refine_landmarks=True)
        self.iris_tracker = IrisTracker()
        self.head_pose_est = None
        
        # Regression & Filtering
        self.regression_model = GazeRegressionModel()
        self.kalman_filter = KalmanFilter2D(process_noise=1e-4, measurement_noise=1e-1, ema_alpha=0.5)
        
        # Make sure to set to 16-point if desired
        self.calibration_mgr = CalibrationManager(mode='16-point', screen_w=screen_w, screen_h=screen_h)
        self.calib_ui = CalibrationUI(screen_w=screen_w, screen_h=screen_h)
        
        self.state = "CALIBRATION"
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            sys.exit(1)
            
        self.window_name = "Gaze Calibration"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.profile_path = "gaze_profile.pkl"
        self.training_error = 0.0
        self.validation_start = 0
        
        # Overlay reference (set later)
        self.overlay = None

        if self.regression_model.load_model(self.profile_path):
            self.state = "TRACKING"
            cv2.destroyWindow(self.window_name)
            print("Loaded existing calibration profile.")
        else:
            self.state = "CALIBRATION"
            self.calibration_mgr.start_point()
            print("No profile found. Starting calibration...")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if self.head_pose_est is None:
            self.head_pose_est = HeadPoseEstimator(img_w=w, img_h=h)

        results = self.extractor.process_frame(frame)
        landmarks = self.extractor.extract_landmarks(results, w, h)
        
        features = None
        blink = False

        if landmarks is not None:
            eye_features = self.extractor.get_eye_features(landmarks)
            iris_features, blink = self.iris_tracker.extract_features(eye_features)
            pose_features = self.head_pose_est.estimate_pose(landmarks)
            features = np.concatenate([iris_features, pose_features])

        # State Machine tick
        self.handle_state(frame, features, blink, results, w, h)

    def handle_state(self, frame, features, blink, results, w, h):
        if self.state == "CALIBRATION":
            point = self.calibration_mgr.get_current_point()
            
            if features is not None and not blink:
                calib_state = self.calibration_mgr.add_frame_data(features)
                if calib_state == "DONE_POINT":
                    more_points = self.calibration_mgr.next_point()
                    if not more_points:
                        self.state = "TRAINING"
                        
            progress = self.calibration_mgr.get_progress()
            
            screen_frame = self.calib_ui.draw_calibration_screen(point, self.calibration_mgr.state, progress)
            cv2.imshow(self.window_name, screen_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.quit_application()
            if key == ord('r'):
                self.trigger_recalibration()

        elif self.state == "TRAINING":
            msg = "Training Neural Network. Please Wait..."
            screen_frame = np.full((self.SCREEN_H, self.SCREEN_W, 3), (15, 15, 15), dtype=np.uint8)
            cv2.putText(screen_frame, msg, (self.SCREEN_W // 2 - 250, self.SCREEN_H // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            cv2.imshow(self.window_name, screen_frame)
            cv2.waitKey(10)

            X, Y = self.calibration_mgr.get_dataset()
            print(f"Collected {len(X)} stable points for training.")
            
            success = self.regression_model.train(X, Y)
            if success:
                self.regression_model.save_model(self.profile_path)
                self.training_error = self.calibration_mgr.calculate_error(self.regression_model)
                print(f"Calibration Avg Error: {self.training_error:.2f} pixels")
                self.state = "VALIDATING"
                self.validation_start = time.time()
            else:
                self.calibration_mgr.reset()
                self.state = "CALIBRATION"
                
        elif self.state == "VALIDATING":
            screen_frame = np.full((self.SCREEN_H, self.SCREEN_W, 3), (15, 15, 15), dtype=np.uint8)
            cv2.putText(screen_frame, "Calibration Complete!", (self.SCREEN_W // 2 - 200, self.SCREEN_H // 2 - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        
            color = (0, 255, 0) if self.training_error < 150 else (0, 150, 255)
            cv2.putText(screen_frame, f"Average Error: {int(self.training_error)} px", (self.SCREEN_W // 2 - 180, self.SCREEN_H // 2 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                            
            cv2.imshow(self.window_name, screen_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.quit_application()
            if key == ord('r'):
                self.trigger_recalibration()
            
            if time.time() - self.validation_start > 3.0:
                cv2.destroyWindow(self.window_name)
                self.state = "TRACKING"
                if self.overlay: self.overlay.show()

        elif self.state == "TRACKING":
            if features is not None and not blink:
                pred_point = self.regression_model.predict(features)
                if pred_point:
                    smoothed_point = self.kalman_filter.update(pred_point[0], pred_point[1])
                    gfx_x = max(0, min(self.SCREEN_W, smoothed_point[0]))
                    gfx_y = max(0, min(self.SCREEN_H, smoothed_point[1]))
                    
                    if self.overlay:
                        self.overlay.update_gaze(gfx_x, gfx_y)
            else:
                self.kalman_filter.reset()

    def trigger_recalibration(self):
        """Called via PyQt signal when 'r' is pressed."""
        if self.state in ["TRACKING", "VALIDATING"]:
            print("Restarting Calibration...")
            self.calibration_mgr.reset()
            self.kalman_filter.reset()
            self.state = "CALIBRATION"
            if self.overlay: self.overlay.hide()
            
    def quit_application(self):
        """Called via PyQt signal when 'q' is pressed."""
        print("Safely quitting...")
        self.cap.release()
        cv2.destroyAllWindows()
        QApplication.instance().quit()

def main():
    import sys
    app = QApplication(sys.argv)
    
    # Handle screen resolution
    monitor = get_monitors()[0]
    SCREEN_W = monitor.width
    SCREEN_H = monitor.height
    
    # Setup PyQt transparent overlay
    overlay = GazeOverlay(SCREEN_W, SCREEN_H)
    
    # Setup tracking backend
    session = TrackingSession(SCREEN_W, SCREEN_H)
    session.overlay = overlay
    
    # Connect Overlay Keypress Signals to the Session
    overlay.recalibrate_signal.connect(session.trigger_recalibration)
    overlay.quit_signal.connect(session.quit_application)
    
    if session.state == "TRACKING":
        overlay.show()
    else:
        overlay.hide()
        
    # Main loop timer triggered by PyQt event loop
    timer = QTimer()
    timer.timeout.connect(session.update_frame)
    timer.start(16) # ~60 FPS polling
    
    print("System started. Press CTRL+C in terminal to exit.")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
