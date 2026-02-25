import cv2
import numpy as np
import time
from screeninfo import get_monitors
import pyautogui
import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, Qt
from gaze_tracking.app_overlay import GazeOverlay

from gaze_tracking.landmarks import LandmarkExtractor
from gaze_tracking.iris import IrisTracker
from gaze_tracking.head_pose import HeadPoseEstimator
from gaze_tracking.smoothing import KalmanFilter2D
from gaze_tracking.calibration import CalibrationManager
from gaze_tracking.geometry import GazeGeometry
from ui.calibration_ui import CalibrationUI
from ui.display import DisplayOverlay

class TrackingSession:
    def __init__(self, screen_w, screen_h):
        self.SCREEN_W = screen_w
        self.SCREEN_H = screen_h
        
        # Initialize 3D Geometric modules
        self.extractor = LandmarkExtractor(max_num_faces=1, refine_landmarks=True)
        self.iris_tracker = IrisTracker() # Still used for blink detection
        self.head_pose_est = HeadPoseEstimator(img_w=self.SCREEN_W, img_h=self.SCREEN_H)
        self.geometry_system = GazeGeometry(screen_w=self.SCREEN_W, screen_h=self.SCREEN_H)
        self.calibration_mgr = CalibrationManager(mode='1-point', screen_w=screen_w, screen_h=screen_h)
        
        # UI & Smoothing
        self.kalman_filter = KalmanFilter2D(process_noise=1e-5, measurement_noise=1e-2)
        self.calib_ui = CalibrationUI(screen_w=screen_w, screen_h=screen_h) 
        self.display = DisplayOverlay()
        
        self.state = "CALIBRATION"
        self.cap = cv2.VideoCapture(0)
        
        # OpenCV Window
        self.window_name = "Webcam Tracking View"
        
        self.overlay = None
        self.mouse_controlled = False

    def toggle_mouse(self):
        self.mouse_controlled = not self.mouse_controlled
        print(f"Mouse Control: {'ON' if self.mouse_controlled else 'OFF'}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        vis_frame = frame.copy()

        # 1. Base landmarks
        results = self.extractor.process_frame(frame)
        landmarks = self.extractor.extract_landmarks(results, w, h)
        
        blink = False
        if landmarks is not None:
            # Draw 2D landmarks for viewing
            self.display.draw_landmarks(vis_frame, landmarks[:,:2])
            
            eye_features = self.extractor.get_eye_features(landmarks)
            _, blink = self.iris_tracker.extract_features(eye_features)
            
            # Find robust rigid head orientation using PCA on nose landmarks
            head_center, R_final, nose_points_3d = self.head_pose_est.estimate_pose(landmarks)
            nose_scale = self.geometry_system.compute_scale(nose_points_3d)

            # Extract 3D iris centers
            iris_l = np.mean(eye_features['left_iris'], axis=0)
            iris_r = np.mean(eye_features['right_iris'], axis=0)

            if self.state == "CALIBRATION":
                # Start calibration if pending
                if self.calibration_mgr.state == "PENDING":
                    self.calibration_mgr.start_calibration()
                
                progress, msg = self.calibration_mgr.process_frame(
                    head_center, R_final, nose_scale, iris_l, iris_r, self.geometry_system
                )
                
                # Draw calibration UI
                current_ui_point = self.calibration_mgr.get_current_ui_point()
                screen_frame = self.calib_ui.draw_calibration_screen(current_ui_point, self.calibration_mgr.state, progress)
                cv2.putText(screen_frame, msg, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Calibration", screen_frame)
                cv2.waitKey(1)
                
                if self.calibration_mgr.state == "DONE":
                    cv2.destroyWindow("Calibration")
                    self.state = "TRACKING"
                    if self.overlay: self.overlay.show()

            elif self.state == "TRACKING":
                if not blink:
                    # Scale ratio from baseline calibrated face distance
                    scale_ratio = nose_scale / self.calibration_mgr.calibration_nose_scale
                    
                    # Estimate physical 3D eyeball location
                    sphere_l = head_center + R_final @ (self.calibration_mgr.left_sphere_local_offset * scale_ratio)
                    sphere_r = head_center + R_final @ (self.calibration_mgr.right_sphere_local_offset * scale_ratio)
                    
                    # 3D Gaze vector
                    gaze_dir = self.geometry_system.compute_combined_gaze(iris_l, iris_r, sphere_l, sphere_r)
                    
                    if gaze_dir is not None:
                        raw_x, raw_y = self.geometry_system.get_screen_coordinates(gaze_dir)
                        
                        # 🔥 APPLY MICRO RECALIBRATION
                        # If actual is passed (e.g. from a user click), it will dynamically learn.
                        corrected = self.calibration_mgr.apply_micro_correction((raw_x, raw_y), actual=None)
                        raw_x, raw_y = corrected

                        smoothed_point = self.kalman_filter.update(raw_x, raw_y)
                        
                        gfx_x = max(0, min(self.SCREEN_W, smoothed_point[0]))
                        gfx_y = max(0, min(self.SCREEN_H, smoothed_point[1]))
                        
                        if self.overlay:
                            self.overlay.update_gaze(gfx_x, gfx_y)
                            
                        # Move OS Mouse
                        if self.mouse_controlled:
                            try:
                                pyautogui.moveTo(int(gfx_x), int(gfx_y), _pause=False)
                            except Exception as e:
                                pass
                            
            cv2.circle(vis_frame, (w//2, h//2), 3, (0,255,0), -1) # Center of camera
        
        if self.state == "TRACKING":
            self.display.update_and_draw_fps(vis_frame)
            vis_frame = self.display.draw_ui_panel(vis_frame, self.state, blink)
            
            # Print mouse status on UI
            cv2.putText(vis_frame, f"Mouse [F7/m]: {'ON' if self.mouse_controlled else 'OFF'}", 
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (0, 255, 0) if self.mouse_controlled else (0,0,255), 2)

            cv2.imshow(self.window_name, vis_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.quit_application()
            if key == ord('r'):
                self.trigger_recalibration()
            if key == ord('m'):
                self.toggle_mouse()

    def trigger_recalibration(self):
        if self.state in ["TRACKING"]:
            print("Restarting Calibration...")
            self.calibration_mgr.reset()
            self.kalman_filter.reset()
            self.state = "CALIBRATION"
            if self.overlay: self.overlay.hide()
            
    def quit_application(self):
        self.cap.release()
        cv2.destroyAllWindows()
        QApplication.instance().quit()


def main():
    # Make pyautogui faster
    pyautogui.MINIMUM_DURATION = 0
    pyautogui.MINIMUM_SLEEP = 0
    pyautogui.PAUSE = 0
    
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
    overlay.mouse_toggle_signal.connect(session.toggle_mouse)
    
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
