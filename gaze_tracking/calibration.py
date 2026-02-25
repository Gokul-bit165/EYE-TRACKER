import numpy as np
import time

class CalibrationManager:
    def __init__(self, mode='5-point', screen_w=1920, screen_h=1080):
        self.screen_w = screen_w
        self.screen_h = screen_h
        
        self.state = "PENDING"
        self.step_start_time = 0.0
        
        self.left_sphere_local_offset = None
        self.right_sphere_local_offset = None
        self.calibration_nose_scale = 1.0
        
        self.points = [
            ("CENTER", (screen_w // 2, screen_h // 2)),
            ("TOP_LEFT", (100, 100)),
            ("TOP_RIGHT", (screen_w - 100, 100)),
            ("BOTTOM_RIGHT", (screen_w - 100, screen_h - 100)),
            ("BOTTOM_LEFT", (100, screen_h - 100))
        ]
        
        self.current_point_idx = 0
        self.collected_gaze_data = {}
        self.vector_buffer = []
        
        # 🔥 NEW: Adaptive Tracking Data
        self.confidence_scores = []
        self.global_offset = np.array([0.0, 0.0])
        self.drift_history = []

    def reset(self):
        self.__init__(mode='5-point', screen_w=self.screen_w, screen_h=self.screen_h)

    def start_calibration(self):
        self.state = "LOCK_EYES"
        self.step_start_time = time.time()
        self.vector_buffer = []

    def get_current_ui_point(self):
        if self.current_point_idx < len(self.points):
            return self.points[self.current_point_idx][1]
        return (self.screen_w // 2, self.screen_h // 2)

    # 🔥 OUTLIER FILTER
    def filter_outliers(self, vectors):
        if len(vectors) < 5:
            return vectors
        
        mean = np.mean(vectors, axis=0)
        std = np.std(vectors, axis=0)
        
        filtered = []
        for v in vectors:
            if np.all(np.abs(v - mean) < 2 * std):
                filtered.append(v)
        return filtered

    # 🔥 CONFIDENCE SCORE
    def compute_confidence(self, vectors):
        if len(vectors) < 2:
            return 0.0
        variance = np.var(vectors)
        return 1.0 / (variance + 1e-6)

    def process_frame(self, head_center, R_final, nose_scale, iris_l, iris_r, geometry_system):
        elapsed = time.time() - self.step_start_time
        
        if self.state == "LOCK_EYES":
            if elapsed > 3.0:
                base_radius = 20.0
                camera_dir_local = R_final.T @ np.array([0, 0, 1])
                
                self.left_sphere_local_offset = R_final.T @ (iris_l - head_center) + base_radius * camera_dir_local
                self.right_sphere_local_offset = R_final.T @ (iris_r - head_center) + base_radius * camera_dir_local
                self.calibration_nose_scale = nose_scale
                
                self.state = "WAIT_POINT"
                self.step_start_time = time.time()
                return 1.0, "Eyes Locked!"
            
            return min(1.0, elapsed / 3.0), "Look at CAMERA..."

        elif self.state == "WAIT_POINT":
            if elapsed > 1.0:
                self.state = "CAPTURE_POINT"
                self.step_start_time = time.time()
                self.vector_buffer = []
            return 1.0, "Focus on dot..."

        elif self.state == "CAPTURE_POINT":
            if elapsed > 0.5:
                scale_ratio = nose_scale / self.calibration_nose_scale
                
                sphere_l = head_center + R_final @ (self.left_sphere_local_offset * scale_ratio)
                sphere_r = head_center + R_final @ (self.right_sphere_local_offset * scale_ratio)
                
                gaze_dir = geometry_system.compute_combined_gaze(iris_l, iris_r, sphere_l, sphere_r)
                
                if gaze_dir is not None:
                    self.vector_buffer.append(gaze_dir)

            if elapsed > 2.0:
                filtered = self.filter_outliers(self.vector_buffer)
                
                if len(filtered) > 0:
                    avg_gaze = np.mean(filtered, axis=0)
                    confidence = self.compute_confidence(filtered)
                    
                    point_name = self.points[self.current_point_idx][0]
                    self.collected_gaze_data[point_name] = avg_gaze
                    self.confidence_scores.append(confidence)
                    
                    self.current_point_idx += 1
                    
                    if self.current_point_idx >= len(self.points):
                        yaw_off, pitch_off, y_fov, p_fov = geometry_system.calibrate_fov(
                            self.collected_gaze_data["CENTER"],
                            self.collected_gaze_data["TOP_LEFT"],
                            self.collected_gaze_data["TOP_RIGHT"],
                            self.collected_gaze_data["BOTTOM_RIGHT"],
                            self.collected_gaze_data["BOTTOM_LEFT"]
                        )
                        
                        print(f"[CALIB DONE] Yaw:{yaw_off:.2f} Pitch:{pitch_off:.2f} (Confidence: {np.mean(self.confidence_scores):.2f})")
                        self.state = "DONE"
                    else:
                        self.state = "WAIT_POINT"
                        self.step_start_time = time.time()
                
                return 1.0, "Captured"
            
            return min(1.0, elapsed / 2.0), "Capturing..."

        elif self.state == "DONE":
            return 1.0, "Calibration Complete"

        return 0.0, "Waiting..."

    # 🔥 MICRO RECALIBRATION (USE IN TRACKING)
    def apply_micro_correction(self, predicted, actual=None):
        if actual is not None:
            error = np.array(actual) - np.array(predicted)
            self.global_offset += 0.05 * error
        return predicted + self.global_offset
