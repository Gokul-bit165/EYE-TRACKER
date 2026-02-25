import numpy as np
import time

class CalibrationManager:
    def __init__(self, mode='9-point', screen_w=1920, screen_h=1080):
        self.mode = mode
        self.screen_w = screen_w
        self.screen_h = screen_h
        
        self.points = self._generate_points(mode)
        self.current_point_idx = 0
        
        # Store corresponding data
        self.collected_features = []
        self.collected_targets = []
        
        # Temporary collection
        self.current_point_features = []
        self.point_start_time = 0
        self.state = "DELAY" # States: DELAY, CAPTURING, DONE_POINT
        self.frames_to_collect = 30

    def _generate_points(self, mode: str):
        """Generate screen coordinates for calibration."""
        w, h = self.screen_w, self.screen_h
        dx = int(w * 0.15)
        dy = int(h * 0.15)
        
        if mode == '9-point':
            return [
                (dx, dy), (w // 2, dy), (w - dx, dy),
                (dx, h // 2), (w // 2, h // 2), (w - dx, h // 2),
                (dx, h - dy), (w // 2, h - dy), (w - dx, h - dy)
            ]
        elif mode == '5-point':
            return [
                (dx, dy), (w - dx, dy),
                (w // 2, h // 2),
                (dx, h - dy), (w - dx, h - dy)
            ]
        elif mode == '16-point':
            # 4x4 grid across the screen
            xs = [dx, int(w*0.33), int(w*0.66), w - dx]
            ys = [dy, int(h*0.33), int(h*0.66), h - dy]
            pts = []
            for y in ys:
                for x in xs:
                    pts.append((x, y))
            return pts
        else:
            raise ValueError("Unsupported calibration mode.")

    def get_current_point(self):
        if self.current_point_idx < len(self.points):
            return self.points[self.current_point_idx]
        return None

    def start_point(self):
        self.point_start_time = time.time()
        self.state = "DELAY"
        self.current_point_features = []

    def add_frame_data(self, features: np.ndarray) -> str:
        """
        Adds frame data based on the current state.
        Returns the current state.
        """
        if self.state == "DELAY":
            if time.time() - self.point_start_time > 0.5: # 500ms delay
                self.state = "CAPTURING"
            return "DELAY"
            
        elif self.state == "CAPTURING":
            self.current_point_features.append(features)
            if len(self.current_point_features) >= self.frames_to_collect:
                self.state = "DONE_POINT"
            return "CAPTURING"
            
        return self.state

    def get_progress(self):
        if self.state == "DELAY": return 0.0
        return len(self.current_point_features) / self.frames_to_collect

    def next_point(self):
        """
        Processes the collected features, rejects outliers, averages them, and advances.
        """
        if len(self.current_point_features) > 0:
            features_array = np.array(self.current_point_features)
            
            # Reject outliers using Z-score
            if len(features_array) > 5:
                z_scores = np.abs((features_array - np.mean(features_array, axis=0)) / (np.std(features_array, axis=0) + 1e-5))
                mask = (z_scores < 2.0).all(axis=1) # Keep within 2 std devs
                filtered = features_array[mask]
                if len(filtered) == 0:
                    filtered = features_array # Fallback
            else:
                filtered = features_array
                
            # Average the stable samples
            avg_features = np.mean(filtered, axis=0)
            target = self.points[self.current_point_idx]
            
            self.collected_features.append(avg_features)
            self.collected_targets.append(target)
            
        self.current_point_idx += 1
        if self.current_point_idx < len(self.points):
            self.start_point()
            return True
        return False

    def is_complete(self):
        return self.current_point_idx >= len(self.points)

    def get_dataset(self):
        return np.array(self.collected_features), np.array(self.collected_targets)

    def calculate_error(self, model) -> float:
        """Calculate average error in pixels over the calibration points."""
        if len(self.collected_features) == 0: return 9999.0
        X = np.array(self.collected_features)
        Y = np.array(self.collected_targets)
        
        total_error = 0
        for x, y_true in zip(X, Y):
            y_pred = model.predict(x)
            if y_pred:
                dist = np.linalg.norm(np.array(y_pred) - np.array(y_true))
                total_error += dist
        return total_error / len(X)

    def reset(self):
        self.current_point_idx = 0
        self.collected_features = []
        self.collected_targets = []
        self.start_point()
