import cv2
import numpy as np
import time

class DisplayOverlay:
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0
        
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray, color=(0, 255, 0), radius=1):
        """Draws small circles for landmarks."""
        for x, y in landmarks:
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)

    def draw_gaze_cursor(self, frame: np.ndarray, point: tuple, color=(0, 0, 255), radius=15):
        """Draw a circle representing where the user is looking."""
        x, y = point
        # Ensure it's drawn within screen boundaries approximately
        h, w = frame.shape[:2]
        
        # Transparent overlay for cursor
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), radius, color, -1)
        cv2.circle(overlay, (x, y), int(radius*1.5), (0, 255, 255), 2)
        
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    def update_and_draw_fps(self, frame: np.ndarray) -> float:
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        self.fps = self.fps * 0.9 + fps * 0.1 # Exponential moving average
        
        cv2.putText(frame, f"FPS: {int(self.fps)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return self.fps

    def draw_ui_panel(self, frame: np.ndarray, state: str, blink: bool):
        """Draws a professional UI panel with state info."""
        h, w = frame.shape[:2]
        panel = np.zeros((h, 250, 3), dtype=np.uint8)
        
        # Status text
        cv2.putText(panel, "Gaze Tracker", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(panel, f"State: {state}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Blink indicator
        color = (0, 0, 255) if blink else (0, 255, 0)
        cv2.putText(panel, f"Blink: {'YES' if blink else 'NO'}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Combine
        # Replace right side of frame with panel if frame is large enough
        if w > 250:
             # Alpha blend panel instead of just placing it
             overlay = frame.copy()
             overlay[:, w-250:w] = panel
             cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        return frame
