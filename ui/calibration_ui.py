import cv2
import numpy as np

class CalibrationUI:
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        
        # UI settings
        self.bg_color = (15, 15, 15) # Darker modern theme
        self.color_delay = (0, 150, 255) # Orange for Delay
        self.color_capture = (0, 255, 0) # Green for Capture
        self.text_color = (220, 220, 220)

        # Animation state
        self.base_radius = 15
        self.pulse = 0
        self.pulse_dir = 1

    def draw_calibration_screen(self, point: tuple, state: str, progress: float) -> np.ndarray:
        """
        Draws the calibration screen with an interactive, state-aware animated dot.
        """
        frame = np.full((self.screen_h, self.screen_w, 3), self.bg_color, dtype=np.uint8)
        
        if point is not None:
            # Animate pulse
            self.pulse += 1 * self.pulse_dir
            if self.pulse >= 12: self.pulse_dir = -1
            if self.pulse <= 0: self.pulse_dir = 1
            
            radius = self.base_radius + self.pulse
            x, y = point
            
            if state == "DELAY":
                color_inner = self.color_delay
                color_outer = (0, 80, 200)
                msg_main = "Focus on the dot..."
                msg_sub = "Preparing to capture"
            else:
                color_inner = self.color_capture
                color_outer = (0, 150, 0)
                msg_main = "Hold steady!"
                msg_sub = "Capturing data..."
            
            # Draw outer expanding circle
            cv2.circle(frame, (x, y), radius + 10, color_outer, 2)
            # Draw inner solid circle
            cv2.circle(frame, (x, y), radius, color_inner, -1)
            
            # Draw progress bar circle around dot if collecting
            if progress > 0.0:
                axes = (radius + 20, radius + 20)
                angle = 0
                startAngle = -90
                endAngle = -90 + int(360 * progress)
                cv2.ellipse(frame, (x, y), axes, angle, startAngle, endAngle, (0, 255, 255), 4)

            # Draw instructions near the dot or center top
            cv2.putText(frame, msg_main, (self.screen_w // 2 - 150, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.text_color, 2)
            cv2.putText(frame, msg_sub, (self.screen_w // 2 - 120, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
                        
        return frame
