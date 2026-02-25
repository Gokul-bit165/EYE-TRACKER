import numpy as np
import math

class GazeGeometry:
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        
        # Calibration offsets (set by looking at center of screen)
        self.offset_yaw = 0.0
        self.offset_pitch = 0.0
        
        # Field of View mapping limits (degrees)
        self.yaw_fov = 15.0      # ±15 degrees horizontal
        self.pitch_fov = 5.0     # ±5 degrees vertical

    def compute_scale(self, points_3d: np.ndarray) -> float:
        """Computes the average pairwise distance of points to get a stable scale factor"""
        n = len(points_3d)
        if n < 2: return 1.0
        
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(points_3d[i] - points_3d[j])
                total += dist
                count += 1
        return total / count if count > 0 else 1.0

    def compute_combined_gaze(self, iris_l, iris_r, sphere_l, sphere_r) -> np.ndarray:
        """Computes a normalized combined gaze direction vector from both eyes"""
        left_gaze = iris_l - sphere_l
        if np.linalg.norm(left_gaze) > 1e-9:
            left_gaze /= np.linalg.norm(left_gaze)
            
        right_gaze = iris_r - sphere_r
        if np.linalg.norm(right_gaze) > 1e-9:
            right_gaze /= np.linalg.norm(right_gaze)
            
        combined = (left_gaze + right_gaze) * 0.5
        if np.linalg.norm(combined) > 1e-9:
            combined /= np.linalg.norm(combined)
            
        return combined

    def get_raw_angles(self, gaze_dir: np.ndarray) -> tuple:
        """Converts a 3D gaze vector into raw Yaw and Pitch angles in degrees"""
        reference_forward = np.array([0, 0, -1])  # Z-axis into the screen
        avg_direction = gaze_dir / np.linalg.norm(gaze_dir)

        # Horizontal (yaw)
        xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
        xz_norm = np.linalg.norm(xz_proj)
        if xz_norm > 1e-9: xz_proj /= xz_norm
        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
        if avg_direction[0] < 0:
            yaw_rad = -yaw_rad

        # Vertical (pitch)
        yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
        yz_norm = np.linalg.norm(yz_proj)
        if yz_norm > 1e-9: yz_proj /= yz_norm
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
        if avg_direction[1] > 0:
            pitch_rad = -pitch_rad  # up is positive

        yaw_deg = np.degrees(yaw_rad)
        pitch_deg = np.degrees(pitch_rad)

        # Mirror mappings depending on camera setup
        if yaw_deg < 0:
            yaw_deg = -yaw_deg
        elif yaw_deg > 0:
            yaw_deg = -yaw_deg

        return yaw_deg, pitch_deg

    def calibrate_fov(self, center_gaze, tl_gaze, tr_gaze, br_gaze, bl_gaze):
        """Sets the center offset and calculates the actual FOV from the 4 corners"""
        # 1. Set Center Offset
        raw_yaw, raw_pitch = self.get_raw_angles(center_gaze)
        self.offset_yaw = -raw_yaw
        self.offset_pitch = -raw_pitch
        
        # 2. Extract angles for corners
        yaw_tl, pitch_tl = self.get_raw_angles(tl_gaze)
        yaw_tr, pitch_tr = self.get_raw_angles(tr_gaze)
        yaw_br, pitch_br = self.get_raw_angles(br_gaze)
        yaw_bl, pitch_bl = self.get_raw_angles(bl_gaze)
        
        # 3. Apply the offset to the corners
        yaw_tl += self.offset_yaw
        yaw_tr += self.offset_yaw
        yaw_br += self.offset_yaw
        yaw_bl += self.offset_yaw
        
        pitch_tl += self.offset_pitch
        pitch_tr += self.offset_pitch
        pitch_br += self.offset_pitch
        pitch_bl += self.offset_pitch
        
        # 4. Calculate FOV
        # The FOV is half of the total angular width/height.
        # Top-Left should have negative Yaw and positive Pitch (in standard Cartesian, or whatever mirror mapping we are using).
        # We'll just take the absolute averages of the extremes.
        
        avg_horizontal_fov = (abs(yaw_tl) + abs(yaw_tr) + abs(yaw_br) + abs(yaw_bl)) / 4.0
        avg_vertical_fov = (abs(pitch_tl) + abs(pitch_tr) + abs(pitch_br) + abs(pitch_bl)) / 4.0
        
        # Add a tiny little margin so the extreme edges are reachable
        self.yaw_fov = max(5.0, avg_horizontal_fov * 1.05)
        self.pitch_fov = max(2.0, avg_vertical_fov * 1.05)
        
        return self.offset_yaw, self.offset_pitch, self.yaw_fov, self.pitch_fov

    def get_screen_coordinates(self, gaze_dir: np.ndarray) -> tuple:
        """Maps the 3D gaze vector to strictly clamped 2D Screen coordinates"""
        yaw_deg, pitch_deg = self.get_raw_angles(gaze_dir)
        
        # Apply offsets
        yaw_deg += self.offset_yaw
        pitch_deg += self.offset_pitch

        # Map to full screen resolution
        screen_x = int(((yaw_deg + self.yaw_fov) / (2 * self.yaw_fov)) * self.screen_w)
        screen_y = int(((self.pitch_fov - pitch_deg) / (2 * self.pitch_fov)) * self.screen_h)

        # Clamp screen position to monitor bounds
        screen_x = max(0, min(screen_x, self.screen_w - 1))
        screen_y = max(0, min(screen_y, self.screen_h - 1))

        return screen_x, screen_y
