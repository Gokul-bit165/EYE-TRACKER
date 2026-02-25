import numpy as np
import cv2

class KalmanFilter2D:
    def __init__(self, process_noise=1e-5, measurement_noise=1e-2, ema_alpha=0.5):
        """
        Initializes a Kalman Filter for 2D points (x, y) with an EMA pre-filter.
        State vector: [x, y, dx, dy] (position and velocity)
        Measurement vector: [x, y]
        """
        self.kf = cv2.KalmanFilter(4, 2)
        
        # State transition matrix
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        # Measurement matrix (we only measure x and y)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        # Process noise covariance matrix
        self.kf.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32) * process_noise

        # Measurement noise covariance matrix
        self.kf.measurementNoiseCov = np.array([
            [1, 0],
            [0, 1]
        ], np.float32) * measurement_noise
        
        # Initial guess error
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        # EMA variables
        self.ema_x = None
        self.ema_y = None
        self.ema_alpha = ema_alpha
        
        self.initialized = False

    def update(self, x, y):
        """
        Updates the filter with a new measurement (x,y).
        First applies EMA, then Kalman prediction & correction.
        """
        if not self.initialized:
            self.ema_x = np.float32(x)
            self.ema_y = np.float32(y)
            self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [0.], [0.]], dtype=np.float32)
            self.initialized = True
            return (int(x), int(y))
            
        # Apply Exponential Moving Average (EMA) and cast explicitly to float32
        self.ema_x = np.float32(self.ema_alpha * float(x) + (1 - self.ema_alpha) * self.ema_x)
        self.ema_y = np.float32(self.ema_alpha * float(y) + (1 - self.ema_alpha) * self.ema_y)
        
        measurement = np.array([[self.ema_x], [self.ema_y]], dtype=np.float32)
        
        self.kf.predict()
        self.kf.correct(measurement)
        
        # Return smoothed position
        state = self.kf.statePost
        return (int(state[0][0]), int(state[1][0]))

    def reset(self):
        self.initialized = False
        self.ema_x = None
        self.ema_y = None
