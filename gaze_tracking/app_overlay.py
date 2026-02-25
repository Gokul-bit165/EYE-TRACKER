import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen
import collections

class GazeOverlay(QWidget):
    # Signals to communicate back to the main session
    recalibrate_signal = pyqtSignal()
    quit_signal = pyqtSignal()

    def __init__(self, screen_w, screen_h, trail_length=30):
        super().__init__()
        self.screen_w = screen_w
        self.screen_h = screen_h
        
        # Transparent, always on top, no window framing, pass-through clicks
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool |
            Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Set to cover the entire screen
        self.setGeometry(0, 0, screen_w, screen_h)
        
        # Trail configuration
        self.trail_length = trail_length
        self.points = collections.deque(maxlen=self.trail_length)
        
        self.current_x = -100
        self.current_y = -100

    def update_gaze(self, x, y):
        """Updates the current gaze position and adds it to the trail."""
        self.current_x = x
        self.current_y = y
        self.points.append((x, y))
        self.update() # Trigger repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if len(self.points) == 0:
            return

        # Draw Trail
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i+1]
            
            # Fade out the tail
            alpha = int(255 * (i / self.trail_length))
            pen = QPen(QColor(0, 255, 255, alpha), 3)
            painter.setPen(pen)
            
            painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
            
        # Draw current Cursor (Head)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 255, 0, 200)) # Green dot
        painter.drawEllipse(QPoint(int(self.current_x), int(self.current_y)), 10, 10)
        
        # Outer ring
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(0, 255, 0, 150), 2))
        painter.drawEllipse(QPoint(int(self.current_x), int(self.current_y)), 18, 18)

    def keyPressEvent(self, event):
        """Handle global keyboard shortcuts while the overlay is focused."""
        if event.key() == Qt.Key_R:
            self.recalibrate_signal.emit()
        elif event.key() == Qt.Key_Q or event.key() == Qt.Key_Escape:
            self.quit_signal.emit()

def create_overlay(screen_w, screen_h):
    app = QApplication(sys.argv)
    overlay = GazeOverlay(screen_w, screen_h)
    overlay.show()
    return app, overlay
