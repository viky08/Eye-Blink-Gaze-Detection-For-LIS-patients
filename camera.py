import numpy as np
import cv2

class Camera():
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3,540)
        self.cap.set(4,320)
    
    def get_frame(self):
        ret, frame = self.cap.read()
        # formated_data = cv2.imencode('.jpeg', frame)[1]
        # frame_bytes = formated_data.tobytes()
        return frame
    
    def __del__(self):
        self.cap.release()