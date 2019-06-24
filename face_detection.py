from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils,time
import dlib
import cv2
from io import BytesIO
import base64
from PIL import Image
import threading
from time import sleep
import binascii

########################################################

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
EYE_AR_THRESH = 0.19
EYE_AR_CONSEC_FRAMES = 6

########################################################

class blinkDetect():

    def __init__(self):

        self.predict_path='shape_predictor_68_face_landmarks.dat'
        self.faceDetector = dlib.get_frontal_face_detector()
        self.flmDetector = dlib.shape_predictor(self.predict_path)
        self.closedEyeFrame = 0
        self.totalBlink = 0
        self.ear = 0
        self.to_process = []
        self.to_output = []

###############################################################

    def pil_image_to_base64(self,pil_image):
        buf = BytesIO()
        pil_image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue())


    def base64_to_pil_image(self,base64_img):
        return Image.open(BytesIO(base64.b64decode(base64_img)))

        
##########################################################
        

    def eye_aspect_ratio(self,eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
     
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
     
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
     
        # return the eye aspect ratio
        return ear

#################################################################


    def processFrame(self,frame):
        
        frame = imutils.resize(frame, height=200,width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.faceDetector(gray, 1)
        
        for (i, rect) in enumerate(rects):
            
            shape = self.flmDetector(gray, rect)
            shape = face_utils.shape_to_np(shape)
        
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            
            self.ear = (leftEAR + rightEAR) / 2.0
            
          
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            
            if self.ear < EYE_AR_THRESH:
                self.closedEyeFrame += 1 
            
            else:            
                if self.closedEyeFrame >= EYE_AR_CONSEC_FRAMES:
                    self.totalBlink += 1 
            
                self.closedEyeFrame = 0       
        
        return frame



################################################################

    def blink_detect (self):
        
        if not self.to_process:
            return

        #-----------------------------------------

        # Conversion of image str to opencv image
        input_str = self.to_process.pop(0)
        input_img = self.base64_to_pil_image(input_str)

        frame = input_img

        pil_image = frame.convert('RGB')
        opencvImage = np.array(pil_image)
        #----------------------------------------------------#

        frame=self.processFrame(opencvImage)

        frame=cv2.flip(frame,1)       
            
        #--Textual information on frame--------------
        # cv2.putText(frame, "Blinks: {}".format(self.totalBlink), (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "EAR: {:.2f}".format(self.ear), (150, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #---------------------------------------------

        #Conversion of opencv image back to string--------------
        frame = Image.fromarray(frame)

        output_str = self.pil_image_to_base64(frame)
        op = binascii.a2b_base64(output_str)
        #-------------------------------------------------------

        self.to_output.append(op)

##########################################################################

    def enqueue_input(self, input):
        self.to_process.append(input)
        self.blink_detect()

###########################################################################

    def get_frame(self):
        while not self.to_output:
            sleep(0.05)
        return self.to_output.pop(0)
