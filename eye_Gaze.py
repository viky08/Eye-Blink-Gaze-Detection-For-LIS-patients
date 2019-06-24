import os,pickle
from PIL import Image
import keras,imutils,time,dlib,cv2,time
from imutils import face_utils
import numpy as np
import operator,dlib
import binascii
from io import BytesIO
import base64
import keras.engine.sequential
#############################################
class eyeGaze():

	def __init__(self):

		self.to_process = []
		self.to_output = []
		self.direction ='center'
		self.eye_landmarks_model = pickle.load(open("./eye_landmarks_model.pkl","rb"))
		self.Left_Eye_Gaze_Model = pickle.load(open("./Left_Eye_Gaze_Model.pkl","rb"))
		self.Right_Eye_Gaze_Model = pickle.load(open("./Right_Eye_Gaze_Model.pkl","rb"))
		self.faceDetector = dlib.get_frontal_face_detector()

###############################################################
	def pil_image_to_base64(self,pil_image):
		buf = BytesIO()
		pil_image.save(buf, format="JPEG")
		return base64.b64encode(buf.getvalue())

	
	def base64_to_pil_image(self,base64_img):
		return Image.open(BytesIO(base64.b64decode(base64_img)))

        
##########################################################

	def rect_to_bb(self,rect):

		x = rect.left()
		y = rect.top()
		w = rect.right() - x
		h = rect.bottom() - y

		return (x, y, w, h)


	def face_extraction(self,gray):
	    
	    # image = frame
	    # # Convert the image to RGB colorspace
	    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	    # # Convert the RGB  image to grayscale
	    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	    
	    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

	    faces = face_cascade.detectMultiScale(gray, 1.25, 6)
	    
	    # image_with_detections = np.copy(image)
	    res = 0

	    (x, y, w, h) = (0,0,0,0)
	  
	    if(len(faces)>0):
	        (x, y, w, h) = faces[0]
	        res = 1    
	        gray = gray[y:y+h, x:x+w]
	        gray = cv2.resize(gray, (96, 96)) / 255
	    
	    return res,gray, (x, y, w, h)


	def get_eyes(self,prediction, frame, face_details):
	    (x, y, w, h) = face_details
	    left_eye_inner_corner_x = (prediction[4] * w / 96) + x
	    left_eye_inner_corner_y = (prediction[5] * h / 96) + y
	    left_eye_outer_corner_x = (prediction[6] * w / 96) + x
	    left_eye_outer_corner_y = (prediction[7] * h / 96) + y
	    right_eye_inner_corner_x = (prediction[8] * w / 96) + x
	    right_eye_inner_corner_y = (prediction[9] * h / 96) + y
	    right_eye_outer_corner_x = (prediction[10] * w / 96) + x
	    right_eye_outer_corner_y = (prediction[11] * h / 96) + y
	    left_eyebrow_inner_end_x = (prediction[12] * w / 96) + x
	    left_eyebrow_inner_end_y = (prediction[13] * h / 96) + y
	    left_eyebrow_outer_end_x = (prediction[14] * w / 96) + x
	    left_eyebrow_outer_end_y = (prediction[15] * h / 96) + y
	    right_eyebrow_inner_end_x = (prediction[16] * w / 96) + x
	    right_eyebrow_inner_end_y = (prediction[17] * h / 96) + y
	    right_eyebrow_outer_end_x = (prediction[18] * w / 96) + x
	    right_eyebrow_outer_end_y = (prediction[19] * h / 96) + y
	    height_diff_left_inner = abs(left_eyebrow_inner_end_y - left_eye_inner_corner_y)
	    height_diff_left_outer = abs(left_eyebrow_outer_end_y - left_eye_outer_corner_y)
	    height_left_eye = max(height_diff_left_inner, height_diff_left_outer) * 2
	    height_diff_right_inner = abs(right_eyebrow_inner_end_y - right_eye_inner_corner_y)
	    height_diff_right_outer = abs(right_eyebrow_outer_end_y - right_eye_outer_corner_y)
	    height_right_eye = max(height_diff_right_inner, height_diff_right_outer) * 2
	   
	    if abs(left_eyebrow_outer_end_x - left_eyebrow_inner_end_x) > abs(left_eye_outer_corner_x - left_eye_inner_corner_x):
	        width_left_eye = abs(left_eyebrow_outer_end_x - left_eyebrow_inner_end_x)
	        left_eye_x = min(left_eyebrow_outer_end_x, left_eyebrow_inner_end_x)
	    else:
	        width_left_eye = abs(left_eye_outer_corner_x - left_eye_inner_corner_x)
	        left_eye_x = min(left_eye_outer_corner_x, left_eye_inner_corner_x)
	    if abs(right_eyebrow_outer_end_x - right_eyebrow_inner_end_x) > abs(right_eye_outer_corner_x - right_eye_inner_corner_x):
	        width_right_eye = abs(right_eyebrow_outer_end_x - right_eyebrow_inner_end_x)
	        right_eye_x = min(right_eyebrow_outer_end_x, right_eyebrow_inner_end_x)
	    else:
	        width_right_eye = abs(right_eye_outer_corner_x - right_eye_inner_corner_x)
	        right_eye_x = min(right_eye_outer_corner_x, right_eye_inner_corner_x)
	    
	    #PIL image
	    cv2.imwrite('img.jpg',frame)
	    img = Image.open('img.jpg')
	    
	    left_eye_rect = (left_eye_x, left_eyebrow_outer_end_y, left_eye_x + width_left_eye, left_eyebrow_outer_end_y + height_left_eye)
	    left_eye_img = img.crop(left_eye_rect).convert('L')
	    left_eye_img = left_eye_img.resize((50,42), Image.ANTIALIAS)
	    right_eye_rect = (right_eye_x, right_eyebrow_inner_end_y, right_eye_x + width_right_eye, right_eyebrow_inner_end_y + height_right_eye)
	    right_eye_img = img.crop(right_eye_rect).convert('L')
	    right_eye_img = right_eye_img.resize((50, 42), Image.ANTIALIAS)
	    return left_eye_img, right_eye_img


	
	def final_prediction(self,left_eye_prob, right_eye_prob):
	    overall_prob = (left_eye_prob + right_eye_prob) / 2
	    index, value = max(enumerate(overall_prob), key=operator.itemgetter(1))
	    return index


	def predict(self,frame,dum,x1,y1,w1,h1):
	    
	    class_name = ['Center', 'Down Left', 'Down Right', 'Left', 'Right', 'Up Left', 'Up Right']
	    face_img, x, y, w, h = dum,x1,y1,w1,h1
	    
	    landmarks_prediction = np.squeeze(self.eye_landmarks_model.predict(np.expand_dims(np.expand_dims(face_img, axis=-1), axis=0)))
	    landmarks_prediction = landmarks_prediction * 48 + 48
	    left_eye_img, right_eye_img = self.get_eyes(landmarks_prediction, frame, (x, y, w, h)) 
	    
	    left_eye_img.load()
	    img_arr_left = np.asarray(left_eye_img, dtype="int32")
	    img_arr_left = img_arr_left.reshape(42,50,1)
	    
	    right_eye_img.load()
	    img_arr_right = np.asarray(right_eye_img, dtype="int32")
	    img_arr_right = img_arr_right.reshape(42,50,1)
	    
	    left_eye_prediction = self.Left_Eye_Gaze_Model.predict(img_arr_left.reshape(1, 42, 50, 1))[0]
	    right_eye_prediction = self.Right_Eye_Gaze_Model.predict(img_arr_right.reshape(1, 42, 50, 1))[0]
	    pred_class = self.final_prediction(left_eye_prediction, right_eye_prediction)
	    
	    if (pred_class == 1 or pred_class == 3 or pred_class == 5) :
	        return "left"
	    elif (pred_class == 2 or pred_class == 4 or pred_class == 6) :
	        return "right"

	    return "center"


################################
# mixer.init()
# mixer.music.load('sound.mp3')
# count = 0
#################################


	def processFrame(self,frame):

		# frame = imutils.resize(frame, height=200,width=300)
		frame = cv2.resize(frame,(300,200))
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		rects = self.faceDetector(gray, 1)
        
		for (i, rect) in enumerate(rects):
			(x,y,w,h) = self.rect_to_bb(rect)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

			gray = gray[y:y+h, x:x+w]
			gray = cv2.resize(gray, (96, 96)) / 255

			xx=self.predict(frame,gray,x,y,w,h)
			self.direction = xx;
			cv2.putText(frame,xx, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2 )

					

		# res,dum,(x,y,w,h) =  self.face_extraction(gray)

		# if(res):
		# 	# xx=self.predict(frame,dum,x,y,w,h)
		# 	# cv2.putText(frame,xx, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2 )
		# 	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# 	# self.direction = xx;

		return frame

# if xx=='right':
	        #     count= count+1
	        #     if count>=3:
	        #         mixer.music.play()
	        #         count = 0



	def eyeGaze_main(self):

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
		
		opencvImage=cv2.flip(opencvImage,1)	
		frame=self.processFrame(opencvImage)
		
         


        #Conversion of opencv image back to string--------------
		frame = Image.fromarray(frame)

		output_str = self.pil_image_to_base64(frame)
		op = binascii.a2b_base64(output_str)
        #-------------------------------------------------------

		self.to_output.append(op)
############################################################################        

	def enqueue_input(self, input):
		self.to_process.append(input)
		self.eyeGaze_main()


	def get_frame(self):
		while not self.to_output:
			sleep(0.05)
		return self.to_output.pop(0)
