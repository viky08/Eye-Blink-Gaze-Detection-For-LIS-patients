from flask import Flask, render_template, Response
import dlib,cv2,sys
from imutils import face_utils
import camera, face_detection,helper
from flask_socketio import SocketIO
from PIL import Image
from io import BytesIO
import base64,json
import pandas as pd
import eye_Gaze,sendSMS


app = Flask(__name__)
app.config['DEBUG'] = True
app.config['ENV'] = True
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app)
trie = helper.Trie()

#----------------------------------------------------------


#------------------------------------------------------------
# Home page 

homeBlinkDetect = face_detection.blinkDetect()
homeBlinkChange = 0

@socketio.on('homeImage')
def test_message(input):

	input_str = input.split(",")[1]

	prevBlinkCount = homeBlinkDetect.totalBlink

	homeBlinkDetect.enqueue_input(input_str)
	op = homeBlinkDetect.get_frame()

	homeBlinkChange = homeBlinkDetect.totalBlink - prevBlinkCount
	
	if homeBlinkChange :
		socketio.emit('ima1',op)
	else:
		socketio.emit('ima2',op)


#------------------------------------------------------
#MOde 1

mode1BlinkDetect = face_detection.blinkDetect()
mode1BlinkChange = 0

@socketio.on('mode1Image')
def test_message(input):

	input_str = input.split(",")[1]

	prevBlinkCount = mode1BlinkDetect.totalBlink

	mode1BlinkDetect.enqueue_input(input_str)
	op = mode1BlinkDetect.get_frame()

	mode1BlinkChange = mode1BlinkDetect.totalBlink - prevBlinkCount
	
	if mode1BlinkChange :
		socketio.emit('imag1',op)
	else:
		socketio.emit('imag2',op)
#--------------------------------------------------------


# MODE 2
blinkDetect = face_detection.blinkDetect()
change = 0

@socketio.on('input image')
def test_message(input):

	input_str = input.split(",")[1]

	prevBlinkCount = blinkDetect.totalBlink

	blinkDetect.enqueue_input(input_str)
	op = blinkDetect.get_frame()

	change = blinkDetect.totalBlink - prevBlinkCount
	
	if change :
		socketio.emit('im1',{'op': op,'blinkCount':blinkDetect.totalBlink, 'ear': round(blinkDetect.ear,2) } )
	else:
		socketio.emit('im2',{'op': op,'blinkCount':blinkDetect.totalBlink, 'ear': round(blinkDetect.ear,2) } )
#-----------------------------------------------------------------------------------------


#Mode 3

mode3BlinkDetect = face_detection.blinkDetect()
mode3BlinkChange = 0

@socketio.on('mode3Image')
def test_message(input):

	input_str = input.split(",")[1]

	prevBlinkCount = mode3BlinkDetect.totalBlink

	mode3BlinkDetect.enqueue_input(input_str)
	op = mode3BlinkDetect.get_frame()

	mode3BlinkChange = mode3BlinkDetect.totalBlink - prevBlinkCount
	
	if mode3BlinkChange :
		socketio.emit('image1',op)
	# else:
	# 	socketio.emit('image2',op)
#------------------------------------------------------------------------------

# EyeGaze
eyeGaze = eye_Gaze.eyeGaze()

@socketio.on('gaze')
def test(input):
	
	input_str = input.split(",")[1]
	eyeGaze.direction = ""
	eyeGaze.enqueue_input(input_str)
	op = eyeGaze.get_frame()

	if eyeGaze.direction == "left":
		socketio.emit('gazeLeft',op)
		
	elif eyeGaze.direction == "right":
		socketio.emit('gazeRight',op)
	else:
		socketio.emit('gazeCenter',op)

#------------------------------------------------------------------------------


# Auto Complete
@socketio.on('autocomplete')
def predictingWord(input):
	#print(input)
	input = input.lower()
	words = list(trie.autocomplete(input))
	# print(words)
	socketio.emit('words',words[0:5])
	

#------------------------------------------------------------------------------

sms = sendSMS.sendSMS()
@socketio.on('sendsms')
def sendsms(input):
	sms.sendmessage(input)


#------------------------------------------------------------------------------

@app.route('/')
@app.route('/index')

def index():
	user={'username':'VineeT Goyal'}
	return render_template('index.html')


#-------------------------------------------
@app.route('/mode1')
def mode1():
	return render_template('mode1.html')
#-------------------------------------------


#-------------------------------------------
@app.route('/mode2')
def mode2():
	f = open('data.txt')

	for word in f.read().split():
		trie.insert(word)

	return render_template('mode2.html')

#-------------------------------------------

@app.route('/mode3')
def mode3():
	return render_template('mode3.html')
#--------------------------------------------




if __name__=='__main__':
	socketio.run(app)

#------------------------------------------------------------------------------------------------
# Ignore

# def gen():
# 	while True:
# 		frame = blinkDetect.get_frame()
# 		yield frame

# @app.route('/video_feed')
# def video_feed():
# 	for video_frame in gen():
# 		socket.emit('from_flask',{'data':video_frame.decode()},namespace='/test')


################################################################################

# def gen(camera):
# 	global change
# 	while True:

# 		frame = camera.get_frame()
# 		prevBlinkCount = blinkDetect.totalBlink
# 		frame = blinkDetect.blink_detect(frame)
# 		change = blinkDetect.totalBlink - prevBlinkCount
		
		# formated_data = cv2.imencode('.jpeg', frame)[1]
		# frame_bytes = formated_data.tobytes()

# 		# return frame_bytes

# 		yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes+ b'\r\n')




# for video_frame in gen(camera.Camera()):
# 	socketio.emit('from_flask', {'data': video_frame})
# @app.route('/video_feed')
# def video_feed():
# 	return Response(gen(camera.Camera()),
#                 mimetype='multipart/x-mixed-replace; boundary=frame')

