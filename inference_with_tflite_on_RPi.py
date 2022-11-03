
#from tensorflow.keras.preprocessing.image import img_to_array #Can use cv2 or other libraries.
import cv2
from tflite_runtime.interpreter import Interpreter

import numpy as np
import time

face_classifier=cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')

#### PREDICT USING tflite ###
#On RPi you can try both opt and no opt. 

# Load the TFLite model and allocate tensors.
emotion_interpreter = Interpreter(model_path="emotion_detection_model_100epochs_no_opt.tflite")
emotion_interpreter.allocate_tensors()


# Get input and output tensors.
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()


# Test the model on input data.
emotion_input_shape = emotion_input_details[0]['shape']

###########################


class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    labels=[]
    
    #frame = cv2.imread(frame)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    start = time.time()

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        #Get image ready for prediction
    
        roi=roi_gray.astype('float')/255.0  #Scale
        roi = np.array(roi)
        #roi=img_to_array(roi)
        #roi = np.float32(roi)
        roi=np.expand_dims(roi,axis=0)  #Expand dims to get it ready for prediction (1, 48, 48, 1)
        
        
        #emotion_input_details[0]['index'] = np.float32(emotion_input_details[0]['index'])
        #print(emotion_interpreter.get_input_details())
        #print(emotion_input_details[0]['index'])
        #print(roi)
        emotion_interpreter.set_tensor(emotion_input_details[0]['index'], roi)
        emotion_interpreter.invoke()
        emotion_preds = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])

        #preds=emotion_model.predict(roi)[0]  #Yields one hot encoded result for 7 classes
        emotion_label=class_labels[emotion_preds.argmax()]  #Find the label
        emotion_label_position=(x,y)
        cv2.putText(frame,emotion_label,emotion_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
    end=time.time()
    print("Total time=", end-start)
    
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  #Press q to exit
        break
cap.release()
cv2.destroyAllWindows()


##############################

