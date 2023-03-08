import tensorflow as tf 
import cv2 
import os 
import matplotlib.pyplot as plt 
import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers 
import random
from pygame import mixer 
import threading
import time

font_scale = 1.5
font= cv2.FONT_HERSHEY_PLAIN
#set the rectangle background to white
rectangle_bgr = (255, 255, 255)
# make a black image
img= np.zeros((500, 500))
#set some text
text= "Some text in a box!"
# get the width and height of the text box
(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
# set the text start position
text_offset_x = 10
text_offset_y=img.shape [0] - 25
# make the coords of the box with a small padding of two pixels
box_coords= ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height-2))
cv2.rectangle (img, box_coords [0], box_coords [1], rectangle_bgr, cv2.FILLED)
cv2.putText (img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)


cap= cv2.VideoCapture (0)



def play_music(path): 
    # loder le fichier
    mixer.init()
    mixer.music.load(path)
    mixer.music.play()
    time.sleep(15)
   
    
    
    
    
music_thread=None


def emotion(status, folder_path):
    global music_thread
    

    x1, y1, w1, h1 = 0,0,175,75
    # Draw black background rectangle
    cv2.rectangle (frame, (x1, x1), (x1+ w1, y1 + h1), (0,0,0), -1)
    # Add text
    cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255),2,cv2.LINE_4)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

    files = [f for f in os.listdir(folder_path) if f.endswith(".mp3")]        
    random_file = random.choice(files)
    file_path = os.path.join(folder_path, random_file)

    if music_thread and music_thread.is_alive():
        music_thread.join()
        mixer.music.stop()

    music_thread = threading.Thread(target=play_music, args=(file_path,))
    music_thread.start()

    


    
while True:
    ret, frame = cap.read()
    print(frame)
    new_model = tf.keras.models.load_model('emotion_model.hdf5', compile=False)
    FaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces= FaceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color= frame [y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # BGR
        facess=FaceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex:ex + ew]
        
        final_image = cv2.resize(face_roi, (64, 64))
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
        final_image = np.expand_dims(final_image, axis=-1)
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0

       
        font = cv2.FONT_HERSHEY_SIMPLEX
        Predictions = new_model.predict(final_image)
        font_scale = 1.5
        font =cv2.FONT_HERSHEY_PLAIN
        
        
        
        if (np.argmax (Predictions) ==0):
            emotion("Angry","music/angry/")
            
        elif (np.argmax (Predictions) == 1):
            
            emotion("Disgust","music/disgust/")
            
        elif (np.argmax (Predictions) == 2):
           
            emotion("Fear","music/fear/")
            
            
        elif (np.argmax (Predictions) == 3):
            emotion("Happy","music/happy/")
            
            
        elif (np.argmax (Predictions) == 4):
           
            emotion("Sad","music/sad/")
            
        elif (np.argmax (Predictions) == 5):
            
            emotion("Surprise","music/surprise/")

            
        elif (np.argmax (Predictions) == 6):
            
            emotion("Neutral","music/neutral/")
            
            
        cv2.imshow('Face Emotion Recognition',frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
            
cap.release()
pygame.quit()
cv2.destroyAllWindows()