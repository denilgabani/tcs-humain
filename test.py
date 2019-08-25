# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 23:51:49 2019

@author: DG
"""
#import necessary files
from face_detection import face_detector as fd 
import cv2
from keras.models import load_model
import numpy as np

#Function for loading models
def load():
    emotions = load_model('emotions.h5')
    age = load_model('age.h5')
    ethnicity = load_model('ethnicity.h5')
    gender = load_model('gender.h5')
    return emotions, age, ethnicity, gender
    

#Fucntion for predicting given image or frame and return corresponding classes
def test(e,a,eth,g,img):
    emo_list = ['Angry','Happy','Neutral','Sad']
    age_list = ['20-30','30-40','40-50','Above_50','Below_20']
    eth_list = ['Arab','Asian','Black','Hispanic','Indian','White']
    gender_list = ['Male','Female']
    
    img = cv2.resize(img,(75,75))
    img = np.array(img)
    img = img.reshape((1,75,75,3))
    img = img/255
    pred_e = e.predict(img) #predicting emotion from image file
    pred_a = a.predict(img) #predicting age from image file
    pred_eth = eth.predict(img) #predicting ethnicity from image file
    pred_g = g.predict(img) #predicting gender from image file
    idx_e = np.argmax(pred_e) #Getting predicted emotion
    idx_a = np.argmax(pred_a) #Getting predicted age
    idx_eth = np.argmax(pred_eth) #Getting predicted ethnicity
    idx_g = np.argmax(pred_g) #Getting predicted gender
    class_names = 'emotoins:'+emo_list[idx_e]+' age:'+age_list[idx_a]+' ethnicity:'+eth_list[idx_eth]+' gender:'+gender_list[idx_g]
    return class_names
    
    
 #Two optins predicting from image or real time(webcam)   
print("Choose a Option:\n 1.Predict from Image \n 2.Predict from Webcam")
opt = int(input())
image_name = None

if(opt==1):
    print('Enter image name:')
    image_name = input()
    img = cv2.imread(image_name,cv2.IMREAD_COLOR)
    faces = fd(img)#find face by using face detector function of face detection file 
    if len(faces)==0:
        exit()
    e,a,eth,g = load()#loading models
    print('Model Load Successfully')

    #predicting and drawing rectangle of each face in image
    for (startX, startY, endX, endY) in faces:
        d = img[startY:endY,startX:endX]
        text = test(e,a,eth,g,d)
        print(text)
        y= startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(img, (startX, startY), (endX, endY),
        			(0, 0, 255), 3)#drawing rectangle in image on face
        cv2.putText(img, text, (startX, y),
                			cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)#write text on rectangle
    cv2.imwrite('output.jpg',img)        
    cv2.imshow("Output", cv2.resize(img,(500,500)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif(opt==2):#if second option selected
    camera = cv2.VideoCapture(0)#capture from webcam
    e,a,eth,g = load()#loading models
    print('Model Load Successfully')
    while True:
        ret, frame = camera.read()#reading frame from webcam
        if ret==True:
            faces = fd(frame)#detecting faces from corresponding frames
            if len(faces)==0:#if face not detect in current frame then skip the iteration
                continue
            #predicting and drawing rectangle of each face in image
            for (startX, startY, endX, endY) in faces:
                d = frame[startY:endY,startX:endX]
                text = test(e,a,eth,g,d)
                
                print(text)
                y= startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                			(0, 0, 255), 3)
                cv2.putText(frame, text, (startX, y),
                			cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                    
            cv2.imshow("Output", cv2.resize(frame,(500,500)))
            k = cv2.waitKey(33)
            if k==27:#Close the window by pressing ESC key
                break
    camera.release()
    cv2.destroyAllWindows()
else:
    print('Enter Valid Input')
    













