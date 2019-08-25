# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:24:55 2019

@author: DG
"""
import cv2
import numpy as np

def face_detector(img):
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
    	(300, 300), (104.0, 177.0, 123.0))
    
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt','res10_300x300_ssd_iter_140000_fp16.caffemodel')
    
    
    net.setInput(blob)
    detections = net.forward()
    
    faces =[]
    for i in range(0, detections.shape[2]):
    	# extract the confidence (i.e., probability) associated with the
    	# prediction
        confidence = detections[0, 0, i, 2]
     
    	# filter out weak detections by ensuring the `confidence` is
    	# greater than the minimum confidence
        if confidence > 0.5:
    		# compute the (x, y)-coordinates of the bounding box for the
    		# object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append(box.astype("int"))
    		# draw the bounding box of the face along with the associated
    return faces
            


















