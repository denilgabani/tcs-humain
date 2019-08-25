# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 15:12:13 2019

@author: DG
"""

import keras
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow

#Create a model 
def get_model():
        
    model = keras.applications.vgg19.VGG19(include_top=False, input_shape=(75,75,3))
    x=model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(4,activation='softmax')(x) #final layer with softmax activation
    model.summary()
    model = Model(inputs=model.input, outputs=preds)
    
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', 
              metrics=['accuracy'])
    return model

def train_emotions():
    #Generator for Image Augmentation
    datagen = ImageDataGenerator(
        preprocessing_function= \
        tensorflow.keras.applications.mobilenet.preprocess_input,
        rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True)
    
    #Taking image from directory
    train_batches = datagen.flow_from_directory('./images/emotions',
                                                target_size=(75,75),
                                                color_mode="rgb",
                                                batch_size=32)
    
    STEP_SIZE_TRAIN=train_batches.n//train_batches.batch_size
    
    
    #Using CheckPoint for getting best accuracy and save weights into directory
    file_path = "emotions.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
    
    reduce_on_plateau = ReduceLROnPlateau(monitor="acc", mode="max", factor=0.1, patience=20, verbose=1)
    
    callbacks_list = [checkpoint, reduce_on_plateau]
    
    #Get model from function
    model = get_model()
    
    #Training model
    result = model.fit_generator(train_batches, steps_per_epoch=STEP_SIZE_TRAIN, 
                        epochs=100, callbacks=callbacks_list)
    
    return result
    

    
    
    
    
    
    
    
    
    
