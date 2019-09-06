# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:13:47 2019

@author: DG
"""
#import libaries

import shutil
import os
import pandas as pd
from urllib.request import urlretrieve
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



#Function for downloading images
def download_img(data):
    if not os.path.exists('images'):
        os.mkdir('images')
    main_path = 'images/extract/'
    if not os.path.exists(main_path):
        os.mkdir(main_path)
    x=data.iloc[:64,1]
    count=0
    for link in x:
        fn = main_path+str(count)+'.jpeg'
        urlretrieve(link, fn)
        print(fn)
        count+=1
    #Here Same code written again because it drop connection after long time and give TimeOut Error
    x=data.iloc[64:,1]
    count=64
    for link in x:
        fn = main_path+str(count)+'.jpeg'
        urlretrieve(link, fn)
        print(fn)
        count+=1
        
        


#function for cropping image at given points and save as new image
def cropping(img_name,idx,h,w,point):
    x1 = point[0]['x']
    y1 = point[0]['y']
    x2 = point[1]['x']
    y2 = point[1]['y']
    x1 = x1*w
    y1 = y1*h
    x2 = x2*w
    y2 = y2*h
    img = Image.open('images/extract/'+str(img_name)+".jpeg").convert('RGB')
    img2 = img.crop((x1,y1,x2,y2))
    fn = str(img_name)+'_'+str(idx)+".jpeg"
    img2.save('images/extract/'+fn)
    return fn

#Preprocessing Data
def data_preprocess(data):
    # Taking annotation attribute of data
    annots = data['annotation'] 
    filename = []
    emotions = []
    age = []
    ecity = []
    gender = []
    for img,annot in enumerate(annots):
        if img==7:#Problem in 7th image it doesn't open
            continue
        for idx,lbl in enumerate(annot):
            print(str(img)+'--'+str(idx))
            if len(lbl['label'])==0 or lbl['label'][0]=='Not_Face':
                continue
            height = lbl['imageHeight']
            width = lbl['imageWidth']
            point = lbl['points']
            #Selecting Face from image by point given in data using above cropping function
            fname = cropping(img,idx,height,width,point)
            #Appending filename in list of face with emotions,age,ethnicity,
            filename.append(fname) 
            try:
                emotions.append(lbl['label'][0])
            except IndexError:
                emotions.append(None)
            try:
                age.append(lbl['label'][1])
            except IndexError:
                age.append(None)
            try:
                ecity.append(lbl['label'][2])
            except IndexError:
                ecity.append(None)
            try:
                gender.append(lbl['label'][3])
            except IndexError:
                gender.append(None)
            
    #Creating Dataframe with field name, emotions, age,ethnicity, gender
    df = pd.DataFrame(data={'name':filename,'emotions':emotions,'age':age,'ethnicity':ecity,'gender':gender})
    
    df = df.dropna()
    #Some age and emotions value swap in dataframe
    df['emotions'].unique()
    df.loc[[109,166],['emotions','age']] = df.loc[[109,166],['age','emotions']].values
    
    df['age'].unique()
    #Remove 75 index row because all column values is age
    df = df.drop(index=75)
    
    #Swap Some gender and ethnicity value
    df['ethnicity'].unique()
    mask = df['ethnicity'].apply(lambda x:'True' if x in ['G_Male','G_ Female'] else 'False' )
    i = df.index[mask=='True'].tolist()
    df.loc[i,['ethnicity','gender']] = df.loc[i,['gender','ethnicity']].values
    
    #assigning Male value manually in 119 row
    df['gender'].unique()
    df.loc[119,'gender'] = 'G_Male'
    
    #Taking list of all column's ubique values
    e = df['emotions'].unique()
    a = df['age'].unique()
    eth = df['ethnicity'].unique()
    g = df['gender'].unique()

    mklist = [e,a,eth,g] 
    
    #Creating individual directory for all unique attributes
    for idx,n in enumerate(mklist):
        path=None
        if idx==0:
            path = 'images/emotions/'
            if not os.path.exists(path):
                os.mkdir(path)
            
        if idx==1:
            path = 'images/age/'
            if not os.path.exists(path):
                os.mkdir(path)
            
        if idx==2:
            path = 'images/ethnicity/'
            if not os.path.exists(path):
                os.mkdir(path)
            
            
        if idx==3:
            path = 'images/gender/'
            if not os.path.exists(path):
                os.mkdir(path)
        for k in n:
            if not os.path.exists(path+k):
                os.mkdir(path+k)
    
    
    #Copying image file to corresponding directory, not moving because one image can belong to many categories 
    for ind in df.index:
        fn = df.loc[ind,'name']
        ed = df.loc[ind,'emotions']
        ad = df.loc[ind,'age']
        enthd = df.loc[ind,'ethnicity']
        gd = df.loc[ind,'gender']
        shutil.copy('images/extract/'+fn,'images/emotions/'+ed)
        shutil.copy('images/extract/'+fn,'images/age/'+ad)
        shutil.copy('images/extract/'+fn,'images/ethnicity/'+enthd)
        shutil.copy('images/extract/'+fn,'images/gender/'+gd)
        
    shutil.rmtree('images/extract')


def run():
    data = pd.read_json('Face_Recognition.json',lines=True)
    
    download_img(data)
    
    data_preprocess(data)

run()

