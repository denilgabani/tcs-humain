# tcs-humain

Here i provide solution of tcs humain competition problem which is Face Recognition- Identify Age, Emotion and Ethnicity of a person.

For training and testing follow the below steps:
  
# Setp:1 Preprocess the Data
  
  ### Run Following Command:
  ```
  python preprocess.py
  ```
  In competition we have given dataset in json format which is Face_Recognition.json.
  
  Now first this file transform the dataset into dataframe (dataframe is easier to process) from json file then it give this dtaframe to    download_img() function and it download the all images from given link into 'images/extract' folder (make sure you have better connection otherwise it disconnect from network and give TimeOut Error).
  
  Now it run second function data_preprocess() function.It first pass image height,width and given faces point to cropping function. cropping() function crop the all faces from image at give points which extract from json file and save it that crop image - face image into 'images/extract' folder. Then emotions, age, gender, ethnicity labels save with corresponding images and add into dataframe.
  
  This dataframe contains image name, emotions, age, ethnicity and gender. Then drop rows which contains none value and process other value which are swap in each other columns.Then it makes direcotry emotions, age, gender, ethnicity with sub-directory which is same as labels in columns and after it copy all images from 'images/extract' folder to corresponding folder and then romove 'images' folder.
  
  Now we have directories which contain all images to belonging to their classes.We can simply used it with keras flow_from_directory() function.
  

# Step:2 Training the models

  ### Run following command:
      
     '''python train.py'''
     
   Now it train the models one by one.It first run the train_emotions() function of emotions.py file. Now it train the emotions folder images onto a pre-trained model VGG19.
   
    

  
  
  













