# tcs-humain

Here i provide solution of tcs humain competition problem which is Face Recognition- Identify Age, Emotion and Ethnicity of a person.

For training and testing follow the below steps:
  
## Setp:1 Preprocess the Data
  
  ### Run Following Command:
  ```
  python preprocess.py
  ```
  In competition we have given dataset in json format which is Face_Recognition.json.
  
  Now first this file transform the dataset into dataframe (dataframe is easier to process) from json file then it give this dtaframe to    download_img() function and it download the all images from given link into 'images/extract' folder (make sure you have better connection otherwise it disconnect from network and give TimeOut Error).
  
  Now it run second function data_preprocess() function.It first pass image height,width and given faces point to cropping function. cropping() function crop the all faces from image at give points which extract from json file and save it that crop image - face image into 'images/extract' folder. Then emotions, age, gender, ethnicity labels save with corresponding images and add into dataframe.
  
  This dataframe contains image name, emotions, age, ethnicity and gender. Then drop rows which contains none value and process other value which are swap in each other columns.Then it makes direcotry emotions, age, gender, ethnicity with sub-directory which is same as labels in columns and after it copy all images from 'images/extract' folder to corresponding folder and then romove 'images' folder.
  
  Now we have directories which contain all images to belonging to their classes.We can simply used it with keras flow_from_directory() function.
  

## Step:2 Training the models

  ### Run following command:
      
     python train.py
     
   Now it train the models one by one.It first run the train_emotions() function of emotions.py file. Now it train the emotions folder images onto a pre-trained model VGG19 whose layers shown in below image:
   ![Image of VGG19](https://miro.medium.com/max/2408/1*6U9FJ_se7SIuFKJRyPMHuA.png)
   
   train_emotions() function train the images with image augmentation techniques - for generating more images - and save the model in emotions.h5. It save with best weights by using ModelCheckPoint() function and ReduceLr of keras.
   
   Then it runs train_age(), train_ethnicity(), train_gender() function with same as above technique and save the model as age.h5, ethnicity.h5, gender.h5.
   
   ```You must train models with powerful GPU or colab ```
   
## Step:3 Testing the model:

  ### Run following command:
  
    python test.py
    
  It provide two options:
  
     1.Predict from Image
     2.Predict from Webcam
   
  If you want to predict emotions, age, ethnicity, gender from image choose option 1 or
  If tou want to predict emotions, age, ethnicity, gender from webcam(real time) then choose option 2.
  
  If you choose option 1 it will be ask you to enter image name in this you must enter image path. Then it read image and load models and predict corresponding class labels then it write preidcted labels with rectangular box onto face in image and sshow you which is as below:
  
  ![Image of Input]
  ![Image of Output]
  
  If you choose option 2 it open webcam and give frames into loaded models and predict class labels. It show you a window in which video will play with labels and rectangulr red box onto image. You can close it by simply pressing **esc** key
  
  
  
  

  
  
  













