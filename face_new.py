
# coding: utf-8

# In[1]:

import dlib


# In[2]:

import face_recognition


# In[3]:

import cv2


# In[4]:

import cv2
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('http://192.168.43.1:8080/video')
print(video_capture.isOpened())


# In[5]:

get_ipython().system('pwd')


# In[6]:

get_ipython().system('ls')


# In[7]:

import os


# In[8]:

os.getcwd()


# In[10]:

os.chdir('C:/Users/ARUN/Downloads/Celebrity')


# In[11]:

get_ipython().system('pwd')


# In[16]:

images = os.listdir('c:/Users/ARUN/Downloads/Celebrity')


# In[ ]:




# In[ ]:




# In[17]:

print(images)


# In[18]:

image_to_be_matched = face_recognition.load_image_file('xyz.jpg')


# In[19]:

image_to_be_matched_encoded = face_recognition.face_encodings(
    image_to_be_matched)[0]


# In[24]:

# iterate over each image
for image in images:
    # load the image
    current_image = face_recognition.load_image_file("C:/Users/ARUN/Downloads/Celebrity/" + image)
    # encode the loaded image into a feature vector
    current_image_encoded = face_recognition.face_encodings(current_image)[0]
    # match your image with the image and check if it matches
    result = face_recognition.compare_faces(
        [image_to_be_matched_encoded], current_image_encoded)
    # check if it was a match
    if result[0] == True:
        print ("Matched: " + image)
    else:
        print ("Not matched: " + image)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



