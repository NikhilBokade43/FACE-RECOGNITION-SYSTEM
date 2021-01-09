
# coding: utf-8

# In[194]:


#importing nececessary packages
import warnings
warnings.filterwarnings('ignore')
import imutils 
import argparse
from imutils.video import VideoStream
import numpy as np 
import time
import cv2
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder, Normalizer
from keras.models import load_model
import math


# In[ ]:


#load the encoding 
data = np.load('5-celebrity-faces-dataset.npz')
embd_data = np.load('5-celebrity-faces-embeddings.npz')
trainx_embd, trainy, testx_embd, testy = embd_data['arr_0'], embd_data['arr_1'], embd_data['arr_2'], embd_data['arr_3']
le = LabelEncoder()

#using mtcnn to detect faces in an video or frames of the video
detector = MTCNN()
print('<< detector loaded')
facenet = load_model('facenet_keras.h5', compile =  False)
print('<< facenet loaded')
norm = Normalizer('l2')


# In[202]:


def distance(embeddings1, embeddings2, distance_metric = 0):
    if distance_metric == 0:
        'taking euclidean distance :'
        dist = np.linalg.norm(embeddings1 - embeddings2)
    elif distance_metric == 1:
        'taking cosine distance :'
        dot = np.sum(np.multiply(embeddings1, embeddings2))
        denominator = np.multiply(np.linalg.norm(embeddings1), np.linalg.norm(embeddings2) )
        similarity = dot/denominator
        dist = np.arccos(similarity) / math.pi #np.arccos([1, -1]) = array([ 0.        ,  3.14159265])
    return dist


# In[251]:


# #constriucting the argument parser and parse the argument
# ap = argparse.ArgumentParser()
# ap.add_argument('--e', '--encodings', required = True, 
#                 help = 'path to the normalized encoding of faces in train data')
# ap.add_argument("-o", "--output", type=str,
#                 help="path to output video")
# ap.add_argument("-y", "--display", type=int, default=1,
#                 help="whether or not to display output frame to screen")

# args = vars(ap.parse_args())

video_capture = cv2.VideoCapture(0)#'0' is used for my webcam, 
while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1 ) #(for example, 1) means flipping around y-axis.
    result = detector.detect_faces(frame)
    x1, y1, width, height = result[0]['box']
    y2 = y1 + height
    x2 = x1 + width
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
    face = frame[y1:y2, x1:x2] #need this to get face embeddings 
    face = cv2.resize(face, (160, 160))
    face_pixels = face.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean)/std
    face_pixels = np.expand_dims(face_pixels, axis = 0)
    embd = facenet.predict(face_pixels)
    min_dist = 100 #lets take and then we will change this iteratively
    for i in range(trainx_embd.shape[0]):
        actual_name = trainy[i]
        dist = distance(trainx_embd[i].reshape(-1,1) , embd.reshape(-1,1), 1 )
        if dist < min_dist:
            min_dist = dist
            identity = actual_name #if the distance is the min of all then only we chnage the identity here
    if min_dist < 0.39 :
        cv2.putText(frame, "Face : " + identity, (x1, y1 - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.putText(frame, "Dist : " + str(min_dist), (x1, y1 - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Unknown Face', (x1, y1 - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
    
    cv2.imshow('face_rec_syatem', frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
video_capture.release()
cv2.destroyAllWindows()

