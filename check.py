import pickle
import os
from keras.layers import Flatten,Dense,Dropout,Activation,MaxPooling2D,Conv2D
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.models import Sequential
import cv2

CATEGORIES=['Dog',"Cat"]
test_dir= 'D:\\Learing_Document\\Machine Learning\\Database\\Dog_and_Cat\\Test_set'
training_dir='D:\\Learing_Document\\Machine Learning\\Database\\Dog_and_Cat\\Training_set'
def create_feature_label(dir,img_size=50):
    for categories in CATEGORIES:
        path=os.path.join(dir,categories)
        feature=[]
        label=[]
        for img in os.listdir(path):
            try:
                img_array= cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                img_array= cv2.resize(img_array,(img_size,img_size))
                feature.append(img_array)
                label.append(CATEGORIES.index(categories))
                print(img)
            except:
                print('Error: %s'%categories, img)
    database= dict([("data",feature),('label',label)])
    return database
database=create_feature_label(training_dir)
print(len(database['label']))

