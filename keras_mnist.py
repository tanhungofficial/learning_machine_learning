from keras import layers,optimizers,losses,models,datasets,utils
import matplotlib.pyplot as plt
import  numpy as np
import  cv2
from sklearn import  neighbors
(xTrain,yTrain),(xTest,yTest)=datasets.mnist.load_data()
class CNN():
    def __init__(self):
        pass
    def trainingModel(self,epochs=2,batch_size=500):
        model = models.Sequential()
        model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='softmax'))
        model.compile(optimizer=optimizers.Adam(), loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        model.summary()
        model.fit(self.xTrain, self.yTrain, batch_size=batch_size, epochs=epochs, verbose=1)
        model.save("cnn_model_mnist")
    def initialize(self,xTrain,yTrain,xTest,yTest):#,xTrain,yTrain,xTest,yTest
        self.xTrain= xTrain.reshape(60000,28,28,1)
        self.yTrain= yTrain
        self.xTest= xTest.reshape(10000,28,28,1)
        self.yTest= yTest

    def predict(self,index):
        data= self.xTest[index]
        data= data.reshape(1,28,28,1)
        new_model = models.load_model('cnn_model_mnist')
        score = new_model.predict([data])
        score = np.argmax(score, axis=1)
        print('{:<35} {:<5}'.format('Predict input number as:', '%d' % score))
        print('{:<35} {:<5}'.format('Truth label of input image:', '%d' % yTest[index]))
        img = cv2.resize(xTest[index], (512, 512))
        cv2.imshow("Input number", img)
        cv2.waitKey()
        cv2.destroyAllWindows()

def KNN(index, xTrain,yTrain):
    shape= xTrain.shape
    knn= neighbors.KNeighborsClassifier(n_neighbors=10)
    dataTrain = xTrain.reshape(shape[0],784)
    knn.fit(dataTrain,yTrain)
    result=knn.predict(xTest[index].reshape(1,784))
    print('{:<35} {:<5}'.format('Predict input number as:', '%d'%result))
    print('{:<35} {:<5}'.format('Truth label of input image:','%d'%yTest[index]))
    img = cv2.resize(xTest[index],(256,256))
    cv2.imshow("Input image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
#Cach dung KNN
#   KNN(index,xTrain[0:1000],yTrain[0:1000])    muon chay chi can thay doi so index

#Cach dung CNN
#   cnn=CNN()
#   cnn.initialize(xTrain,yTrain,xTest,yTest)
#   cnn.predict(index) #chi cam thay doi so index