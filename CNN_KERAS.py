
import uncrypting_data 
import keras
import csv 
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
import csv 
import numpy as np

def use_neural_network(learning_rate= 0.5, momentum = 0, decay = 0, batch_size = 300, nb_epoch = 30):    
    model = Sequential()
    
    #construction réseau convolutif de neurones vide
    model.add(Conv2D(64,kernel_size=(5,5),activation='sigmoid',input_shape=(28, 28, 1),padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32,kernel_size=(5,5),activation='sigmoid',input_shape=(28, 28, 1),padding='same'))
    MaxPooling2D(pool_size=(2, 2))
    model.add(Flatten())
    
    model.add(Dense(100,  input_dim=784, name='fc1'))
    model.add(Activation('sigmoid'))
    model.add(Dense(30,  input_dim=100, name='fc12'))
    model.add(Activation('sigmoid'))
    model.add(Dense(10,  input_dim=30, name='fc13'))
    model.add(Activation('softmax'))
    
    # descente de gradient stochastique (sgd), et métrique d’évaluation (accuracy):
    sgd = optimizers.SGD(learning_rate, momentum, decay)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    
    #l’apprentissage du modèle sur des données d’apprentissage ( et convertion des classes de vecteurs en clase de matrice binaire)
    Y_train = keras.utils.to_categorical(y_train, 10)
    Y_test = keras.utils.to_categorical(y_test, 10)
    model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)
    
    #évaluer les performances du modèle sur l’ensemble de test avec la fonction evaluate. Le premier élément de score renvoie la fonction de coût sur la base de test, le second élément renvoie le taux de bonne détection (accuracy).
    scores = model.evaluate(X_test, Y_test, verbose=0)
    
    return scores[1]*100
    

if __name__=="__main__":
    
    y_train = np.array(uncrypting_data.training_data()[1])
    y_test = np.array(uncrypting_data.test_data()[1])
    X_train = np.array(uncrypting_data.training_data()[0])
    X_test = np.array(uncrypting_data.test_data()[0])
    print ("\nData updated \n")
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    
    print(use_neural_network(learning_rate = 1, momentum = 0.5,
                             decay = 0.001, batch_size = 100, nb_epoch = 10))