# -*- coding: utf-8 -*-

import uncrypting_data 
import keras
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import optimizers
import csv 

def use_neural_network(learning_rate= 0.5, momentum = 0, decay = 0, batch_size = 300, nb_epoch = 30):    
    model = Sequential()
    
    #ajout d’une couche de projection linéaire (couche complètement connectée) de taille 10, suivi de l’ajout d’une couche d’activation de type softmax
    
    model.add(Dense(100,  input_dim=784, name='fc1'))
    model.add(Activation('sigmoid'))
    model.add(Dense(30,  input_dim=100, name='fc12'))
    model.add(Activation('sigmoid'))
    model.add(Dense(10,  input_dim=30, name='fc13'))
    
    model.add(Activation('softmax'))
    
    #visualiser l’architecture du réseau avec la méthode summary() du modèle
    # print (model.summary())
    
    
    #compiler le modèle en lui passant un loss (ici l” entropie croisée), une méthode d’optimisation (ici uns descente de gradient stochastique, stochatic gradient descent, sgd), et une métrique d’évaluation (ici le taux de bonne prédiction des catégories, accuracy):

    
    #print (" learning rate is : ", learning_rate,"\n momentum is : ", momentum, "\n decay is : ", decay)
    sgd = optimizers.SGD(learning_rate, momentum, decay)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    
    #l’apprentissage du modèle sur des données d’apprentissage est mis en place avec la méthode fit :
#    batch_size = 300 #initial : 300
#    nb_epoch = 30 #initial : 10
    
    # convert class vectors to binary class matrices
    Y_train = keras.utils.to_categorical(y_train, 10)
    Y_test = keras.utils.to_categorical(y_test, 10)
    model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)
    
    #évaluer les performances du modèle dur l’ensemble de test avec la fonction evaluate. Le premier élément de score renvoie la fonction de coût sur la base de test, le second élément renvoie le taux de bonne détection (accuracy).
    
    scores = model.evaluate(X_test, Y_test, verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return scores[1]*100
    

if __name__=="__main__":
    
    y_train = uncrypting_data.training_data()[1]
    y_test = uncrypting_data.test_data()[1]
    X_train = uncrypting_data.training_data()[0]
    X_test = uncrypting_data.test_data()[0]

    print ("\nData updated \n")
    
    counter = 0  
    
    print('\n ha!')
    file = open("MONFICHIER.csv", "w")
    c = csv.writer(file)
    c.writerow(['learning_rate','momentum','decay', 'result'])
    
    print('\n ha!')
    
#    for i in range(0.05, 0.81, 0.05):
#        for j in range(0, 1.05, 0.1):
#            for k in range(-5, 0, 1):
#                c.writerow( (  i, j, k, use_neural_network(learning_rate= i, momentum = j, decay = k)))
#                counter += 1
#                print(counter)
    c.writerow([0.03, 0.5, 0.001, use_neural_network(learning_rate= 0.03, momentum = 0.5, decay = 0.001)])
    
    print('\n ha!')
    
    file.close()
                
                
                
                
                
    