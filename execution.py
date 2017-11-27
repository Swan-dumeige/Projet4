# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:52:38 2017

@author: Sylvain
"""
import uncrypting_data
from sklearn import  svm, metrics#, datasets
import numpy as np
from pprint import pprint

def sklearn_use(training_data, test_data, C = 8, gamma = 0.0000001):
    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    
    n_samples = len(training_data[0])
    data = np.array(training_data[0]).reshape((n_samples, -1))
    
    n_samples_test = len(test_data[0])
    data_test = np.array(test_data[0]).reshape((n_samples_test, -1))
    
    # Create a classifier: a support vector classifier
    classifier = svm.SVC(C = C, gamma = gamma)
    
    # We learn the digits on the first half of the digits
    classifier.fit(data, training_data[1])
    
    # Now predict the value of the digit on the second half:
    expected = test_data[1]
    predicted = classifier.predict(data_test)
    
#    print("Classification report for classifier %s:\n%s\n"
#          % (classifier, metrics.classification_report(expected, predicted)))
#    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    precision = 0
#    print("\n\nthe confusion matrix : \n")
    for elm in metrics.confusion_matrix(expected, predicted):
#        print (['{0:4d}'.format(elm2) for elm2 in elm],"\n")
        precision += max(elm)
    return (precision/100)
        
#    
#    images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
#    for index, (image, prediction) in enumerate(images_and_predictions[:4]):
#        plt.subplot(2, 4, index + 5)
#        plt.axis('off')
#        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#        plt.title('Prediction: %i' % prediction)


if __name__=='__main__':
    print("1 etape")
    training_data = uncrypting_data.training_data()
    print("2 etape")
    test_data = uncrypting_data.test_data()
    print("3 etape")
    mat = []
    for i in range(3):
        mat2 = []
        for j in range(10):
            mat2.append(sklearn_use(training_data, test_data, C = 7.5+(0.5*i), gamma = 0.0000000001*(10**i)))
            print(i, "\t", j)
        mat.append(mat2)
    pprint(mat)
    print("fini")