# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import os
os.chdir("C:/Users/Sylvain/Desktop/analyse_de_donnees")


def import_image_train(rank):
    """
    function returning the image in image_train database at the rank of 'rank'
    
    The image is a matrix of size 28X28
    
    exemple : import_image_train(1)
    return : the fisrt matrix (an image of 5)
    """

    file = open("train-images.idx3-ubyte", "rb")
    data = file.read()
    
    key = []
    for char in data[0:8]:
        key.append('{0:3d}'.format(char)) 
    assert key == ['  0', '  0', '  8', '  3', '  0', '  0', '234', ' 96']
    
    i = (8 + (rank * 784))
    mat = []
    for j in range(0, 28):
        row = []
        for char in data[int(i + (28 * j)): int(i + (28 * j) + 28)]:
            row.append(char)
        mat.append(row)
        
    file.close()
    return mat




def import_label_train(rank):
    """
    function returning the label in label_train database at the rank of 'rank'
    
    The label is an integer
    
    exemple : import_label_train(1)
    return : 5
    """

    file = open("train-labels.idx1-ubyte", "rb")
    data = file.read()
    
    key = []
    for char in data[0:8]:
        key.append('{0:3d}'.format(char))
    assert key == ['  0', '  0', '  8', '  1', '  0', '  0', '234', ' 96']
    
    
    i = (7 + rank)
    label = int(data[i])
        
    file.close()
    return label





print (import_image_train(1))
print (import_label_train(1))
        
        
        
        
        
        
        
        
        
        
#        
#    file = open("train-images.idx3-ubyte", "rb")
#    
#    data = file.read()
#    print(data[0:150])
#    
#    key = []
#    
#    #for char in data[0:120]:
#    ##    print('{0:08b} - {0:3d} - {1:s}'.format(char, str(bytes([char]))))
#    #    print('{0:3d}'.format(char))
#    #    key.append('{0:3d}'.format(char))
#        
#    #assert key == ['  0', '  0', '  8', '  3', '  0', '  0', '234', ' 96']
#    
#    
#    for i in range(8, (8 + (1 * 784)), 784):
#        mat = []
#        for j in range(0, 28):
#            row = []
#            for char in data[int(i + (28 * j)): int(i + (28 * j) + 28)]:
#                row.append(char)
#            mat.append(row)
#            print(row)
#        print((i-8)/784)
#        next_index = input()
#        
