def vecteurs (matrix):
    """
    fonction qui prend en entrée une martice et qui lui associe deux vecteurs : 
    X qui est le tableau de la somme de pixels noirs de l'image par colonne
    Y qui est le tableau de la somme de pixels noirs de l'image par ligne 
    """
    Y = []
    X = []
    for y in matrix:
        nby = y.count(255) 
        Y.append(nby)
    
    for l in range (len(matrix[1])):
        nbx = 0 
        for y in matrix:
            if y[l] == 255:
                nbx += 1            
        X.append(nbx)
    return (X,Y)