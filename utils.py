import numpy as np


def is_diag_dom(X: np.array, verbose=False):
    
    D = np.diag(np.abs(X)) # Vettore degli elementi diagonali
    S = np.sum(np.abs(X), axis=1) - D # Vettore delle somme degli elementi sulle righe, tranne gli elementi diagonali
    
    if verbose:
        print(D)
        print(S)
   
    if np.all(D > S):
        return True
    else:
        return False


def hilbert_matrix(n: int):
    H = np.ones((n, n))

    for i in range(n):
        for k in range(n):
            H[i][k] /= i+k+1
    
    return H
    