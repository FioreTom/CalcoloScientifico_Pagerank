import numpy as np


def back_substitution(A: np.array, b: np.array, verbose=False):
    """
    Risoluzione di un sistema triangolare superiore mediante il metodo di sostituzione all’indietro 

    :param A: matrice dei coefficienti, quadrata triangolare superiore, non singolare.
    :param b: vettore colonna dei termini noti
    :return: vettore colonna x, soluzione del sistema lineare Ax=b
    :rtype: np.array
    """

    n = np.size(A, 1)  # numero di righe di A
    x = np.zeros((n, 1))  # inizializza x ad un vettore colonna nullo

    x[n-1] = b[n-1] / A[n-1, n-1]
    if verbose:
        print("Per i =", n-1, ", x[i] =", x[n-1])

    for i in range(n-2, -1, -1):  # si procede all’indietro
        sommatoria = 0
        for j in range(i+1, n):
            sommatoria += float(A[i, j]) * float(x[j])

        if verbose:
            print("Per i =", i, ", sommatoria =", sommatoria)

        x[i] = (b[i] - sommatoria) / A[i, i]

        if verbose:
            print("Per i =", i, ", x[i] =", x[i])

    return x
