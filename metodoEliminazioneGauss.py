import numpy as np


def MEG_naive(A: np.array, b: np.array, verbose=False):
    """
    Trasformazione di un sistema in uno equivalente, triangolare superiore.
    Non effettua scambi di riga

    :param A: matrice dei coefficienti, quadrata.
    :param b: vettore colonna dei termini noti
    :return: matrice dei coefficienti U, quadrata triangolare superiore.
    :return: vettore colonna dei termini noti c
    :return: determinante della matrice U
    :rtype: np.array, np.array, float
    """

    U = np.copy(A)
    c = np.copy(b)
    det = 1

    rows = np.size(U, 0)
    cols = np.size(U, 1)
    if rows != cols:
        print("Matrice dei coefficienti non quadrata")
        return U, c, None
    if rows != np.size(c, 0):
        print("Matrice dei coefficienti e vettore dei termini noti non compatibili")
        return U, c, None

    C = np.linalg.norm(A, np.inf)

    for k in range(cols-1):

        if U[k][k] == 0:
            print("Elemento nullo sulla diagonale")
            return U, c, None

        if abs(U[k][k]) < abs(np.finfo(float).eps * C):
            print("Warning! Possibile elemento nullo sulla diagonale")

        det *= U[k][k]  # aggiornamento del determinante

        for i in range(k+1, rows):
            moltiplicatore = U[i][k] / U[k][k]
            U[i, k:] = U[i, k:] - moltiplicatore * U[k, k:]
            c[i] = c[i] - moltiplicatore * c[k]
            if verbose:
                print(np.append(U, c, 1))
                print()

    # aggiornamento del determinante e controllo
    if abs(U[rows-1][cols-1]) < abs(np.finfo(float).eps * C):
        print("Warning! Il determinante potrebbe essere nullo")
    det *= U[rows-1][cols-1]

    return U, c, det


def MEG_pivoting(A: np.array, b: np.array, verbose=False):
    """
    Trasformazione di un sistema in uno equivalente, triangolare superiore, 
    con scambi di righe

    :param A: matrice dei coefficienti, quadrata.
    :param b: vettore colonna dei termini noti
    :return: matrice dei coefficienti U, quadrata triangolare superiore.
    :return: vettore colonna dei termini noti c
    :return: determinante della matrice U
    :rtype: np.array, np.array, float
    """

    U = np.copy(A)
    c = np.copy(b)
    det = 1

    rows = np.size(U, 0)
    cols = np.size(U, 1)
    if rows != cols:
        print("Matrice dei coefficienti non quadrata")
        return U, c, None
    if rows != np.size(c, 0):
        print("Matrice dei coefficienti e vettore dei termini noti non compatibili")
        return U, c, None

    C = np.linalg.norm(A, np.inf)

    pivot = np.arange(cols)

    for k in range(cols-1):

        argmax = abs((U[k:rows, k])).argmax() + k
        max_element = U[argmax][k]
        if verbose:
            print("argmax =", argmax)
            print("maxElement =", max_element)

        if argmax > k:
            U[[k, argmax]] = U[[argmax, k]]  # scambio di riga in U
            c[[k, argmax]] = c[[argmax, k]]  # scambio di riga in c
            # scambio indici in pivot
            pivot[k], pivot[argmax] = pivot[argmax], pivot[k]
            det = -det  # in caso di scambio di riga, il determinante cambia segno

        if U[k][k] == 0:
            print("Elemento nullo sulla diagonale")
            return U, c, None

        if abs(U[k][k]) < abs(np.finfo(float).eps * C):
            print("Warning! Possibile elemento nullo sulla diagonale")

        det *= U[k][k]  # aggiornamento del determinante

        for i in range(k+1, rows):
            moltiplicatore = U[i][k] / U[k][k]
            U[i, k:] = U[i, k:] - moltiplicatore * U[k, k:]
            c[i] = c[i] - moltiplicatore * c[k]
            if verbose:
                print(np.append(U, c, 1))
                print()

    # aggiornamento del determinante e controllo
    if abs(U[rows-1][cols-1]) < abs(np.finfo(float).eps * C):
        print("Warning! Il determinante potrebbe essere nullo")
    det *= U[rows-1][cols-1]

    if verbose:
        print("array pivot: ", pivot)

    return U, c, det
