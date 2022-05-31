import numpy as np


def jacobi(A: np.array, b: np.array, x0: np.array, toll: float, n_max: int, verbose=False):
    """
    Risoluzione di un sistema lineare Ax=b data un'approssimazione iniziale x0.
    L'algoritmo implementa il metodo iterativo di Jacobi.

    :param A: matrice dei coefficienti
    :param b: vettore colonna dei termini noti
    :param x0: approssimazione iniziale
    :param toll: tolleranza, ovvero precisione relativa richiesta
    :param n_max: massimo numero di iterazioni
    :param verbose: se True, stampa informazioni sulle iterazioni
    :return: soluzione x, stima dell'errore, numero di iterazioni eseguite 
    :rtype: np.array, float, int
    """

    # Inizializzazioni
    rows = np.size(A, 0)
    cols = np.size(A, 1)
    n_iter = 0
    stima_errore = 1.0
    x = np.copy(x0)

    # Processo iterativo
    while n_iter < n_max and stima_errore >= toll:

        for i in range(rows):

            somma1 = 0
            for j in range(i):
                somma1 += A[i, j] * x0[j]

            somma2 = 0
            for j in range(i+1, cols):
                somma2 += A[i, j] * x0[j]

            x[i] = (b[i] - somma1 - somma2) / A[i, i]

        if verbose:
            print("valore di x alla iterazione numero ",
                  n_iter, ", stima errore =", stima_errore)
            print(x.T)

        stima_errore = np.linalg.norm(x-x0, np.inf) / np.linalg.norm(x, np.inf)
        n_iter += 1
        x0 = np.copy(x)

    return x, stima_errore, n_iter


def jacobi_decomposizione(A, b, x0, toll, n_max, verbose=False):
    """
    Risoluzione di un sistema lineare Ax=b data un'approssimazione iniziale x0.
    L'algoritmo implementa il metodo iterativo di Jacobi usando la decomposizione 
    della matrice dei coefficienti A = D + C.

    :param A: matrice dei coefficienti
    :param b: vettore colonna dei termini noti
    :param x0: approssimazione iniziale
    :param toll: tolleranza, ovvero precisione relativa richiesta
    :param n_max: massimo numero di iterazioni
    :param verbose: se True, stampa informazioni sulle iterazioni
    :return: soluzione x, stima dell'errore, numero di iterazioni eseguite
    :rtype: np.array, float, int
    """

    # calcolo matrici D e C
    D = np.diag(np.diag(A))
    C = A - D

    # Inizializzazioni
    n_iter = 0
    x = np.copy(x0)

    # Processo iterativo
    while n_iter < n_max:
        x = np.dot(np.linalg.inv(D), (b - np.dot(C, x0)))

        stima_errore = np.linalg.norm(x-x0, np.inf) / np.linalg.norm(x, np.inf)
        if stima_errore <= toll:
            break
        x0 = np.copy(x)
        n_iter += 1
        if verbose:
            print("valore di x alla iterazione numero ",
                  n_iter, ", stima errore =", stima_errore)
            print(x.T)

    return x, stima_errore, n_iter


def jacobi_decomposizione_ottimizzato(A, b, x0, toll, n_max, verbose=False):
    """
    Risoluzione di un sistema lineare Ax=b data un'approssimazione iniziale x0.
    L'algoritmo implementa il metodo iterativo di Jacobi usando la decomposizione 
    della matrice dei coefficienti A = D + C.

    :param A: matrice dei coefficienti
    :param b: vettore colonna dei termini noti
    :param x0: approssimazione iniziale
    :param toll: tolleranza, ovvero precisione relativa richiesta
    :param n_max: massimo numero di iterazioni
    :param verbose: se True, stampa informazioni sulle iterazioni
    :return: soluzione x, stima dell'errore, numero di iterazioni eseguite
    :rtype: np.array, float, int
    """

    # calcolo matrici D e C
    D = np.diag(np.diag(A))
    C = A - D

    # Inizializzazioni
    n_iter = 0
    x = np.copy(x0)

    # Matrici iterative
    B = np.dot(np.linalg.inv(D), b)
    c = np.dot(np.linalg.inv(D), C)

    # Processo iterativo
    while n_iter < n_max:
        x = B - np.dot(c, x0)

        stima_errore = np.linalg.norm(x-x0, np.inf) / np.linalg.norm(x, np.inf)
        if stima_errore <= toll:
            break
        x0 = np.copy(x)
        n_iter += 1
        if verbose:
            print("valore di x alla iterazione numero ",
                  n_iter, ", stima errore =", stima_errore)
            print(x.T)

    return x, stima_errore, n_iter


def gauss_seidel(A: np.array, b: np.array, x0: np.array, toll: float, n_max: int, verbose=False):
    """
    Risoluzione di un sistema lineare Ax=b data un'approssimazione iniziale x0.
    L'algoritmo implementa il metodo iterativo di Gauss Seidel.

    :param A: matrice dei coefficienti
    :param b: vettore colonna dei termini noti
    :param x0: approssimazione iniziale
    :param toll: tolleranza, ovvero precisione relativa richiesta
    :param n_max: massimo numero di iterazioni
    :param verbose: se True, stampa informazioni sulle iterazioni
    :return: soluzione x, stima dell'errore, numero di iterazioni eseguite
    :rtype: np.array, float, int
    """

    # Inizializzazioni
    rows = np.size(A, 0)
    n_iter = 0
    stima_errore = 1.0
    x = np.copy(x0)

    # Processo iterativo
    while n_iter < n_max and stima_errore >= toll:

        for i in range(0, rows):

            somma1 = 0
            for j in range(i):
                somma1 = somma1 + A[i, j] * x[j]

            somma2 = 0
            for j in range(i+1, rows):
                somma2 = somma2 + A[i, j] * x0[j]

            x[i] = (b[i] - somma1 - somma2) / A[i, i]

        if verbose:
            print("valore di x alla iterazione numero ", n_iter)
            print(x.T)

        stima_errore = np.linalg.norm(x-x0, np.inf) / np.linalg.norm(x, np.inf)
        n_iter += 1
        x0 = np.copy(x)

    return x, stima_errore, n_iter


def gauss_seidel_decomposizione(A, b, x0, toll, iter_max, verbose=False):
    """
    Risoluzione di un sistema lineare Ax=b data un'approssimazione iniziale x0.
    L'algoritmo implementa il metodo iterativo di Gauss Seidel usando la decomposizione 
    della matrice dei coefficienti A = L + U.

    :param A: matrice dei coefficienti
    :param b: vettore colonna dei termini noti
    :param x0: approssimazione iniziale
    :param toll: tolleranza, ovvero precisione relativa richiesta
    :param n_max: massimo numero di iterazioni
    :param verbose: se True, stampa informazioni sulle iterazioni
    :return: soluzione x, stima dell'errore, numero di iterazioni eseguite
    :rtype: np.array, float, int
    """

    # calcolo matrici L=lower e U=upper
    lower = np.tril(A)
    upper = A - lower

    # Inizializzazioni
    n_iter = 0
    x = np.copy(x0)

    # Processo iterativo
    while n_iter < iter_max:
        x = np.dot(np.linalg.inv(lower), (b - np.dot(upper, x0)))
        n_iter += 1
        stima_errore = np.linalg.norm(x-x0, np.inf) / np.linalg.norm(x, np.inf)
        if stima_errore <= toll:
            break
        x0 = np.copy(x)

    return x, stima_errore, n_iter


def gauss_seidel_decomposizione_ottimizzato(A, b, x0, toll, iter_max, verbose=False):
    """
    Risoluzione di un sistema lineare Ax=b data un'approssimazione iniziale x0.
    L'algoritmo implementa il metodo iterativo di Gauss Seidel usando la decomposizione 
    della matrice dei coefficienti A = L + U.

    :param A: matrice dei coefficienti
    :param b: vettore colonna dei termini noti
    :param x0: approssimazione iniziale
    :param toll: tolleranza, ovvero precisione relativa richiesta
    :param n_max: massimo numero di iterazioni
    :param verbose: se True, stampa informazioni sulle iterazioni
    :return: soluzione x, stima dell'errore, numero di iterazioni eseguite
    :rtype: np.array, float, int
    """

    # calcolo matrici L=lower e U=upper
    lower = np.tril(A)
    upper = A - lower

    # Inizializzazioni
    n_iter = 0
    x = np.copy(x0)

    # Matrici iterative
    B = np.dot(np.linalg.inv(lower), b)
    c = np.dot(np.linalg.inv(lower), upper)

    # Processo iterativo
    while n_iter < iter_max:
        x = B - np.dot(c, x0)
        n_iter += 1
        stima_errore = np.linalg.norm(x-x0, np.inf) / np.linalg.norm(x, np.inf)
        if stima_errore <= toll:
            break
        x0 = np.copy(x)

    return x, stima_errore, n_iter
