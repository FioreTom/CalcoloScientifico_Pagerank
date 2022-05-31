import numpy as np


def metodo_potenze_base(A: np.array, toll: float, n_iter_max: int, v0: np.array, verbose=False):
    """
    Calcolo dell’autovalore di modulo massimo e relativo autovettore.
    La crescita delle componenti può portare a errori di overflow

    :param A: matrice di cui calcolare l’autovalore di modulo massimo
    :param toll: tolleranza, ovvero precisione relativa richiesta
    :param n_iter_max: numeri massimo di iterazioni
    :param v0: vettore iniziale per approssimare l’autovettore
    :param verbose: se True, stampa informazioni sulle iterazioni
    :return: autovalore di modulo massimo lambdaa, stima Errore, numero di iterazioni eseguite, autovettore realitivo a lambdaa
    :rtype: float, float, int, np.array
    """

    n_iter = 0
    stima_errore = 1
    k =  1
    lambdaa0 = 1

    while n_iter < n_iter_max and stima_errore >= toll:
        v = np.dot(A, v0)
        lambdaa = v[k] / v0[k]
        stima_errore = np.abs(lambdaa-lambdaa0)/np.abs(lambdaa)
        lambdaa0 = lambdaa
        v0 = np.copy(v)
        n_iter += 1

    return lambdaa, stima_errore, n_iter, v



def metodo_potenze(A: np.array, toll: float, n_iter_max: int, y0: np.array, verbose=False):
    """
    Calcolo dell’autovalore di modulo massimo e relativo autovettore.

    :param A: matrice di cui calcolare l’autovalore di modulo massimo
    :param toll: tolleranza, ovvero precisione relativa richiesta
    :param n_iter_max: numeri massimo di iterazioni
    :param y0: vettore iniziale per approssimare l’autovettore
    :param verbose: se True, stampa informazioni sulle iterazioni
    :return: autovalore di modulo massimo lambdaa, stima Errore, numero di iterazioni eseguite, autovettore realitivo a lambdaa
    :rtype: float, float, int, np.array
    """

    n_iter = 0
    stima_errore = 1.0
    lambdaa0 = 0.0

    while n_iter < n_iter_max and stima_errore >= toll:
        w = np.dot(A, y0)
        k = np.argmax(np.abs(w))

        lambdaa = w[k] / y0[k]
        stima_errore = abs(lambdaa-lambdaa0)/abs(lambdaa)
        lambdaa0 = lambdaa

        y = w / np.linalg.norm(w, ord=1)
        y0 = np.copy(y)
        n_iter += 1
        if verbose:
            print("n_iter =", n_iter, ", lambda =", lambdaa)
            print(y.T)

    return lambdaa, stima_errore, n_iter, y


def metodo_potenze_minimo(A: np.array, toll: float, n_iter_max: int, y0: np.array, verbose=False):
    """
    Calcolo dell’autovalore di modulo minimo e relativo autovettore.

    :param A: matrice di cui calcolare l’autovalore di modulo minimo
    :param toll: tolleranza, ovvero precisione relativa richiesta
    :param n_iter_max: numeri massimo di iterazioni
    :param y0:  vettore iniziale per approssimare l’autovettore
    :param verbose: se True, stampa informazioni sulle iterazioni
    :return: autovalore di modulo minimo lambdaa, stima Errore, numero di iterazioni eseguite, autovettore realitivo a lambdaa
    :rtype: float, float, int, np.array
    """

    n_iter = 0
    stima_errore = 1.0
    lambdaa0 = 0.0

    try:
        A_inv = np.linalg.inv(A)
    except (np.linalg.LinAlgError) as err:
        print("ERRORE: ", err)
        print("a is not square or inversion fails")
        return

    while n_iter < n_iter_max and stima_errore >= toll:
        w = np.dot(A_inv, y0)
        k = np.argmax(np.abs(w))

        lambdaa = w[k] / y0[k]
        stima_errore = abs(lambdaa-lambdaa0)/abs(lambdaa)
        lambdaa0 = lambdaa

        y = w / np.linalg.norm(w, ord=1)
        y0 = np.copy(y)
        n_iter += 1
        if verbose:
            print("n_iter =", n_iter, ", lambda =", lambdaa)
            print(y.T)

    return 1/lambdaa, stima_errore, n_iter, y
