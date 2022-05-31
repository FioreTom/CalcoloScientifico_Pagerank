import numpy as np
import metodoEliminazioneGauss as meg
import backSubstitution as bs

A = np.array([[2.0, -3.0, 4.0],
              [1.0, 6.0, -3.0],
              [-3.0, 4.0, 8.0]],
             dtype=float)
b = np.array([[3.0],
              [4.0],
              [-32.0]],
             dtype=float)
print("Matrice completa del sistema:")
print(np.append(A, b, 1))
print("Indice di condizionamento:", np.linalg.cond(A))

xVero = np.array([[4],
                  [-1],
                  [-2]],
                 dtype=float)
print("Soluzione carta e penna, det =", np.linalg.det(A))
print(xVero, )

U_naive, c_naive, det_naive = meg.MEG_naive(A, b)
if det_naive is not None and det_naive != 0:
    print()
    x_naive = bs.back_substitution(U_naive, c_naive)
    print("Soluzione MEG_naive + bs, det =", det_naive)
    print(x_naive)
    errRel = np.linalg.norm(x_naive-xVero, np.inf) / \
        np.linalg.norm(xVero, np.inf)
    print("errore relativo con norma inf =", errRel)

U, c, det = meg.MEG_pivoting(A, b)
if det is not None and det != 0:
    print()
    x = bs.back_substitution(U, c)
    print("Soluzione MEG_pivoting + bs, det =", det)
    print(x)
    errRel = np.linalg.norm(x-xVero, np.inf)/np.linalg.norm(xVero, np.inf)
    print("errore relativo con norma inf =", errRel)


# xNumpy = np.linalg.solve(A, b)
# print("Soluzione numpy.linalg.solve:")
# print(xNumpy)
