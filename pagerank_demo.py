from matplotlib import pyplot as plt
import numpy as np
import pathlib

import metodoEliminazioneGauss as meg
import backSubstitution as bs
import metodiIterativi as mi
import metodoPotenze as mp
import surfer

def main():
    root = "https://www.unisa.it"
    toll = 1e-15
    nMax = 1000
    depth = 200
    x0 = np.full((depth, 1), 1/depth)
    p = 0.85

    U, G = surfer.surfer(root, depth, verbose=False)

    pathFile = pathlib.Path(__file__).parent
    plt.figure(1)
    plt.imshow(G, interpolation="none")
    plt.savefig(pathFile / "spyG")
    
    A, Ap, b = surfer.pagerank(G, p)

    ################################
    print("Running linalg.solve...")
    x = np.linalg.solve(A, b)
   
    ################################
    print("Running MEG_naive...")
    U_naive, c_naive, det_naive = meg.MEG_naive(A, b)
    if det_naive is not None and det_naive != 0:
        x_naive = bs.back_substitution(U_naive, c_naive)
        errRel_naive = np.linalg.norm(
            x_naive-x, np.inf) / np.linalg.norm(x, np.inf)
        print("errRel_naive =", errRel_naive)   
    xSort = np.argsort(x_naive, axis=0).reshape(1, depth)[0]
    best = [U[i] for i in xSort]
    links = best[0] + " #" + str(xSort[0]) + "\n" + best[1] + " #" + str(xSort[1]) + "\n" + \
        best[-2] + " #" + str(xSort[-2]) + "\n" + \
        best[-1] + " #" + str(xSort[-1]) + "\n"
    print(links)

    ################################
    print("Running MEG_pivoting...")
    U_piv, c_piv, det_piv = meg.MEG_pivoting(A, b)
    if det_piv is not None and det_piv != 0:
        x_piv = bs.back_substitution(U_piv, c_piv)
        errRel_piv = np.linalg.norm(
            x_piv-x, np.inf) / np.linalg.norm(x, np.inf)
        print("errRel_piv =", errRel_piv)
    xSort = np.argsort(x_piv, axis=0).reshape(1, depth)[0]
    best = [U[i] for i in xSort]
    links = best[0] + " #" + str(xSort[0]) + "\n" + best[1] + " #" + str(xSort[1]) + "\n" + \
        best[-2] + " #" + str(xSort[-2]) + "\n" + \
        best[-1] + " #" + str(xSort[-1]) + "\n"
    print(links)

    ################################
    print("Running metodoJacobiDecomposizione...")
    xJacobi, stimaErroreJ, nIterJ = mi.jacobi_decomposizione_ottimizzato(
        A, b, x0, toll, nMax)
    errRelJ = np.linalg.norm(xJacobi-x, np.inf) / np.linalg.norm(x, np.inf)
    print("errRelJ =", errRelJ)
    print("stimaErroreJ =", stimaErroreJ)
    print("nIterJ =", nIterJ)
    xSort = np.argsort(xJacobi, axis=0).reshape(1, depth)[0]
    best = [U[i] for i in xSort]
    links = best[0] + " #" + str(xSort[0]) + "\n" + best[1] + " #" + str(xSort[1]) + "\n" + \
        best[-2] + " #" + str(xSort[-2]) + "\n" + \
        best[-1] + " #" + str(xSort[-1]) + "\n"
    print(links)

    ################################
    print("Running metodoGaussSeidel con decomposizione...")
    xGaussSeidel, stimaErroreGS, nIterGS = mi.gauss_seidel_decomposizione_ottimizzato(
        A, b, x0, toll, nMax)
    errRelGS = np.linalg.norm(xGaussSeidel-x, np.inf) / \
        np.linalg.norm(x, np.inf)
    print("errRelGS =", errRelGS)
    print("stimaErroreGS =", stimaErroreGS)
    print("nIterGS =", nIterGS)
    xSort = np.argsort(xGaussSeidel, axis=0).reshape(1, depth)[0]
    best = [U[i] for i in xSort]
    links = best[0] + " #" + str(xSort[0]) + "\n" + best[1] + " #" + str(xSort[1]) + "\n" + \
        best[-2] + " #" + str(xSort[-2]) + "\n" + \
        best[-1] + " #" + str(xSort[-1]) + "\n"
    print(links)

    ################################
    print("Running metodoPotenze...")
    lambdaa, stimaErroreP, nIterP, v = mp.metodo_potenze(Ap, toll, nMax, x0)
    eigvalues = np.linalg.eigvals(A)
    i = np.argmax(np.abs(eigvalues))
    lambdaa_np = eigvalues[i]
    errRelP = np.abs(lambdaa_np-lambdaa)/abs(lambdaa_np)
    print("errRelP =", errRelP )
    print("stimaErroreP =", stimaErroreP)
    print("nIterP =", nIterP)
    xSort = np.argsort(v, axis=0).reshape(1, depth)[0]
    best = [U[i] for i in xSort]
    links = best[0] + " #" + str(xSort[0]) + "\n" + best[1] + " #" + str(xSort[1]) + "\n" + \
        best[-2] + " #" + str(xSort[-2]) + "\n" + \
        best[-1] + " #" + str(xSort[-1]) + "\n"
    print(links)

if __name__ == "__main__":
    main()
