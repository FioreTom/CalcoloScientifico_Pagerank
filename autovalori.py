import numpy as np


A = np.array([  [1, 2, 3],
                [3, 2, 1], 
                [1, 0, -1]])

w, v = np.linalg.eig(A)
print(w)
print(v)

u = v[:,1] # Estrazione del secondo autovettore
print(u)
lam = w[1] # Estrazione del secondo autovalore
print(lam)

print(np.dot(A,u))	# Calcolo e stampa di 𝐴𝑥
print(lam*u)	    # Calcolo e stampa di 𝜆𝑥

# Ordinamento degli autovalori
w, v = np.linalg.eig(A)
idx = np.argsort(w)
w = w[idx]
v = v[:,idx]
