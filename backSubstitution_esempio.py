import numpy as np
import backSubstitution as bs

A = np.array([  [2, 2, 4],
                [0, 7, 11],
                [0, 0, 2]   ],
             dtype=float)
print(A)
   
b = np.array([  [5],
                [8],
                [2] ],
             dtype=float)
print(b)    

x = bs.back_substitution(A, b) 
print(x)    
