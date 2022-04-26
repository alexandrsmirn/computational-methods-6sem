import numpy as np
from scipy import linalg
import pandas as pd


def cond_spectr(A: np.ndarray) -> float:
    return linalg.norm(A) * linalg.norm(linalg.inv(A))


def cond_vol(A: np.ndarray) -> float:
    dividend = 1.0
    for row in A:
        dividend = dividend * linalg.norm(row, 2)

    return dividend / abs(linalg.det(A))


def cond_angle(A: np.ndarray) -> float:
    A_inv = linalg.inv(A)
    max_val = 0
    for i in range(A.shape[0]):
        curr_val = linalg.norm(A[i]) * linalg.norm(A_inv[..., i])
        max_val = max(max_val, curr_val)

    return max_val


def hilbert_experiment():
    dim = 10
    round_val = 10

    res = np.ones((dim-1, round_val-1), dtype=np.float64)
    cond_nums = np.ones((dim-1, 3), dtype=np.float64)
    for n in range(2, dim + 1):
        h = linalg.hilbert(n)
        x = np.ones(n, dtype=np.float64)
        b = h @ x
        cond_nums[n-2] = np.array([cond_spectr(h), cond_vol(h), cond_angle(h)])
        for r in range (2, round_val + 1):
            x_bad = linalg.solve(h.round(decimals=r), b.round(decimals=r))
            res[n-2, r-2] = linalg.norm(x - x_bad)
    #print(pd.DataFrame(res))
    print(cond_nums)

#print("Cond. spectr:\t", cond_spectr(H))
#print("Cond. volume:\t", cond_vol(H))
#print("Cond. angle:\t", cond_angle(H))
#print("Error norm:\t", linalg.norm(error))

def pak_experiment():
    pak_matr = np.array([[-402.5, 200.5],\
                        [1203.0, -603.0]])

    print("Cond. spectr:\t", cond_spectr(pak_matr))
    print("Cond. volume:\t", cond_vol(pak_matr))
    print("Cond. angle:\t", cond_angle(pak_matr))

    b = np.array([200, -600])
    b_var = np.array([199, -601])

    x = linalg.solve(pak_matr, b)
    x_var = linalg.solve(pak_matr, b_var)
    err = linalg.norm(x - x_var) / linalg.norm(x)
    print("Error:\t", err)

def tridiag_experiment():
    tri_matr = np.array([[1, 4, 0, 0],\
                         [3, 4, 1, 0],\
                         [0, 2, 3, 4],\
                         [0, 0, 1, 3]])

    print("Cond. spectr:\t", cond_spectr(tri_matr))
    print("Cond. volume:\t", cond_vol(tri_matr))
    print("Cond. angle:\t", cond_angle(tri_matr))

    b = np.array([5, 6, 7, 8], dtype=np.float64)
    b_var = b + np.random.uniform(low=-0.5, high=0.5, size=b.shape)

    x = linalg.solve(tri_matr, b)
    x_var = linalg.solve(tri_matr, b_var)
    err = linalg.norm(x - x_var) / linalg.norm(x)
    print("Error:\t", err)