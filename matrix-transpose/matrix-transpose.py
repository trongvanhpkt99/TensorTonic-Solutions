import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    N, M = A.shape
    
    # tạo matrix kết quả (M, N)
    result = np.zeros((M, N), dtype=A.dtype)
    
    # fill bằng loop
    for i in range(N):
        for j in range(M):
            result[j, i] = A[i, j]
    return result