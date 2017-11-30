import numpy as np

#From here:  https://stackoverflow.com/questions/31528800/how-to-implement-zca-whitening-python
def compute_zca_whitening(X,epsilon=1.e-1):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [N x M] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    N = len(X)
    X = np.transpose(X)
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = sigma = np.dot(X,X.T)/N#np.cov(X, rowvar=True) # [M x M]#np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    # epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix

def get_precomputed_zca(path):
    import h5py
    try:
        f = h5py.File(path)
        return f["ZCA"][:]
    except Exception as e:
        raise RuntimeError("No ZCA Matrix at %s" % path)
    f.close()

def store_zca(path,ZCAMatrix):
    import h5py
    f = h5py.File(path)
    del f['ZCA']
    f.create_dataset("ZCA",data=ZCAMatrix)
    f.close()