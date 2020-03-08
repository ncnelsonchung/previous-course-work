import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def create_data(x1, x2, x3):
    x4 = -4.0 * x1
    x5 = 10 * x1 + 10
    x6 = -1 * x2 / 2
    x7 = np.multiply(x2, x2)
    x8 = -1 * x3 / 10
    x9 = 2.0 * x3 + 2.0
    X = np.hstack((x1, x2, x3, x4, x5, x6, x7, x8, x9))
    return X

def pca(X):
    '''
    # PCA step by step
    #   1. normalize matrix X
    #   2. compute the covariance matrix of the normalized matrix X
    #   3. do the eigenvalue decomposition on the covariance matrix
    # If you do not remember Eigenvalue Decomposition, please review the linear
    # algebra
    # In this assignment, we use the ``unbiased estimator'' of covariance. You
    # can refer to this website for more information
    # http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.cov.html
    # Actually, Singular Value Decomposition (SVD) is another way to do the
    # PCA, if you are interested, you can google SVD.
    # YOUR CODE HERE!
    '''
    X_norm = X - X.mean(axis=0)
    S = np.cov(X_norm.T, bias=False)
    V, D = np.linalg.eigh(S)
    V = np.flip(V.reshape(-1, 1), axis=0)
    D = np.flip(D, axis=1)
    ####################################################################
    # here V is the matrix containing all the eigenvectors, D is the
    # column vector containing all the corresponding eigenvalues.
    return [V, D]


def main():
    N = 1000
    shape = (N, 1)
    x1 = np.random.normal(0, 1, shape) # samples from normal distribution
    x2 = np.random.exponential(10.0, shape) # samples from exponential distribution
    x3 = np.random.uniform(-100, 100, shape) # uniformly sampled data points
    X = create_data(x1, x2, x3)

    ####################################################################
    # Use the definition in the lecture notes,
    #   1. perform PCA on matrix X
    #   2. plot the eigenvalues against the order of eigenvalues,
    #   3. plot POV v.s. the order of eigenvalues
    # YOUR CODE HERE!

    ####################################################################

    V, D = pca(X)
    #     print(X)
    #     print(V)
    #     print(D)
    plt.plot(range(1, len(V) + 1), V)
    plt.xlabel("Eigenvectors")
    plt.ylabel("Eigenvalues")
    plt.show()

    current = 0
    PoV = []
    for i in V:
        current += i[0]
        PoV.append(current / float(V.sum(axis=0)))
        # print(PoV)
    plt.plot(range(1, len(V) + 1), PoV)
    plt.xlabel("Eigenvectors")
    plt.ylabel("Prop. of var.")
    plt.show()


    # skpca = PCA()
    # skpca.fit(X)
    # print(np.array(skpca.explained_variance_ratio_).cumsum())


if __name__ == '__main__':
    main()

