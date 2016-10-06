""" Alternating Least Squares for Collaborative Filtering
"""
# Author: Vladimir Larin <vladimir@vlarine.ru>
# License: MIT

import numpy as np

__all__ = ['ALS', ]

class ALS():
    """ Alternating Least Squares for Collaborative Filtering

    For now supports implicit ALS only.


    Parameters
    ----------
    n_components: int, optional, defalult: 15
        The number of components for factorisation.

    lambda_: float, optional, dedault: 0.01
        The regularisation parameter in ALS.

    alpha: int, optional, default: 15
        The parameter associated with the confidence matrix
        in the implicit ALS algorithm.

    n_iter: int, optional, default: 20
        The number of iterations of the ALS algorithm.

    method: 'implicit' | 'explicit', default: 'implicit'
        The ALS method. For now supports implicit ALS only.

    n_jobs: int, optional, default: 1
        The number of jobs to use for computation.
        For now supports 1 job only.

    random_state: int seed or None (default)
        Random number generator seed.


    References
    ----------
    Collaborative Filtering for Implicit Feedback Datasets.
    Yifan Hu. AT&T Labs – Research. Florham Park, NJ 07932.
    Yehuda Koren. Yahoo! Research.
    http://yifanhu.net/PUB/cf.pdf

    Ben Frederickson. Fast Python Collaborative Filtering
    for Implicit Datasets.
    https://github.com/benfred/implicit

    """

    def __init__(self, n_components=15, lambda_=0.01, alpha=15, n_iter=20,
                 method='implicit', n_jobs=1, random_state=None):
        self.n_components = n_components
        self.lambda_ = lambda_
        self.alpha = alpha
        self.n_iter = n_iter
        self.method = method
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X):
        """Learn an ALS model.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape (n_rows, n_columns)
            Data matrix to learn a model.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        self._als((np.array(X) * self.alpha).astype('double'))
        return self

    def predict(self, X):
        """Learn an ALS model.

        Parameters
        ----------
        X: iterable with two integers
            Pairs of row index, column index to predict.

        Returns
        -------
        pred : array
            Returns array of predictions.
        """

        pred = []
        for item in X:
            i = item[0]
            j = item[1]
            pred.append(self.rows_[i, :].dot(self.columns_[j, :]))
        return np.array(pred)

    def _nonzeros(self, m, row):
        """ returns the non zeroes of a row in csr_matrix """

        for index in range(m.indptr[row], m.indptr[row+1]):
            yield m.indices[index], m.data[index]

    def _als(self, Cui):

        dtype = np.float64

        self.n_rows_, self.n_columns_ = Cui.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.rows_ = np.random.rand(self.n_rows_, self.n_components).astype(dtype) * 0.01
        self.columns_ = np.random.rand(self.n_columns_, self.n_components).astype(dtype) * 0.01

        Cui, Ciu = Cui.tocsr(), Cui.T.tocsr()

        solver = self._least_squares

        for iteration in range(self.n_iter):
            solver(Cui, self.rows_, self.columns_, self.lambda_)
            solver(Ciu, self.columns_, self.rows_, self.lambda_)

    def _least_squares(self, Cui, X, Y, regularization):
        """ For each user in Cui, calculate factors Xu for them
        using least squares on Y.
        """

        users, factors = X.shape
        YtY = Y.T.dot(Y)

        for u in range(users):
            # accumulate YtCuY + regularization*I in A
            A = YtY + regularization * np.eye(factors)

            # accumulate YtCuPu in b
            b = np.zeros(factors)

            for i, confidence in self._nonzeros(Cui, u):
                factor = Y[i]
                A += (confidence - 1) * np.outer(factor, factor)
                b += confidence * factor

            # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
            X[u] = np.linalg.solve(A, b)
