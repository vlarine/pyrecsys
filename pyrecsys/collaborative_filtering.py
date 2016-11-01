""" Alternating Least Squares for Collaborative Filtering
"""
# Author: Vladimir Larin <vladimir@vlarine.ru>
# License: MIT

import numpy as np
import scipy.sparse as sp
import six

GOT_NUMBA = True
try:
    from pyrecsys._polara.lib.hosvd import tucker_als
except ImportError:
    GOT_NUMBA = False

__all__ = ['ALS', ]


###############################################################################
def _pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'
    Parameters
    ----------
    params: dict
        The dictionary to pretty print
    offset: int
        The offset in characters to add at the begin of each line.
    printer:
        The function to convert entries to strings, typically
        the builtin str or repr
    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(six.iteritems(params))):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines


###############################################################################
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

    method: 'implicit' | 'explicit' | 'polara', default: 'implicit'
        The ALS method. For now supports implicit ALS only.

    rank: int, optional, default: 5
        Polara-specific. Base rating rank.

    growth_tol: float, optional, dedault: 0.0001
        Polara-specific. Threshold for early stopping.

    mlrank: (int, int, int), optional, default: (13, 10, 2)
        Polara-specific. Tuple of model ranks.

    n_jobs: int, optional, default: 1
        The number of jobs to use for computation.
        For now supports 1 job only.

    random_state: int seed or None (default)
        Random number generator seed.

    verbose: int, optional (default=0)
        Controls the verbosity of the model building process.

    References
    ----------
    Collaborative Filtering for Implicit Feedback Datasets.
    Yifan Hu. AT&T Labs – Research. Florham Park, NJ 07932.
    Yehuda Koren. Yahoo! Research.
    http://yifanhu.net/PUB/cf.pdf

    Ben Frederickson. Fast Python Collaborative Filtering
    for Implicit Datasets.
    https://github.com/benfred/implicit

    Evgeny Frolov, Ivan Oseledets. Fifty Shades of Ratings: How to Benefit
    from a Negative Feedback in Top-N Recommendations Tasks.
    https://arxiv.org/abs/1607.04228
    https://github.com/Evfro/polara

    """

    def __init__(self, n_components=15, lambda_=0.01, alpha=15, n_iter=20,
                 method='implicit', n_jobs=1, rank=5, growth_tol=0.0001,
                 mlrank=(13, 10, 2), random_state=None, verbose=0):
        self.n_components = n_components
        self.lambda_ = lambda_
        self.alpha = alpha
        self.n_iter = n_iter
        self.method = method
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.rank = rank
        self.mlrank = mlrank
        self.growth_tol = growth_tol
        self.verbose = verbose
        self._eps = 0.0000001 # Small value for evoid a division by zero

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

        if self.method == 'implicit':
            self._als(X * self.alpha)
        elif self.method == 'explicit':
            self._als(X)
        elif self.method == 'polara':
            if GOT_NUMBA:
                self._polara_als(X)
            else:
                raise ImportError('Numba is not installed')
        else:
            raise NotImplementedError('Method {} is not implemented.'.format(self.method))


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
        if self.method == 'polara':
            u, v, w, c = self.rows_, self.columns_, self.feedback_factors_, self.core_
            for item in X:
                i = item[0]
                j = item[1]
                if i < u.shape[0] and j < v.shape[0]:
                    p = v[j, :].dot(c.T.dot(u[i, :]).T).dot(w.T).argmax()
                else:
                    p = (self.rank - 1) / 2
                p = p * (self.x_max_ - self.x_min_) / (self.rank + self._eps) + self.x_min_
                pred.append(p)
        else:
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

        if self.method == 'implicit':
            solver = self._implicit_least_squares
        elif self.method == 'explicit':
            solver = self._explicit_least_squares
        else:
            raise NotImplementedError('Method {} is not implemented.'.format(self.method))

        for iteration in range(self.n_iter):
            solver(Cui, self.rows_, self.columns_, self.lambda_)
            solver(Ciu, self.columns_, self.rows_, self.lambda_)

    def _polara_als(self, Cui):
        Cui = sp.coo_matrix(Cui)
        self.x_min_ = Cui.data.min()
        self.x_max_ = Cui.data.max()
        Cui.data -= self.x_min_
        if self.x_max_ > self.x_min_:
            Cui.data /= (self.x_max_ - self.x_min_)
        Cui.data *= (self.rank - self._eps)

        Cui = np.ascontiguousarray(np.transpose(np.array((Cui.row, Cui.col, Cui.data), dtype=np.int64)))
        shp = tuple(Cui.max(axis=0) + 1)
        val = np.ascontiguousarray(np.ones(Cui.shape[0], ))

        users_factors, items_factors, feedback_factors, core = \
                            tucker_als(Cui, val, shp, self.mlrank,
                            growth_tol=self.growth_tol,
                            iters=self.n_iter,
                            batch_run=False if self.verbose else True)
        self.rows_ = users_factors
        self.columns_ = items_factors
        self.feedback_factors_ = feedback_factors
        self.core_ = core



    def _explicit_least_squares(self, Cui, X, Y, regularization):
        users, factors = X.shape
        YtY = Y.T.dot(Y)

        for u in range(users):
            # accumulate YtCuY + regularization*I in A
            A = YtY + regularization * np.eye(factors)

            # accumulate YtCuPu in b
            b = np.zeros(factors)

            for i, confidence in self._nonzeros(Cui, u):
                factor = Y[i]
                b += confidence * factor

            X[u] = np.linalg.solve(A, b)

    def _implicit_least_squares_(self, Cui, X, Y, regularization):

        users, factors = X.shape
        YtY = Y.T.dot(Y)

        for u in range(users):
            indexes = [x[0] for x in self._nonzeros(Cui, u)]
            if len(indexes) > 0:
                Hix = Y[indexes, :]
                M = YtY + self.alpha * Hix.T.dot(Hix) + np.diag(self.lambda_ * np.eye(factors))
                X[u] = np.dot(np.linalg.inv(M), (1 + self.alpha) * Hix.sum(axis=0))
            else:
                X[u] = np.zeros(factors)

    def _implicit_least_squares(self, Cui, X, Y, regularization):
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

    def get_params(self):
        """Get parameters for this model.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()

        names = ['n_components', 'lambda_', 'alpha', 'n_iter',\
                 'method', 'n_jobs', 'random_state', 'verbose']
        if self.method == 'polara':
            names += ['rank', 'mlrank', 'growth_tol']

        for key in names:
            out[key] = getattr(self, key, None)

        return out



    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(),
                                               offset=len(class_name),),)



