"""IPython.parallel implementation for Randomized Cross-Validated Grid Seach


"""


# Author: MinRK <benjaminrk@gmail.com>
#         Olivier Grisel <olivier.grisel@gmail.com>
# License: BSD Style.

import copy
import random
from collections import defaultdict
from operator import itemgetter

import numpy as np
import scipy.sparse as sp

from IPython import parallel

from sklearn.grid_search import IterGrid
from sklearn.base import clone, is_classifier
from sklearn.cross_validation import check_cv


def fit_grid_point(X, y, base_clf, params, train, test, loss_func,
                   score_func, param_id=None):
    """Run fit on one set of parameters

    Returns the score and the instance of the classifier
    """
    # update parameters of the classifier after a copy of its base structure
    clf = copy.deepcopy(base_clf)
    clf.set_params(**params)

    if isinstance(X, list) or isinstance(X, tuple):
        X_train = [X[i] for i, cond in enumerate(train) if cond]
        X_test = [X[i] for i, cond in enumerate(test) if cond]
    else:
        if sp.issparse(X):
            # For sparse matrices, slicing only works with indices
            # (no masked array). Convert to CSR format for efficiency and
            # because some sparse formats don't support row slicing.
            X = sp.csr_matrix(X)
            ind = np.arange(X.shape[0])
            train = ind[train]
            test = ind[test]
        if hasattr(base_clf, 'kernel_function'):
            # cannot compute the kernel values with custom function
            raise ValueError(
                "Cannot use a custom kernel function. "
                "Precompute the kernel matrix instead.")
        if getattr(base_clf, 'kernel', '') == 'precomputed':
            # X is a precomputed square kernel matrix
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square kernel matrix")
            X_train = X[np.ix_(train, train)]
            X_test = X[np.ix_(test, train)]
        else:
            X_train = X[train]
            X_test = X[test]
    if y is not None:
        y_test = y[test]
        y_train = y[train]
    else:
        y_test = None
        y_train = None

    clf.fit(X_train, y_train)

    if loss_func is not None:
        y_pred = clf.predict(X_test)
        this_score = -loss_func(y_test, y_pred)
    elif score_func is not None:
        y_pred = clf.predict(X_test)
        this_score = score_func(y_test, y_pred)
    else:
        this_score = clf.score(X_test, y_test)
    return param_id, params, this_score


class IPythonGridSearchCV(object):
    """Grid search on the parameters of an estimator w/ a scoring function"""

    def __init__(self, estimator, param_grid, loss_func=None, score_func=None,
                 cv=None, view=None):
        if not hasattr(estimator, 'fit') or \
           not (hasattr(estimator, 'predict') or hasattr(estimator, 'score')):
            raise TypeError("estimator should a be an estimator implementing"
                            " 'fit' and 'predict' or 'score' methods,"
                            " %s (type %s) was passed" %
                            (estimator, type(estimator)))
        if loss_func is None and score_func is None:
            if not hasattr(estimator, 'score'):
                raise TypeError(
                    "If no loss_func is specified, the estimator passed "
                    "should have a 'score' method. The estimator %s "
                    "does not." % estimator)

        self.estimator = estimator
        self.param_grid = param_grid
        self.loss_func = loss_func
        self.score_func = score_func
        self.cv = cv
        self.view = view
        self._push_results = None
        self._fit_results = None

    def fit_async(self, X, y=None):
        """Run fit asynchronously with all sets of parameters

        Returns the best classifier

        Parameters
        ----------

        X: array, [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y: array-like, shape = [n_samples], optional
            Target vector relative to X for classification;
            None for unsupervised learning.

        """
        import os
        import binascii
        
        estimator = self.estimator
        cv = self.cv
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
        else:
            # support list of unstructured objects on which feature
            # extraction will be applied later in the tranformer chain
            n_samples = len(X)
        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
            y = np.asarray(y)
        cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

        grid = IterGrid(self.param_grid)
        random.shuffle(list(grid))
        base_clf = clone(self.estimator)
        
        suffix = binascii.hexlify(os.urandom(10))
        
        @parallel.util.interactive
        def push_data(X, y, suffix):
            data = dict(X=X, y=y)
            g = globals()
            g.update({'data_'+suffix : data})
        
        push_ars = []
        ids = self.view.targets or self.view.client.ids
        for id in ids:
            with self.view.temp_flags(targets=[id]):
                push_ars.append(self.view.apply_async(push_data, X, y, suffix))
        
        self._push_results = push_ars
        
        self.view.follow = parallel.Dependency(push_ars, all=False)
        
        ars = []
        rX = parallel.Reference('data_%s["X"]' % suffix)
        ry = parallel.Reference('data_%s["y"]' % suffix)
        for param_id, clf_params in enumerate(grid):
            for train,test in cv:
                ars.append(self.view.apply_async(fit_grid_point,
                        rX, ry, base_clf, clf_params, train, test,
                        self.loss_func, self.score_func,
                        param_id=param_id)
                )
        
        # clear folllow dep
        self.view.follow = None
        
        self._fit_results = ars
        return len(ars)

    def collect_results(self):
        
        if self._fit_results is None:
            raise RuntimeError("run fit() before collecting its results")
        
        self.view.spin()
        
        ready_results = [ ar for ar in self._fit_results if ar.ready() ]
        not_ready = [ ar for ar in self._fit_results if not ar.ready() ]
        
        runtime = lambda ar: (ar.completed - ar.started).total_seconds()
        
        grouped_scores = defaultdict(list)
        grouped_durations = defaultdict(list)
        parameters = dict()
        for ar in ready_results:
            param_id, clf_params, this_score = ar.get()
            
            grouped_scores[param_id].append(this_score)
            grouped_durations[param_id].append(runtime(ar))
            
            if param_id not in parameters:
                parameters[param_id] = clf_params
        
        results = []
        for param_id in grouped_scores:
            scores = grouped_scores[param_id]
            durations = grouped_durations[param_id]
            results.append(
                (parameters[param_id],
                 np.mean(scores),
                 np.array(scores),
                 np.mean(durations),
                 np.array(durations),
                )
            )
        
        # sort by mean score
        results.sort(key=itemgetter(1), reverse=True)
        return results, len(not_ready)
