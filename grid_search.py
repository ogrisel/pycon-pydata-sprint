"""
The :mod:`sklearn.grid_search` includes utilities to fine-tune the parameters
of an estimator.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD Style.

import copy
from itertools import product
import random
import time
from collections import defaultdict
from operator import itemgetter

import numpy as np
import scipy.sparse as sp

from IPython import parallel

from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.cross_validation import check_cv
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.utils import deprecated


class IterGrid(object):
    """Generators on the combination of the various parameter lists given

    Parameters
    ----------
    param_grid: dict of string to sequence
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

    Returns
    -------
    params: dict of string to any
        **Yields** dictionaries mapping each estimator parameter to one of its
        allowed values.

    Examples
    --------
    >>> from sklearn.grid_search import IterGrid
    >>> param_grid = {'a':[1, 2], 'b':[True, False]}
    >>> list(IterGrid(param_grid)) #doctest: +NORMALIZE_WHITESPACE
    [{'a': 1, 'b': True}, {'a': 1, 'b': False},
     {'a': 2, 'b': True}, {'a': 2, 'b': False}]

    See also
    --------
    :class:`GridSearchCV`:
        uses ``IterGrid`` to perform a full parallelized grid search.
    """

    def __init__(self, param_grid):
        self.param_grid = param_grid

    def __iter__(self):
        param_grid = self.param_grid
        if hasattr(param_grid, 'items'):
            # wrap dictionary in a singleton list
            param_grid = [param_grid]
        for p in param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                yield params


def fit_grid_point(X, y, base_clf, clf_params, train, test, loss_func,
                   score_func, verbose, param_id=None, **fit_params):
    """Run fit on one set of parameters

    Returns the score and the instance of the classifier
    """
    if verbose > 1:
        start_time = time.time()
        msg = '%s' % (', '.join('%s=%s' % (k, v)
                                     for k, v in clf_params.iteritems()))
        print "[GridSearchCV] %s %s" % (msg, (64 - len(msg)) * '.')

    # update parameters of the classifier after a copy of its base structure
    # FIXME we should be doing a clone here
    clf = copy.deepcopy(base_clf)
    clf.set_params(**clf_params)

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

    clf.fit(X_train, y_train, **fit_params)

    if loss_func is not None:
        y_pred = clf.predict(X_test)
        this_score = -loss_func(y_test, y_pred)
    elif score_func is not None:
        y_pred = clf.predict(X_test)
        this_score = score_func(y_test, y_pred)
    else:
        this_score = clf.score(X_test, y_test)

    if y is not None:
        if hasattr(y, 'shape'):
            this_n_test_samples = y.shape[0]
        else:
            this_n_test_samples = len(y)
    else:
        if hasattr(X, 'shape'):
            this_n_test_samples = X.shape[0]
        else:
            this_n_test_samples = len(X)
    if verbose > 2:
        msg += ", score=%f" % this_score
    if verbose > 1:
        end_msg = "%s -%s" % (msg,
                              logger.short_format_time(time.time() -
                                                       start_time))
        print "[GridSearchCV] %s %s" % ((64 - len(end_msg)) * '.', end_msg)
    return param_id, clf_params, this_score


class IPythonGridSearchCV(BaseEstimator):
    """Grid search on the parameters of a classifier

    Important members are fit, predict.

    GridSearchCV implements a "fit" method and a "predict" method like
    any classifier except that the parameters of the classifier
    used to predict is optimized by cross-validation.

    Parameters
    ----------
    estimator: object type that implements the "fit" and "predict" methods
        A object of that type is instantiated for each grid point.

    param_grid: dict
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values.

    loss_func: callable, optional
        function that takes 2 arguments and compares them in
        order to evaluate the performance of prediciton (small is good)
        if None is passed, the score of the estimator is maximized

    score_func: callable, optional
        A function that takes 2 arguments and compares them in
        order to evaluate the performance of prediction (high is good).
        If None is passed, the score of the estimator is maximized.

    fit_params : dict, optional
        parameters to pass to the fit method

    n_jobs: int, optional
        number of jobs to run in parallel (default 1)

    pre_dispatch: int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediatly
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid: boolean, optional
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : integer or crossvalidation generator, optional
        If an integer is passed, it is the number of fold (default 3).
        Specific crossvalidation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    refit: boolean
        refit the best estimator with the entire dataset

    verbose: integer
        Controls the verbosity: the higher, the more messages.

    Examples
    --------
    >>> from sklearn import svm, grid_search, datasets
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svr = svm.SVC()
    >>> clf = grid_search.GridSearchCV(svr, parameters)
    >>> clf.fit(iris.data, iris.target)
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    GridSearchCV(cv=None,
        estimator=SVC(C=None, cache_size=..., coef0=..., degree=...,
            gamma=..., kernel='rbf', probability=False,
            scale_C=True, shrinking=True, tol=...),
        fit_params={}, iid=True, loss_func=None, n_jobs=1,
            param_grid=...,
            ...)

    Attributes
    ----------
    `grid_scores_` : dict of any to float
        Contains scores for all parameter combinations in param_grid.

    `best_estimator_` : estimator
        Estimator that was choosen by grid search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data.

    `best_score_` : float
        score of best_estimator on the left out data.

    Notes
    ------
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score_func is passed in which case it is used
    instead. If a loss function loss_func is passed, it overrides the score
    functions and is minimized.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is 2 *
    `n_jobs`.

    See Also
    ---------
    :class:`IterGrid`:
        generates all the combinations of a an hyperparameter grid.

    :func:`sklearn.cross_validation.train_test_split`:
        utility function to split the data into a development set usable
        for fitting a GridSearchCV instance and an evaluation set for
        its final evaluation.

    """

    def __init__(self, estimator, param_grid, loss_func=None, score_func=None,
                 fit_params=None, n_jobs=1, iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', view=None,
                ):
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
        self.n_jobs = n_jobs
        self.fit_params = fit_params if fit_params is not None else {}
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.view = view
        self._push_results = None
        self._fit_results = None

    def fit(self, X, y=None, **params):
        """Run fit with all sets of parameters

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
        import itertools
        
        self._set_params(**params)
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
        pre_dispatch = self.pre_dispatch
        
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
                        self.loss_func, self.score_func, self.verbose,
                        param_id=param_id, **self.fit_params)
                )
        
        # clear folllow dep
        self.view.follow = None
        
        self._fit_results = ars
        return self

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
        if not_ready:
            print "%i results remaining to compute" % len(not_ready)
        return results
        

    def score(self, X, y=None):
        if hasattr(self.best_estimator_, 'score'):
            return self.best_estimator_.score(X, y)
        if self.score_func is None:
            raise ValueError("No score function explicitly defined, "
                             "and the estimator doesn't provide one %s"
                             % self.best_estimator_)
        y_predicted = self.predict(X)
        return self.score_func(y, y_predicted)

    @property
    @deprecated('GridSearchCV.best_estimator is deprecated'
                ' and will be removed in version 0.12.'
                ' Please use ``GridSearchCV.best_estimator_`` instead.')
    def best_estimator(self):
        return self.best_estimator_

    @property
    @deprecated('GridSearchCV.best_score is deprecated'
                ' and will be removed in version 0.12.'
                ' Please use ``GridSearchCV.best_score_`` instead.')
    def best_score(self):
        return self.best_score_
