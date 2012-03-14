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
from sklearn.utils import check_random_state


def fit_grid_point(X, y, base_clf, params, train, test, loss_func,
                   score_func, param_id=None):
    """Run fit on one set of parameters

    Returns the parameter set identifier, the parameters and the
    computed score.

    """
    # update parameters of the classifier after a copy of its base structure
    clf = copy.deepcopy(base_clf)
    clf.set_params(**params)

    # TODO: factor out the following in scikit-learn and make it possible to
    # memoize it as a joblib memory mapped file
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
    """Grid search on the parameters of an estimator w/ a scoring function

    The grid search runs asynchronously on the engines of the cluster. The
    client python process can fetch and interact with the partial results
    of the cluster engines while the computation is ongoing.

    The exploration ordering is randomized as recommended by Bergstra
    and Bengio 2012:

      http://people.fas.harvard.edu/~bergstra/random-search.html
      http://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf

    This makes it possible to early introspect the partial results
    without having the first scores being biased towards any particular
    area of the search space and also to stop and restart the computation
    as each job is completely independent of one another.

    """

    def __init__(self, estimator, param_grid, loss_func=None, score_func=None,
                 randomized=True, cv=None, view=None, random_state=None):
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
        self.random_state = random_state
        self.randomized = randomized

    def fit_async(self, X, y=None):
        """Run fit asynchronously with all sets of parameters

        Parameters
        ----------

        X: array or sparse matrix, [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y: array-like, shape = [n_samples], optional
            Target vector relative to X for classification;
            None for unsupervised learning.

        """
        import uuid
        if self.is_running():
            raise RuntimeError("Cannot launch new tasks while the previous"
                               " tasks are still running.")

        session_id = "s_" + uuid.uuid4().hex
        random_state = check_random_state(self.random_state)
        py_random_state = random.Random(random_state.rand())

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

        self.X, self.y = X, y

        grid = IterGrid(self.param_grid)
        if self.randomized:
            # shuffle the grid to implement James Bergstra's randomized
            # search
            py_random_state.shuffle(list(grid))

        base_clf = clone(self.estimator)

        # push data into engines namespaces
        @parallel.util.interactive
        def push_data(X, y, session_id):
            data = dict(X=X, y=y)
            g = globals()
            g.update({session_id + '_data': data})

        push_ars = []
        engine_ids = self.view.targets or self.view.client.ids
        for id in engine_ids:
            with self.view.temp_flags(targets=[id]):
                push_ars.append(self.view.apply_async(
                    push_data, X, y, session_id))

        self._push_results = push_ars

        # load balance the fit tasks themselves as soon at the data is
        # available on the any engines using a dependency in the DAG
        self.view.follow = parallel.Dependency(push_ars, all=False)
        fit_ars = []
        rX = parallel.Reference('%s_data["X"]' % session_id)
        ry = parallel.Reference('%s_data["y"]' % session_id)
        for param_id, clf_params in enumerate(grid):
            for train,test in cv:
                fit_ars.append(self.view.apply_async(fit_grid_point,
                        rX, ry, base_clf, clf_params, train, test,
                        self.loss_func, self.score_func,
                        param_id=param_id))

        # clean up data from engines namespaces
        self.view.follow = None
        self.view.after = parallel.Dependency(fit_ars, all=True)

        @parallel.util.interactive
        def cleanup_namespace(session_id):
            variable_name = session_id + '_data'
            g = globals()
            del g[variable_name]

        cleanup_ars = []
        for id in engine_ids:
            with self.view.temp_flags(targets=[id]):
                cleanup_ars.append(self.view.apply_async(
                    cleanup_namespace, session_id))

        # clear folllow dep
        self.view.after = None

        self._fit_results = fit_ars
        return len(fit_ars)

    def wait_for_completion(self, timeout=None):
        """Block until completion of the running tasks"""
        [ ar.get(timeout) for ar in self._fit_results ]

    def is_running(self):
        """Return True is there existing unfinished task"""
        if self._fit_results is None:
            return False

        for ar in self._fit_results:
            if not ar.ready():
                return True

        return False

    def collect_results(self):
        """Collect the scores of the  of the

        Return (results, n_remaining).
        """
        if self._fit_results is None:
            raise RuntimeError("Run fit_async before collecting its results")

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
                 np.array(durations))
            )

        # sort by mean score
        results.sort(key=itemgetter(1), reverse=True)
        return results, len(not_ready)

    def refit_best(self):
        """Fit the best parameter locally on the full dataset"""
        scores, _ = self.collect_results()
        if len(scores) == 0:
            raise RuntimeError("No parameter available")
        params, _, _, _,  _ = scores[0]
        model = clone(self.estimator)
        model.set_params(**params)
        model.fit(self.X, self.y)
        return model
