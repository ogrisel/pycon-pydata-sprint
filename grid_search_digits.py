"""
=====================================================================
Parameter estimation using grid search with a nested cross-validation
=====================================================================

The classifier is optimized by "nested" cross-validation using the
:class:`sklearn.grid_search.GridSearchCV` object on a development set
that comprises only half of the available labeled data.

The performance of the selected hyper-parameters and trained model is
then measured on a dedicated evaluation set that was not used during
the model selection step.

More details on tools available for model selection can be found in the
sections on :ref:`cross_validation` and :ref:`grid_search`.

"""
print __doc__

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC

from grid_search import IPythonGridSearchCV

from IPython.parallel import Client
rc = Client()
v = rc.load_balanced_view()

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_fraction=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = [
    ('precision', ),
    ('recall', recall_score),
]

# for score_name, score_func in scores:
# print "# Tuning hyper-parameters for %s" % score_name
print

clf = IPythonGridSearchCV(SVC(C=1), tuned_parameters, score_func=precision_score, view=v)
clf.fit(X_train, y_train, cv=5)
print "fit submitted"
while v.outstanding:
    v.wait(timeout=0.1)
    grid_scores = clf.collect_results()
    
    # import IPython
    # IPython.embed()
    # 
    # print "Best parameters set found on development set:"
    # print
    # print clf.best_estimator_
    print
    print "Grid scores on development set:"
    print
    for params, mean_score, scores, mean_duration, durations in grid_scores[:3]:
        print "%0.3f (+/-%0.03f) [%i] for %r (took %0.3f s)" % (
            mean_score, scores.std() / 2, len(scores), params, mean_duration)
    print

    # print "Detailed classification report:"
    # print
    # print "The model is trained on the full development set."
    # print "The scores are computed on the full evaluation set."
    # print
    # y_true, y_pred = y_test, clf.predict(X_test)
    # print classification_report(y_true, y_pred)
    # print

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.
