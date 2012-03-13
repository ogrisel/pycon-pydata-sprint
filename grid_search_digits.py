"""Perform grid search on a IPython.parallel cluster"""
print __doc__

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
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

clf = IPythonGridSearchCV(SVC(C=1), tuned_parameters,
                          score_func=precision_score, view=v, cv=5)
clf.fit_async(X_train, y_train)
print "Launched asynchronous fit on a cluster."


def print_scores(scores):
    for params, mean_score, scores, mean_duration, durations in scores:
        print "%0.3f (+/-%0.03f) [%i] for %r (%0.3fs)" % (
            mean_score, scores.std() / 2, len(scores), params, mean_duration)

while v.outstanding:
    v.wait(timeout=0.5)
    completed_scores, n_remaining = clf.collect_results()
    top_scores = completed_scores[:3]

    print "Current top %d parameters on development set:" % len(top_scores)
    print
    print_scores(top_scores)
    print "%d tasks remaining" % n_remaining
    print

print "Final scores:"
print
all_scores, _ = clf.collect_results()
print_scores(all_scores)

#TODO: refit the best estimator
