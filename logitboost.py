import copy
import operator
import numpy
from random import shuffle
from my_record import MyRecord, interval_base
import pickle

from sklearn.base import ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from logitboost_sample import LogitBoostClassifier
# decision tree, svm, knn, naive bayes

rootdir = '/Users/luke/Documents/ekg_analysis/zagreus/samples/'
repositories = ['SDDB_RR', 'NORM_RR', 'VF_RR', 'VT_RR']
records = list()

def load():
  my_records = pickle.load(open("records.pickle", "rb"))
  for record in my_records:
    records.append(record)

class MyBoostClassifier(LogitBoostClassifier):
  def __init__(self,
               base_estimator=DecisionTreeClassifier(max_depth=1),
               n_estimators=50,
               estimator_params=tuple(),
               learning_rate=1.,
               algorithm='SAMME.R'):
    self.base_estimator = base_estimator
    self.n_estimators = n_estimators
    self.estimator_params = estimator_params
    self.learning_rate = learning_rate
    self.algorithm = algorithm

  def set_estimators(self, estimators):
    self.n_estimators = len(estimators)
    self.estimators_ = numpy.array(estimators)

  def fit(self, X, y, sample_weight=None):
    # Check parameters.
    if self.learning_rate <= 0:
      raise ValueError("learning_rate must be greater than zero.")

    if sample_weight is None:
      # Initialize weights to 1 / n_samples.
      sample_weight = numpy.empty(X.shape[0], dtype=numpy.float)
      sample_weight[:] = 1. / X.shape[0]
    else:
      # Normalize existing weights.
      sample_weight = sample_weight / sample_weight.sum(dtype=numpy.float64)

    # Check that the sample weights sum is positive.
    if sample_weight.sum() <= 0:
      raise ValueError(
        "Attempting to fit with a non-positive "
        "weighted number of samples.")

    # Clear any previous fit results.
    # self.estimators_ = []
    self.estimator_weights_ = numpy.zeros(self.n_estimators, dtype=numpy.float)
    self.estimator_errors_ = numpy.ones(self.n_estimators, dtype=numpy.float)

    for (iboost, estimator) in enumerate(self.estimators_):
      # Fit the estimator.
      estimator.fit(X, y, sample_weight=sample_weight)

      if iboost == 0:
        self.classes_ = getattr(estimator, 'classes_', None)
        self.n_classes_ = len(self.classes_)

      # Generate estimator predictions.
      y_pred = estimator.predict(X)

      # Instances incorrectly classified.
      incorrect = y_pred != y

      # Error fraction.
      estimator_error = numpy.mean(
        numpy.average(incorrect, weights=sample_weight, axis=0))

      # Boost weight using multi-class AdaBoost SAMME alg.
      estimator_weight = self.learning_rate * (
        numpy.log((1. - estimator_error) / estimator_error) +
        numpy.log(self.n_classes_ - 1.))

      # Only boost the weights if there is another iteration of fitting.
      if not iboost == self.n_estimators - 1:
        # Only boost positive weights (logistic loss).
        sample_weight *= numpy.log(1 + numpy.exp(estimator_weight * incorrect *
                                   ((sample_weight > 0) |
                                    (estimator_weight < 0))))

      self.estimator_weights_[iboost] = estimator_weight
      self.estimator_errors_[iboost] = estimator_error

def random_split(test_split_size):
  scd_count = 0
  for record in records:
    if (record.my_class == "SCD"):
      scd_count += 1
  print(scd_count)

  shuffle(records)
  split = int(len(records) * (1 / test_split_size))
  print(len(records))
  train_set = records[:(len(records) - split)]
  test_set = records[split:]
  print("split:", test_split_size, "train:", len(train_set), "test:", split)
  return (train_set, test_set)

def report():
  (train_set, test_set) = random_split(3)
  train_features = list()
  train_classes = list()
  test_features = list()
  test_classes = list()

  for record in train_set:
    train_features.append(list(record.features.values()))
    train_classes.append(record.my_class)
  for record in test_set:
    test_features.append(list(record.features.values()))
    test_classes.append(record.my_class)

  svm_classifier = SVC(kernel="linear", C=0.1)
  svm_classifier.fit(train_features, train_classes)
  print("linear kernel svm accuracy: " +
        str(svm_classifier.score(test_features, test_classes)))

  classifier = MyBoostClassifier()
  estimator_list = numpy.array([SVC(kernel="linear", C=0.1, probability=True),
                                DecisionTreeClassifier(max_depth=1)])
  classifier.set_estimators(estimator_list)
  classifier.fit(numpy.array(train_features), numpy.array(train_classes))
  print("logitboost accuracy: " +
        str(classifier.score(numpy.array(test_features),
                             numpy.array(test_classes))))

load()
report()
