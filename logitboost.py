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

class MyBoostClassifier(ClassifierMixin):
  def __init__(self, estimators):
    self.estimators = estimators

  def score(self, features, classes):
    total_score = 0
    for classifier in self.estimators:
      total_score += classifier.score(test_features, test_classes)
    return total_score / len(self.estimators)

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

  # classifier_list = list()
  # for index in range(15):
  #   temp_classifier = SVC(kernel="linear", C=0.1)
  #   temp_classifier.fit(train_features, train_classes)
  #   classifier_list.append(temp_classifier)

  # classifier = MyBoostClassifier(classifier_list)
  # # classifier.fit(train_features, train_classes)
  # print("adaboost accuracy: " +
  #       str(classifier.score(test_features, test_classes)))

  classifier = LogitBoostClassifier(n_estimators=50,
                                    base_estimator=SVC(kernel="linear", C=0.1),
                                    algorithm='SAMME')
  classifier.fit(numpy.array(train_features), numpy.array(train_classes))
  print("logitboost accuracy: " +
        str(classifier.score(numpy.array(test_features),
                                 numpy.array(test_classes))))

# default_estimators = {
#   "DT": 1,
#   "SVM": 1,
#   "KNN": 1,
#   "NB": 1,
# }

# estimator_funcs = {
#   "DT": LogitBoostClassifier.build_dt,
#   "SVM": LogitBoostClassifier.build_svm,
#   "KNN": LogitBoostClassifier.build_knn,
#   "NB": LogitBoostClassifier.build_nb,
# }

# class LogitBoostClassifier(ClassifierMixin):
#   # LogitBoost Classifier with multiple sklearn implemented classifiers
#   # as weak learners.
#   def __init__(self, estimators_dict=default_estimators):
#     self.estimators_dict = estimators_dict

#   def num_estimators(self):
#     return sum(self.estimators_dict.values())

#   def fit(self, x, y):
#     sample_weight = np.empty(x.shape[0], dtype=np.float)
#     sample_weight[:] = 1. / x.shape[0]
#     self.estimators = list()

#     for estimator in self.estimators_dict.keys():
#       estimator_funcs[estimator](self.estimators_dict[estimator])

#   def build_dt(self, amount):
#     for index in amount:
#       self.estimators.append(DecisionTreeClassifier())

#   def build_svm(self, amount):
#     for index in amount:
#       self.estimators.append(SVC())

#   def build_knn(self, amount):
#     for index in amount:
#       self.estimators.append(KNeighborsClassifier())

#   def build_nb(self, amount):
#     for index in amount:
#       self.estimators.append(GaussianNB())

load()
report()
