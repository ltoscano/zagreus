import os
import re
import string
import nltk
import math
from random import shuffle
from sklearn import svm
import numpy
from my_record import MyRecord, interval_base
import pickle
from sklearn.ensemble import AdaBoostClassifier

rootdir = '/Users/luke/Documents/ekg_analysis/zagreus/samples/'
repositories = ['SDDB_RR', 'NORM_RR', 'VF_RR', 'VT_RR']
records = list()

def parse():
  for repo in repositories:
    for subdir, dirs, files in os.walk(rootdir + repo):
      for my_file in files:
        filename = os.path.join(subdir, my_file)
        record = MyRecord(repo, my_file)
        record.filename = my_file
        file_p = open(filename, 'r')
        record.process(file_p)
        file_p.close()
        if (len(record.intervals) > 0):
          records.append(record)

  pickle.dump(records, open("records.pickle", "wb"))

def load():
  my_records = pickle.load(open("records.pickle", "rb"))
  for record in my_records:
    records.append(record)

def split_sets(test_split_size):
  scd_count = 0
  for record in records:
    if (record.my_class == "SCD"):
      scd_count += 1
  print(scd_count)

  featuresets = list()
  for record in records:
    featuresets.append((record.features, record.my_class))
  shuffle(featuresets)
  split = int(len(featuresets) * (1 / test_split_size))
  print(len(records))
  train_set = featuresets[:(len(featuresets) - split)]
  test_set = featuresets[split:]
  print("split:", test_split_size, "train:", len(train_set), "test:", split)
  return (train_set, test_set)

def report(test_split_size):
  sets_split = split_sets(test_split_size)
  train_set = sets_split[0]
  test_set = sets_split[1]

  # classifier = nltk.NaiveBayesClassifier.train(train_set)
  # print("naive bayes accuracy: " +
  #       str(nltk.classify.accuracy(classifier, test_set)))

  # classifier = nltk.DecisionTreeClassifier.train(train_set)
  # print("tree accuracy: " + str(nltk.classify.accuracy(classifier, test_set)))

  svm_train_features = list()
  svm_train_classes = list()
  svm_test_features = list()
  svm_test_classes = list()

  for item in train_set:
    svm_train_features.append(list(item[0].values()))
    svm_train_classes.append(item[1])
  for item in test_set:
    svm_test_features.append(list(item[0].values()))
    svm_test_classes.append(item[1])


# sequential minimal optimization (SMO,QP and LS) and soft-margin (0.1 - 2.0).
# PNN  = only the spread value adjustment (0.1 - 3.0)

  # classifier = svm.LinearSVC()
  # classifier.fit(svm_train_features, svm_train_classes)
  # print("LinearSVC accuracy: " +
  #       str(classifier.score(svm_test_features, svm_test_classes)))

  classifier = svm.SVC(kernel="linear", C=0.1)
  classifier.fit(svm_train_features, svm_train_classes)
  print("linear kernel svm accuracy: " +
        str(classifier.score(svm_test_features, svm_test_classes)))

def display_avgs():
  scd_list = dict()
  norm_list = dict()
  scd_count = 0
  norm_count = 0

  for key in records[0].features:
    scd_list[key] = 0
    norm_list[key] = 0

  for record in records:
    if (record.my_class == "SCD"):
      for key in record.features:
        scd_list[key] += record.features[key]
        scd_count += 1
    else:
      for key in record.features:
        norm_list[key] += record.features[key]
        norm_count += 1

  for key in scd_list:
    scd_list[key] /= scd_count
    norm_list[key] /= norm_count

  print(scd_list)
  print(norm_list)

def boost_report(test_split_size):
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

  svm_train_features = list()
  svm_train_classes = list()
  svm_test_features = list()
  svm_test_classes = list()

  for record in train_set:
    svm_train_features.append(list(record.features.values()))
    svm_train_classes.append(record.my_class)
  for record in test_set:
    svm_test_features.append(list(record.features.values()))
    svm_test_classes.append(record.my_class)

  svm_classifier = svm.SVC(kernel="linear", C=0.1)
  svm_classifier.fit(svm_train_features, svm_train_classes)
  print("linear kernel svm accuracy: " +
        str(svm_classifier.score(svm_test_features, svm_test_classes)))

  classifier = AdaBoostClassifier(
    base_estimator=svm_classifier,
    n_estimators=50,
    algorithm='SAMME'
  )
  classifier.fit(svm_train_features, svm_train_classes)
  print("adaboost accuracy: " +
        str(classifier.score(svm_test_features, svm_test_classes)))

  classifier2 = AdaBoostClassifier(
    n_estimators=50,
    algorithm='SAMME'
  )
  classifier2.fit(svm_train_features, svm_train_classes)
  print("adaboost2 accuracy: " +
        str(classifier2.score(svm_test_features, svm_test_classes)))

# parse()
load()
# display_avgs()
boost_report(3)
# boost_report(3)
# boost_report(3)
# boost_report(3)
# boost_report(3)
