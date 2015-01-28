import os
import re
import string
import nltk
import math
from random import shuffle
from sklearn import svm
import numpy
from my_record import MyRecord

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
        records.append(record)
        file_p = open(filename, 'r')
        record.process(file_p)
        file_p.close()

  # for record in records:
  #   print(record.intervals)
    # print(record.my_class)
    # print(record.filename)
    # print(record.features)

def report(test_split_size):
  featuresets = list()
  for record in records:
    featuresets.append((record.features, record.my_class))
  shuffle(featuresets)
  split = int(len(featuresets) / 10) * test_split_size
  train_set = featuresets[split:]
  test_set = featuresets[:split]
  print("split:", test_split_size, "train:", len(train_set), "test:", split)

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

  classifier = svm.SVC()
  classifier.fit(svm_train_features, svm_train_classes)
  print("svm accuracy: " +
        str(classifier.score(svm_test_features, svm_test_classes)))

parse()
report(4)
report(4)
report(4)
report(4)
report(4)
