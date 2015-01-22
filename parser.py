import os
import re
import string
import nltk
import math
from random import shuffle
from sklearn import svm
import numpy

rootdir = '/Users/luke/Documents/ekg_analysis/zagreus/samples/'
repositories = ['SDDB_RR', 'NORM_RR']
records = list()

class MyRecord:

  def __init__(self, my_class):
    self.intervals = list()
    self.features = dict()
    self.my_class = my_class

  def process(self, file_p):
    for line in file_p.readlines():
      words = line.split()
      self.intervals.append(float(words[2]))
    self.generate_features()

  def generate_features(self):
    my_sum = 0
    my_index = 0
    my_max = 0
    my_min = 1000
    std_dev = 0

    for item in self.intervals:
      my_sum += item
      my_index += 1
      if (item > my_max):
        my_max = item
      if (item < my_min):
        my_min = item

    mean = float(my_sum / my_index)

    for item in self.intervals:
      std_dev += pow(item - mean, 2)
    std_dev = math.sqrt(std_dev / my_index)

    sorted_list = sorted(self.intervals)
    length = len(sorted_list)
    median = 0
    if (length % 2 == 0):
      index = int(length / 2)
      median = (sorted_list[index] + sorted_list[index + 1]) / 2
    else:
      index = int((length + 1) / 2)
      median = sorted_list[index]

    self.features['mean'] = mean
    self.features['min'] = my_min
    self.features['max'] = my_max
    self.features['std_dev'] = std_dev
    self.features['median'] = median

def parse():
  for repo in repositories:
    for subdir, dirs, files in os.walk(rootdir + repo):
      for my_file in files:
        filename = os.path.join(subdir, my_file)
        record = MyRecord(repo)
        record.filename = my_file
        records.append(record)
        file_p = open(filename, 'r')
        record.process(file_p)
        file_p.close()

  for record in records:
    # print(record.my_class)
    print(record.filename)
    print(record.features)

def report(test_split_size):
  featuresets = list()
  for record in records:
    featuresets.append((record.features, record.my_class))
  shuffle(featuresets)
  split = int(len(featuresets) / 10) * test_split_size
  train_set = featuresets[split:]
  test_set = featuresets[:split]
  print(test_split_size, split, len(train_set))

  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print("naive bayes accuracy: " + str(nltk.classify.accuracy(classifier, test_set)))

  classifier = nltk.DecisionTreeClassifier.train(train_set)
  print("tree accuracy: " + str(nltk.classify.accuracy(classifier, test_set)))

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
  print("svm accuracy: " + str(classifier.score(svm_test_features, svm_test_classes)))

parse()
report(2)
report(3)
report(4)
report(5)
