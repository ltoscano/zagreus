import os
import re
import string
import nltk
import math
from random import shuffle
from sklearn import svm
import numpy
from my_record import MyRecord
import pickle
from datetime import datetime, timedelta

rootdir = '/Users/luke/Documents/ekg_analysis/zagreus/samples/'
anno_dir = '/Users/luke/Documents/ekg_analysis/zagreus/samples/MIMIC_HR_ANN'
repositories = ['MIMIC_HR']

SECONDS_PER_MIN = 60.0

mim_records = list()
mit_records = list()

class MimicRecord(MyRecord):
  def __init__(self, my_file, filename):
    self.my_class = "SCD"
    file_p = open(filename, 'r')
    self.addable = self.process(file_p, my_file)
    file_p.close()

    if self.addable:
      self.generate_features()

  def process(self, file_p, my_file):
    lines = file_p.readlines()
    if (len(lines) == 0):
      return False

    self.intervals = list()
    self.features = dict()

    annotations = os.path.join(anno_dir, my_file)
    file_p = open(annotations, 'r')
    self.annotate(file_p)
    file_p.close()

    lines = lines[2:]
    for line in lines:
      line = line.split()
      time = line[0][1:].split(":")
      date = line[1][:-1].split("/")
      timestamp = self.transform_timestamp(time, date)
      bpm = line[2]
      if (timestamp < self.onset and bpm != "-" and bpm != "0.000"):
        interval = SECONDS_PER_MIN / float(bpm)
        curr_time = timedelta(hours=float(time[0]),
                              minutes=float(time[1]),
                              seconds=float(time[2]))
        self.intervals.append((curr_time, interval))

    if len(self.intervals) < SECONDS_PER_MIN:
      return False

    return True

  def annotate(self, file_p):
    line = file_p.readlines()[1:][0].split()
    first_time = line[0][1:].split(":")
    first_date = line[1][:-1].split("/")
    self.onset = self.transform_timestamp(first_time, first_date)

  def transform_timestamp(self, time, date):
    return datetime(day=int(date[0]),
                    month=int(date[1]),
                    year=int(date[2]),
                    hour=int(time[0]),
                    minute=int(time[1]),
                    second=int(time[2][:2]))

def mimic_parse():
  for repo in repositories:
    for subdir, dirs, files in os.walk(rootdir + repo):
      for my_file in files:
        filename = os.path.join(subdir, my_file)
        mim_record = MimicRecord(my_file, filename)
        if mim_record.addable:
          mim_records.append(mim_record)
  pickle.dump(mim_records, open("mim_records.pickle", "wb"))

def load():
  my_records = pickle.load(open("records.pickle", "rb"))
  for record in my_records:
    mit_records.append(record)

  my_records_2 = pickle.load(open("mim_records.pickle", "rb"))
  for record in my_records_2:
    mim_records.append(record)

def report():
  svm_train_features = list()
  svm_train_classes = list()
  svm_test_features = list()
  svm_test_classes = list()

  for record in mit_records:
    svm_train_features.append(list(record.features.values()))
    svm_train_classes.append(record.my_class)
  for record in mim_records:
    svm_test_features.append(list(record.features.values()))
    svm_test_classes.append(record.my_class)
  classifier = svm.SVC(kernel="linear", C=0.1)
  classifier.fit(svm_train_features, svm_train_classes)
  print("linear kernel svm accuracy: " +
        str(classifier.score(svm_test_features, svm_test_classes)))

# mimic_parse()
load()
report()
