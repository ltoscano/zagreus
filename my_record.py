import os
import re
import string
import nltk
import math
from random import shuffle
from sklearn import svm
import numpy
from datetime import timedelta
from constants import onsets, spontaneous_vt_onsets, spontaneous_vf_onsets

class MyRecord:

  def __init__(self, my_class, my_file):
    self.intervals = list()
    self.features = dict()
    self.my_class = my_class
    self.id = my_file.split("_")[1].split(".")[0]
    self.onset = None
    if (self.my_class == "SDDB_RR"):
      self.onset = onsets[self.id]
      self.my_class = "SCD"
    if (self.my_class == "VF_RR"):
      self.onset = spontaneous_vf_onsets[self.id]
      self.my_class = "SCD"
    if (self.my_class == "VT_RR"):
      self.onset = spontaneous_vt_onsets[self.id]
      self.my_class = "SCD"

  def process(self, file_p):
    for line in file_p.readlines():
      words = line.split()
      curr_time = words[0].split(":")
      if (len(curr_time) == 2):
        curr_time = timedelta(seconds=float(curr_time[0]),
                              microseconds=float(curr_time[1]))
      else:
        curr_time = timedelta(minutes=float(curr_time[0]),
                              seconds=float(curr_time[1]),
                              microseconds=float(curr_time[2]))
      if (self.onset):
        if (self.onset < curr_time):
          self.intervals.append((curr_time, float(words[2])))
      else:
        self.intervals.append((curr_time, float(words[2])))
    self.generate_features()

  def generate_median(self):
    sorted_list = sorted(self.intervals)
    length = len(sorted_list)
    median = 0
    if (length % 2 == 0):
      index = int(length / 2)
      median = (sorted_list[index][1] + sorted_list[index + 1][1]) / 2
    else:
      index = int((length + 1) / 2)
      median = sorted_list[index][1]
    self.features['median'] = median

  def generate_features(self):
    my_sum = 0
    my_index = 0
    my_max = 0
    my_min = 1000
    std_dev = 0

    for (time, item) in self.intervals:
      my_sum += item
      my_index += 1
      if (item > my_max):
        my_max = item
      if (item < my_min):
        my_min = item

    mean = float(my_sum / my_index)

    for (time, item) in self.intervals:
      std_dev += pow(item - mean, 2)
    std_dev = math.sqrt(std_dev / my_index)

    self.features['mean'] = mean
    self.features['min'] = my_min
    self.features['max'] = my_max
    self.features['std_dev'] = std_dev
    self.generate_median()
