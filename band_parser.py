import os
import re
import string
import nltk
import math
import pickle

rootdir = '/Users/luke/Documents/thesis_sca/zagreus/samples/BAND_HR'
outputdir = '/Users/luke/Documents/thesis_sca/zagreus/samples/BAND_NORM'
records = list()

# converts band hr data files to usable hr data files

def process(file_p):
  heart_rates = list()
  temperatures = list()

  lines = file_p.readlines()
  for line in lines:
    words = line.split()
    if words[0][0] != '+':
      heart_rates.append("%.3f\n" % (60.0 / int(words[0])))

  return heart_rates

def write_file(heart_rates, index):
  file_p = open(os.path.join(outputdir, "BAND_" + str(index) + ".txt"), 'w')
  file_p.writelines(heart_rates)
  file_p.close()

def parse():
  index = 0
  for subdir, dirs, files in os.walk(rootdir):
    for my_file in files:
      if my_file != 'dataset_description.txt':
        filename = os.path.join(subdir, my_file)
        file_p = open(filename, 'r')
        heart_rates = process(file_p)
        print(heart_rates)
        file_p.close()

        write_file(heart_rates, index)
        index += 1

parse()
