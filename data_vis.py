#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import copy
import operator
import numpy
from random import shuffle
from my_record import MyRecord, interval_base
import pickle
import matplotlib

def parallel_coordinates(data_sets, style=None):
  dims = len(data_sets[0])
  x = range(dims)
  fig, axes = plt.subplots(1, dims - 1, sharey=False)

  if style is None:
    style = ['r-'] * len(data_sets)

  # Calculate the limits on the data
  min_max_range = list()
  for m in zip(*data_sets):
    mn = min(m)
    mx = max(m)
    if mn == mx:
        mn -= 0.5
        mx = mn + 1.
    r = float(mx - mn)
    min_max_range.append((mn, mx, r))

  # Normalize the data sets
  norm_data_sets = list()
  for ds in data_sets:
    nds = [(value - min_max_range[dimension][0]) /
            min_max_range[dimension][2]
            for dimension, value in enumerate(ds)]
    norm_data_sets.append(nds)
  data_sets = norm_data_sets

  # Plot the datasets on all the subplots
  for i, ax in enumerate(axes):
    for dsi, d in enumerate(data_sets):
      ax.plot(x, d, style[dsi])
    ax.set_xlim([x[i], x[i + 1]])

  # Set the x axis ticks
  for dimension, (axx, xx) in enumerate(zip(axes, x[:-1])):
    axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
    ticks = len(axx.get_yticklabels())
    labels = list()
    step = min_max_range[dimension][2] / (ticks - 1)
    mn = min_max_range[dimension][0]
    for i in range(ticks):
      v = mn + i * step
      labels.append('%4.2f' % v)
    axx.set_yticklabels(labels)

  # Move the final axis' ticks to the right-hand side
  axx = plt.twinx(axes[-1])
  dimension += 1
  axx.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
  ticks = len(axx.get_yticklabels())
  step = min_max_range[dimension][2] / (ticks - 1)
  mn = min_max_range[dimension][0]
  labels = ['%4.2f' % (mn + i * step) for i in range(ticks)]
  axx.set_yticklabels(labels)

  # Stack the subplots
  plt.subplots_adjust(wspace=0)

  return plt

def load():
  records = list()
  my_records = pickle.load(open("records.pickle", "rb"))
  my_records_band = pickle.load(open("band_records.pickle", "rb"))

  for record in my_records:
    records.append(record)
  for record in my_records_band:
    records.append(record)

  dataset = list()
  for record in records:
    dataset.append(list(record.features.values()))

  colors = list()
  for record in records:
    if record.my_class == "SCD":
      colors.append("r")
    else:
      colors.append("b")

  parallel_coordinates(dataset, style=colors).show()

load()
