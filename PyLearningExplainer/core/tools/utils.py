from functools import reduce
from operator import iconcat
from numpy import sum
from termcolor import colored
import random

from time import time

class Stopwatch:
  def __init__(self):
    self.initial_time = time()

  def elapsed_time(self, *, reset=False):
    elapsed_time = time() - self.initial_time
    if reset:
      self.initial_time = time()
    return "{:.2f}".format(elapsed_time)
    
def flatten(l):
  return reduce(iconcat, l, [])

def shuffle(l):
  random.shuffle(l)
  return l

def add_lists_by_index(list1, list2):
  """
  Adding two lists results in a new list where each element is the sum of the elements in the corresponding positions of the two lists.
  """
  return [x + y for (x, y) in zip(list1, list2)]

def compute_accuracy(prediction, right_prediction):
  return (sum(prediction == right_prediction) / len(right_prediction)) * 100

def display_observation(observation, size=28):
  for i, element in enumerate(observation):
    print(colored('X', 'blue') if element == 0 else colored('X', 'red'), end='')
    if (i+1) % size == 0 and i != 0:
      print()
