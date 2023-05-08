
from PyLearningExplainer.core.tools.encoding import CNFencoding

import c_explainer

from PyLearningExplainer.core.structure.type import TypeReason, ReasonExpressivity

from itertools import chain, combinations
import random
import numpy
#from pycsp3 import SAT, UNSAT, UNKNOWN, OPTIMUM

dd = dict()

class ExplainerBT():

  def __init__(self, boosted_trees):
    self.boosted_trees = boosted_trees # The boosted trees.
    self.c_BT = None

  def display_information(self):
    print("---------   Trees Information   ---------")
    nNodesTotal = sum([len(tree.get_variables()) for tree in self.boosted_trees.forest])
    nNodesMax = max([len(tree.get_variables()) for tree in self.boosted_trees.forest])
    nNodesMaxWithoutRedundancy = max([len(set(tree.get_variables())) for tree in self.boosted_trees.forest])
    nNodesTotalWithoutRedundancy = []
    for tree in self.boosted_trees.forest:
      nNodesTotalWithoutRedundancy.extend(tree.get_variables())
    nNodesTotalWithoutRedundancy = len(list(set(nNodesTotalWithoutRedundancy)))

    print("nNodesMax:", nNodesMax)
    print("nNodesMaxWithoutRedundancy:", nNodesMaxWithoutRedundancy)  
    print("nNodesTotal:", nNodesTotal)
    print("nNodesTotalWithoutRedundancy:", nNodesTotalWithoutRedundancy)  
    print()
    
  def set_instance(self, instance, implicant=None, target_prediction=None):
    self.instance = instance # The target observation.
    self.implicant = implicant # An implicant of self.tree (a term that implies the tree)
    self.target_prediction = target_prediction # The target prediction (0 or 1)
    if self.implicant is None:
      self.implicant = self.boosted_trees.observation_to_binaries(instance)

    if self.target_prediction is None:
      self.target_prediction = self.boosted_trees.predict(instance)
    
    self.implicant_id_features = self.boosted_trees.get_id_features(self.implicant) #The id_features of each variable of the implicant


  # return the indexes of the instance that are involved in the reason 
  def reduce_instance(self, reason):
    reduced_instance = []
    features = [feature for (feature,_,_,_) in self.to_features(reason)]
    return [i for i, _ in enumerate(self.instance) if i+1 in features]

  def to_features(self, implicant):
    return self.boosted_trees.to_features(implicant)

  def compute_propabilities(self):
    return self.boosted_trees.compute_probabilities(self.instance)

  def compute_direct_reason(self):
    seen_in_trees = set()
    for tree in self.boosted_trees.forest:
      seen_in_trees |= set(tree.direct_reason(self.instance))
    return CNFencoding.format(list(seen_in_trees))

  def partial_to_complete(self, partial):
    complete = list(partial).copy()
    to_add = [literal for literal in self.implicant if literal not in complete]
    for literal in to_add:
      sign = random.choice([1, -1])
      complete.append(sign * abs(literal))
    return complete

  def is_abductive_reason(self, abductive_reason, n_samples=1000):
    ok_samples = 0
    for _ in range(n_samples):
      complete = self.partial_to_complete(abductive_reason)
      if self.boosted_trees.n_classes == 2:
        weights = [self.compute_weights(tree, tree.root, complete) for tree in self.boosted_trees.forest]
        assert(all(len(x) == 1 for x in weights))
        weights = sum([x[0] for x in weights])
        prediction = numpy.argmax([0, 1]) if weights > 0 else numpy.argmax([1, 0])
        if prediction == self.target_prediction:
          ok_samples += 1
      else:
        weights_per_class = []
        for cl in self.boosted_trees.classes:
          weights = [self.compute_weights(tree, tree.root, complete) for tree in self.boosted_trees.forest if tree.target_class == cl]
          assert(all(len(x) == 1 for x in weights))
          weights = [x[0] for x in weights]
          weights_per_class.append(sum(weights))
        if numpy.argmax(weights_per_class) == self.target_prediction:
          ok_samples += 1
    result = round((ok_samples * 100) / n_samples, 2)
    return result

  def weight_float_to_int(self, weight):
    return int(weight*pow(10,9))


  # def compute_minimal_abductive_reason_V1(self):
  #   cp_solver = MinimalV1()
  #   data_matrises, data_domains_weights, data_literals_per_matrix, data_classes = cp_solver.data_formatting(self)
  #   cp_solver.create_model_minimal_abductive_BT(self.implicant, data_matrises, data_domains_weights, data_literals_per_matrix, data_classes, self.target_prediction)
  #   solution = cp_solver.solve()
  #   return CNFencoding.format([l for i, l in enumerate(self.implicant) if solution[i] == 1])

  # def compute_minimal_abductive_reason_V2(self, *, time_limit=0, reason_expressivity, from_reason=None):
  #  cp_solver = MinimalV2()
  #  implicant_id_features = self.implicant_id_features if reason_expressivity == ReasonExpressivity.Features else []
  #  cp_solver.create_model_minimal_abductive_BT(
  #    self.implicant, 
  #    self.boosted_trees, 
  #    self.target_prediction,
  #    self.boosted_trees.n_classes,
  #    implicant_id_features,
  #    from_reason)
  #  result, solution = cp_solver.solve(time_limit=time_limit)
  #  if result == UNSAT or result == UNKNOWN:
  #    return result, []
  #  return result, CNFencoding.format([l for i, l in enumerate(self.implicant) if solution[i] == 1])

  #def compute_minimal_abductive_reason_V3(self, *, time_limit=0, reason_expressivity):
  #  cp_solver = MinimalV3()
  #  implicant_id_features = self.implicant_id_features if reason_expressivity == ReasonExpressivity.Features else []
  #  cp_solver.create_model_minimal_abductive_BT(
  #    self.implicant, 
  #    self.boosted_trees, 
  #    self.target_prediction, 
  #    self.boosted_trees.n_classes,
  #    implicant_id_features)
  #  result, solution = cp_solver.solve(time_limit=time_limit)
  #  if result == UNSAT or result == UNKNOWN:
  #    return result, []
  #  return result, CNFencoding.format([l for i, l in enumerate(self.implicant) if solution[i] == 1])

  def is_implicant(self, abductive):
    if self.boosted_trees.n_classes == 2:
      # 2-classes case
      sum_weights = []
      for tree in self.boosted_trees.forest:
        weights = self.compute_weights(tree, tree.root, abductive)
        worst_weight = min(weights) if self.target_prediction == 1 else max(weights)
        #print(worst_weight, " ", end = "")
        sum_weights.append(worst_weight)
      sum_weights = sum(sum_weights)
      prediction = numpy.argmax([0, 1]) if sum_weights > 0 else numpy.argmax([1, 0])
      #print("result = " , prediction)
      return self.target_prediction == prediction
    else:
      # multi-classes case
      worst_one = self.compute_weights_class(abductive, self.target_prediction, king="worst")
      best_ones = [self.compute_weights_class(abductive, cl, king="best") for cl
      in self.boosted_trees.classes if cl != self.target_prediction]
      return all(worst_one > best_one for best_one in best_ones)

  def compute_abductive_reason(self, *, n_iterations=50, time_limit=0, reason_expressivity):
    """
    Compute in c++ several reasons either during 'time_limit' times or a fixed 'n_iterations' number of reasons.
    The parameter 'reason_expressivity' have to be fixed either by ReasonExpressivity.Features or ReasonExpressivity.Conditions. 
    """
    if self.c_BT == None :
      # Preprocessing to give all trees in the c++ library
      self.c_BT = c_explainer.new_BT(self.boosted_trees.n_classes)
      for tree in self.boosted_trees.forest:
        c_explainer.add_tree(self.c_BT, tree.raw_tree())
    return c_explainer.compute_reason(self.c_BT, self.implicant, self.implicant_id_features, self.target_prediction, n_iterations, time_limit, int(reason_expressivity))
    

  def compute_abductive_reason_python(self, seed=0):
    """
    Compute in python only one reason.
    """
    abductive = list(self.implicant).copy()
    copy_implicant = list(self.implicant).copy()
    if seed != 0:
      random.seed(seed)
      random.shuffle(copy_implicant)

    for lit in copy_implicant:
      tmp_abductive = abductive.copy()
      tmp_abductive.remove(lit)
      if self.is_implicant(tmp_abductive):
        abductive.remove(lit)
      
    return CNFencoding.format(abductive)

  def compute_weights_class(self, implicant, cls, king="worst"):
    weights = [self.compute_weights(tree, tree.root, implicant) for tree in self.boosted_trees.forest if tree.target_class == cls]
    weights = [min(weights_per_tree) if king=="worst" else max(weights_per_tree) for weights_per_tree in weights]
    return sum(weights)

  def weight_float_to_int(self, weight):
    return weight
    #return int(weight*pow(10,9))

  def compute_weights(self, tree, node, implicant):

    if tree.root.is_leaf(): #Special case for tree without condition
      return [self.weight_float_to_int(tree.root.value)]

    id_variable = tree.get_id_variable(node)
    weights = []
    if id_variable in implicant:
      if node.right.is_leaf():
        return [self.weight_float_to_int(node.right.value)]
      else:
        weights.extend(self.compute_weights(tree, node.right, implicant))
        return weights
    elif -id_variable in implicant:
      if node.left.is_leaf():
        return [self.weight_float_to_int(node.left.value)]
      else:
        weights.extend(self.compute_weights(tree, node.left, implicant))
        return weights
    else: # the variable is not in the implicant
      if node.left.is_leaf():
        weights.append(self.weight_float_to_int(node.left.value))
      else:
        weights.extend(self.compute_weights(tree, node.left, implicant))
      if node.right.is_leaf():
        weights.append(self.weight_float_to_int(node.right.value))
      else:
        weights.extend(self.compute_weights(tree, node.right, implicant))
    return weights