

from PyLearningExplainer.core.structure.treeEnsembles import TreeEnsembles
from PyLearningExplainer.core.tools.encoding import CNFencoding
from PyLearningExplainer.core.structure.decisionTree import DecisionTree
from PyLearningExplainer.core.structure.type import MethodCNF, TypeTree

import numpy
import math
import os

class BoostedTrees(TreeEnsembles):

  def __init__(self, forest, n_classes=2):
    super().__init__(forest)
    self.n_classes = n_classes
    #print("dict:", self.map_features_to_id_binaries)
    assert all(tree.type_tree is TypeTree.WEIGHT for tree in self.forest), "All trees in a boosted trees have to be of the type WEIGHT."
  
  def reduce_nodes(self, node, tree, implicant, get_min):
    if node.is_leaf():
      return
    self.reduce_nodes(node.left, tree, implicant, get_min)
    self.reduce_nodes(node.right, tree, implicant, get_min)
    if node.left.is_leaf() and node.right.is_leaf():
      id_variable = tree.get_id_variable(node)
      instance_w = node.right.value if id_variable in implicant else node.left.value
      not_instance_w = node.left.value if id_variable in implicant else node.right.value
      if (get_min and instance_w < not_instance_w) or (not get_min and instance_w > not_instance_w):
        node.artificial_leaf = True
        node.value = instance_w


  def reduce_trees(self, implicant, prediction):
    for tree in self.forest:
      for node in tree.nodes:
        node.artificial_leaf = False
      self.reduce_nodes(tree.root, tree, implicant, prediction == 1 if self.n_classes == 2 else tree.target_class == prediction)

  def remove_reduce_trees(self):
    for tree in self.forest:
      for node in tree.nodes:
        node.artificial_leaf = False


  def __str__(self):
    s = "nTrees: " + str(self.n_trees) + os.linesep
    #s += "nFeatures in the biggest tree: " + str(max(tree.existing_variables() for tree in self.forest)) + os.linesep 
    s += "nVariables:" + str(len(self.map_id_binaries_to_features)-1) + os.linesep
    return s
  
  def compute_probabilities(self, instance):
    scores = numpy.asarray([tree.take_decisions(instance) for tree in self.forest])  
    if self.n_classes > 2:
      class_scores = numpy.asarray([(math.exp((scores[i::self.n_classes]).sum())) for i in range(self.n_classes)])
      return class_scores/class_scores.sum()
    else:
      return [0, 1] if sum(scores) > 0 else [1, 0]

  def predict(self, instance):
    """
    Return the prediction (the classification) of an instance according to the trees
    """
    return numpy.argmax(self.compute_probabilities(instance))

  def to_CNF(self, observation, method=MethodCNF.COMPLEMENTARY, target_prediction=None):
    pass
