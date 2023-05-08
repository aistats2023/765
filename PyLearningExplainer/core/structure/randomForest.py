

from PyLearningExplainer.core.structure.treeEnsembles import TreeEnsembles
from PyLearningExplainer.core.tools.encoding import CNFencoding
from PyLearningExplainer.core.structure.decisionTree import DecisionTree
from PyLearningExplainer.core.structure.type import MethodCNF, TypeTree

from math import floor
from pysat.card import CardEnc, EncType
import numpy

class RandomForest(TreeEnsembles):

  def __init__(self, forest):
    super().__init__(forest)
    assert all(tree.type_tree is TypeTree.PREDICTION for tree in self.forest), "All trees in a random forest have to be of the type PREDICTION."
  

  def take_decisions(self, observation):
    """
    Return the prediction (the classification) of an observation according to this node
    """
    n_votes = numpy.zeros(len(self.classes))
    for tree in self.forest:
      n_votes[tree.take_decisions(observation)] += 1
    return numpy.argmax(n_votes)

  def to_CNF(self, observation, method=MethodCNF.COMPLEMENTARY, target_prediction=None):
    if target_prediction is None: target_prediction = self.take_decisions(observation)  
    print("target_prediction:", target_prediction)

    cnf = []
    # We add firsly the cardinality constraint dealing with the votes of the trees in the forest.
    new_variables_atleast = [v for v in range(1,self.n_trees+1)] # Each tree is represented by a new variable
    condition_atleast = floor(self.n_trees/2)+1
    atleast_clauses = CardEnc.atleast(lits=new_variables_atleast, encoding=EncType.seqcounter, bound=condition_atleast).clauses
    atleast_clauses = [[l + ((1 if l > 0 else -1)*self.n_features) for l in clause] for clause in atleast_clauses]
    new_variables_atleast = [l for l in range(1+self.n_features,1+self.n_features+self.n_trees)]
    cnf.extend(atleast_clauses)
    print("atleast_clauses:", atleast_clauses)

    # We secondly encode the trees
    for new_variable in new_variables_atleast:
      current_tree = self.forest[new_variable - (self.n_features+1)]
      clauses_for_l = current_tree.to_CNF(observation, format=False)
      clauses_for_not_l = current_tree.to_CNF(observation, format=False, inverse_coding=True)
      for clause in clauses_for_l: clause.append(-new_variable)
      for clause in clauses_for_not_l: clause.append(new_variable)
      cnf.extend(clauses_for_l + clauses_for_not_l)

    return CNFencoding.format(cnf)
