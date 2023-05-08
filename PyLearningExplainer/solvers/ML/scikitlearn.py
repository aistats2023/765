
from PyLearningExplainer.solvers.ML.MLSolver import MLSolver, MLSolverResults
from PyLearningExplainer.core.tools.utils import flatten, shuffle, compute_accuracy
from PyLearningExplainer.core.structure.decisionTree import DecisionTree, DecisionNode, LeafNode
from PyLearningExplainer.core.structure.type import TypeTree, TypeReason

import pandas
import copy

import numpy
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.tree import DecisionTreeClassifier


class Scikitlearn(MLSolver):
    """
    Load the dataset, rename the attributes and separe the prediction from the data
    instance = observation 
    labels = prediction

    """
    def __init__(self, datasetname):
        super().__init__(datasetname)
        

    def fit_and_predict(self, instances_training, instances_test, labels_training, labels_test):
      # Training phase
      decision_tree = DecisionTreeClassifier(max_depth=None)
      decision_tree.fit(instances_training, labels_training)
      
      # Test phase
      result = decision_tree.predict(instances_test)
      return (copy.deepcopy(decision_tree), compute_accuracy(result, labels_test))

    def simple_validation(self):
      self.results.clear()

      #spliting
      indices = numpy.arange(len(self.data))
      instances_training, instances_test, labels_training, labels_test, training_index, test_index = train_test_split(self.data, self.labels, indices, test_size = 0.3, random_state = 0)
      
      #solving
      tree, accuracy = self.fit_and_predict(instances_training, instances_test, labels_training, labels_test)

      self.results.append(MLSolverResults(tree,training_index,test_index,None,accuracy))
      return self

    def cross_validation(self, *, n_trees=4):
      assert n_trees > 1, "cross_validation() expects at least 2 trees. For just one tree, please use simple_validation()"
      self.results.clear()

      #spliting
      quotient, remainder = (self.n_instances//n_trees, self.n_instances%n_trees) 
      groups = shuffle(flatten([quotient*[i] for i in range(1, n_trees + 1)]) + [i for i in range(1, remainder + 1)])
      cross_validator = LeaveOneGroupOut()
      
      for training_index, test_index in cross_validator.split(self.data, self.labels, groups):
        # Select good observations for each of the 'n_trees' experiments.
        instances_training = [self.data[i] for i in training_index]
        labels_training = [self.labels[i] for i in training_index]
        instances_test = [self.data[i] for i in test_index]
        labels_test = [self.labels[i] for i in test_index]
        
        #solving
        tree, accuracy = self.fit_and_predict(instances_training, instances_test, labels_training, labels_test)

        # Save some information
        self.results.append(MLSolverResults(tree,training_index,test_index,groups,accuracy))
      return self

    """
    Return an observation -a instance) from results that is either correct or incorrect.  
    """
    def get_instances(self, tree, n_instances=TypeReason.All, correct=None):
      sk_tree = self.results[tree.id_solver_results].tree
      test_index = self.results[tree.id_solver_results].test_index
      
      n_instances = n_instances if type(n_instances) == int else len(test_index)
      
      instances_test = numpy.array([self.data[x] for x in test_index])
      labels_test = numpy.array([self.labels[x] for x in test_index])

      instances = []
      for j in range(n_instances):
        if correct is True and (sk_tree.predict(instances_test[j].reshape(1, -1)) == labels_test[j])[0]:
          instances.append(instances_test[j])
        if correct is False and (sk_tree.predict(instances_test[j].reshape(1, -1)) != labels_test[j])[0]:
          instances.append(instances_test[j])
        if correct is None:
          instances.append(instances_test[j])
      return instances

    """
    Convert the Scikitlearn's decision trees into the program-specific objects called 'DecisionTree'.
    """
    def to_decision_trees(self):  
      return [self.results_to_trees(id_solver_results) for id_solver_results,_ in enumerate(self.results)]
      
    """
    Convert a specific Scikitlearn's decision tree into a program-specific object called 'DecisionTree'.
    """
    def results_to_trees(self, id_solver_results=0):
      sk_tree = self.results[id_solver_results].tree
      sk_raw_tree = sk_tree.tree_

      nodes = {i:DecisionNode(int(feature + 1), sk_raw_tree.threshold[i], sk_raw_tree.value[i][0], left=None, right=None) 
               for i, feature in enumerate(sk_raw_tree.feature) if feature >= 0}
        
      for i in range(len(sk_raw_tree.feature)):
        if i in nodes:
          # Set left and right of each node
          id_left = sk_raw_tree.children_left[i]
          id_right = sk_raw_tree.children_right[i]          
          nodes[i].left = nodes[id_left] if id_left in nodes else LeafNode(numpy.argmax(sk_raw_tree.value[id_left][0]))
          nodes[i].right = nodes[id_right] if id_right in nodes else LeafNode(numpy.argmax(sk_raw_tree.value[id_right][0]))
      root = nodes[0] if 0 in nodes else DecisionNode(1, 0, sk_raw_tree.value[0][0])
      return DecisionTree(TypeTree.PREDICTION, sk_tree.n_features_, root, sk_tree.classes_, id_solver_results=id_solver_results)

 