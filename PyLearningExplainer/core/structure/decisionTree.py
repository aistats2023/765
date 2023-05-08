

from PyLearningExplainer.core.tools.encoding import CNFencoding
from PyLearningExplainer.core.structure.decisionNode import DecisionNode, LeafNode
from PyLearningExplainer.core.structure.type import TypeLeaf, MethodCNF, TypeTree
from PyLearningExplainer.core.structure.binaryMapping import BinaryMapping

class DecisionTree(BinaryMapping):

  def __init__(self, type_tree, n_features, root, target_class=0, id_solver_results=0):
    self.type_tree = type_tree
    self.id_solver_results = id_solver_results
    self.n_features = n_features
    self.nodes = []
    self.root = root
    self.target_class = target_class
    if not self.root.is_leaf(): 
      self.define_parents(self.root)
    self.map_id_binaries_to_features, self.map_features_to_id_binaries = self.compute_id_binaries()
    super().__init__(self.map_id_binaries_to_features, self.map_features_to_id_binaries)

    assert isinstance(self.type_tree, TypeTree), "Please put the good type of the tree !"

  def raw_tree(self):
    #print("before: ", self.is_leaf())
    #print("ee:", self.to_tuples(self.root))
    raw_t = tuple([self.root.value]) if self.root.is_leaf() else self.to_tuples(self.root)
    return (self.target_class, raw_t)
  
  def to_tuples(self, node):
    """
    For example, this method can return (1, (2, (2.5,3.5)), (3 (-1.5, 0.5)))
    for a tree with 3 nodes and the leaves with the weights 2.5 3.5 -1.5 0.5.
    """
    output = [self.get_id_variable(node)]
    if not node.left.is_leaf():
      output.append(self.to_tuples(node.left))
    else:
      output.append(node.left.value)
    if not node.right.is_leaf():
      output.append(self.to_tuples(node.right))
    else:
      output.append(node.right.value)
    return tuple(output)

  def get_variables(self, implicant=None, node=None):
    if node is None: 
      if self.root.is_leaf(): return []
      node = self.root
    output = []
    if implicant is None:
      output.append(self.get_id_variable(node))
    else:
      if self.get_id_variable(node) in implicant:
        output.append(self.get_id_variable(node))
      if -self.get_id_variable(node) in implicant:
        output.append(-self.get_id_variable(node))
    
    if not node.left.is_leaf() and not node.right.is_leaf():
      return output + self.get_variables(implicant, node.left) + self.get_variables(implicant, node.right)  
    elif not node.left.is_leaf():
      return output + self.get_variables(implicant, node.left)
    elif not node.right.is_leaf():
      return output + self.get_variables(implicant, node.right)
    return output
  
  def direct_reason(self, instance, node=None):
    if node is None: node = self.root
    output = []
    value = instance[node.id_feature - 1]
    if value <= node.threshold:     
      output.append(-self.get_id_variable(node))
      return output + self.direct_reason(instance, node.left) if not node.left.is_leaf() else output
    else:
      output.append(self.get_id_variable(node))
      return output + self.direct_reason(instance, node.right) if not node.right.is_leaf() else output

  def define_parents(self, node, *, parent=None):
    self.nodes.append(node)
    if not node.left.is_leaf():
      self.define_parents(node.left, parent=node)
    if not node.right.is_leaf():
      self.define_parents(node.right, parent=node)
    if parent is not None:
      node.parent = parent

  def compute_id_binaries(self):
    """
    Overload method from the mother class BinaryMapping
    map_id_binaries_to_features: list[id_binary] -> (id_feature, threshold) 
    map_features_to_id_binaries: dict[(id_feature, threshold)] -> [id_binary, n_appears]
    """
    map_id_binaries_to_features = [0]
    map_features_to_id_binaries = {}
    id_binary = 1
    for node in self.nodes:
      if (node.id_feature, node.threshold) not in map_features_to_id_binaries:
        map_features_to_id_binaries[(node.id_feature, node.threshold)] = [id_binary, 1]
        map_id_binaries_to_features.append((node.id_feature, node.threshold))
        id_binary += 1
      else:
        map_features_to_id_binaries[(node.id_feature, node.threshold)][1] += 1 
    return (map_id_binaries_to_features, map_features_to_id_binaries)

  
    
  def get_id_variable(self, node):
    return self.map_features_to_id_binaries[(node.id_feature, node.threshold)][0]

  def is_leaf(self):
    return self.root.is_leaf()

  def compute_nodes_with_leaves(self, node):
    '''
    Return a list of tuple representing the children of start_node 
    '''

    output = []
    if node.left.is_leaf() or node.right.is_leaf():
      output.append(node)
    if not node.left.is_leaf():
      output += self.compute_nodes_with_leaves(node.left)
    if not node.right.is_leaf():
      output += self.compute_nodes_with_leaves(node.right)
    return output

  def display(self, node):
      if node.is_leaf():
        print(node)
      else:
        print(node)
        self.display(node.left)
        self.display(node.right)
        

  def take_decisions(self, observation):
    """
    Return the prediction (the classification) of an observation according to this node
    """
    #print("cls:", self.target_class)
    #print("value:", self.root.take_decisions(observation))
    #if self.root.is_leaf():
    #  print("coucou")
    #  return 0
    return self.root.take_decisions(observation)

  def to_CNF(self, observation, method=MethodCNF.COMPLEMENTARY, *, target_prediction=None, format=True, inverse_coding=False):
    """
    Two method: 
    - TSEITIN: Create a DNF, i.e. a disjunction of cubes.
    Each cube is a model when the observation take the good prediction. 
    Then the Tseitin transformation is applied in order to obtain a CNF.  
    - COMPLEMENTARY: Create a DNF that is true when the observation take the wrong prediction. 
    And we take the complementary of this DNF to obtain a CNF that is true when the observation takes the good prediction.   
    """
    code_prediction = True if method == MethodCNF.TSEITIN else False
    if inverse_coding: code_prediction = not code_prediction

    #Start to create the DNF according to the method TSEITIN or COMPLEMENTARY
    if target_prediction is None: target_prediction = self.take_decisions(observation)  
    dnf = []
    for node in self.compute_nodes_with_leaves(self.root):
      if node.left.is_leaf() and \
      ((code_prediction and node.left.is_prediction(target_prediction))\
      or (not code_prediction and not node.left.is_prediction(target_prediction))):
        dnf.append(self.create_cube(node, TypeLeaf.LEFT))

      if node.right.is_leaf() and \
      ((code_prediction and node.right.is_prediction(target_prediction))\
      or (not code_prediction and not node.right.is_prediction(target_prediction))):
        dnf.append(self.create_cube(node, TypeLeaf.RIGHT))
    
    if method == MethodCNF.COMPLEMENTARY:
      return CNFencoding.format(CNFencoding.complementary(dnf)) if format else CNFencoding.complementary(dnf)
    return CNFencoding.format(CNFencoding.tseitin(dnf)) if format else CNFencoding.tseitin(dnf)


  def create_cube(self, node, type_leaf):
    sign = -1 if type_leaf == TypeLeaf.LEFT else 1
    cube = [sign * self.get_id_variable(node)]
    parent = node.parent
    previous = node
    while parent is not None:
      sign = -1 if isinstance(parent.left, DecisionNode) and parent.left == previous else 1
      # if sign == 1:
      # assert isinstance(parent.right, DecisionNode), "Error: right child have to be a decision node. (id: " + str(parent.id_feature) +") (type: "+ str(type(parent.right)) +")." 
      # assert parent.right == previous, "Error: the parent of the right node is not good (id: " + str(parent.right.id_feature) +")."
      cube.append(sign * self.get_id_variable(parent))
      previous = parent
      parent = parent.parent                  
    return cube


  