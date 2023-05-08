import numpy

class LeafNode:
  def __init__(self, value):
    self.value = value 
  
  def take_decisions(self, observation=None):
    return self.value

  def is_prediction(self, prediction):
    return self.value == prediction

  def is_leaf(self):
    return True

  def __str__(self):
    return ("leaf: {}".format(self.value))
    

class DecisionNode:
  """
  A decision node represent a decision. A decision tree consists of these nodes. 
  """ 
  def __init__(self, id_feature, threshold=0.5, probabilities=[0.5,0.5], *, parent=None, left, right):
    """
    Allow to construct a decision node that do not a leaf
    """
    self.id_feature = id_feature
    self.threshold = threshold
    self.probabilities = probabilities
    self.parent = parent
    self.left = left if isinstance(left, DecisionNode) else LeafNode(left)
    self.right = right if isinstance(right, DecisionNode) else LeafNode(right)
    self.artificial_leaf = False

  def is_leaf(self):
    return self.artificial_leaf

  def __str__(self):
    return ("f{}<{}".format(self.id_feature, self.threshold))
        
  def take_decisions(self, observation):
    """
    Return the prediction (the classification) of an observation according to this node.
    This return value is either 0 or 1: 0 for the first (boolean) prediction value, 1 for the second one. 
    Warning: right nodes are considered as the 'yes' responses of conditions, left nodes as 'no'.  
    """
    #print("self.id_feature:", self.id_feature)
    if observation[self.id_feature - 1] < self.threshold:
      return self.left.take_decisions(observation)
    else:
      return self.right.take_decisions(observation)
    
      
    
          
  