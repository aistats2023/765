
from PyLearningExplainer.core.structure.binaryMapping import BinaryMapping
from PyLearningExplainer.core.structure.decisionTree import DecisionTree

class TreeEnsembles(BinaryMapping):
  """
  Represent a set of trees. This class is used for the class RandomForest (RF) and BoostedTrees (BTs) 
  """
  def __init__(self, forest):
    self.forest = forest
    self.n_trees = len(forest)
    self.n_features = forest[0].n_features
    self.classes = set(tree.target_class for tree in self.forest) 

    
    assert all(isinstance(tree, DecisionTree) for tree in forest), "All trees in the forest have to be of the type DecisionTree."
    assert all(tree.n_features == self.n_features for tree in forest), "All trees in the forest have to have the same number of input (features)."
    
    self.map_id_binaries_to_features, self.map_features_to_id_binaries = self.compute_id_binaries()   
    super().__init__(self.map_id_binaries_to_features, self.map_features_to_id_binaries)

    #Change the encoding of each tree by these new encoding
    for tree in self.forest:
      tree.map_id_binaries_to_features = self.map_id_binaries_to_features
      tree.map_features_to_id_binaries = self.map_features_to_id_binaries
    
    
  def compute_id_binaries(self):
    """
    Overload method from the mother class BinaryMapping
    map_id_binaries_to_features: list[id_binary] -> (id_feature, threshold) 
    map_features_to_id_binaries: dict[(id_feature, threshold)] -> [id_binary, n_appears]
    """
    map_id_binaries_to_features = [0]
    map_features_to_id_binaries = {}
    id_binary = 1

    #Fusion of map_id_binaries_to_features
    for tree in self.forest: 
      map_features_to_id_binaries.update(tree.map_features_to_id_binaries)
    
    #Now we define the good value [id_binary, n_appears] for each key 
    for key in map_features_to_id_binaries.keys():
      map_features_to_id_binaries[key][0] = id_binary
      values = [tree.map_features_to_id_binaries.get(key) for tree in self.forest]
      n_appears = sum(value[1] for value in values if value is not None)
      map_features_to_id_binaries[key][1] = n_appears
      map_id_binaries_to_features.append(key)
      id_binary+=1
    return (map_id_binaries_to_features, map_features_to_id_binaries)