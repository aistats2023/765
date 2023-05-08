from PyLearningExplainer.core.tools.encoding import CNFencoding

class BinaryMapping():

  def __init__(self, map_id_binaries_to_features, map_features_to_id_binaries):
    self.map_id_binaries_to_features = map_id_binaries_to_features
    self.map_features_to_id_binaries = map_features_to_id_binaries

  def compute_id_binaries(self):
    assert False, "Have to be implemented in a child class." 
  
  def observation_to_binaries(self, observation, preference_order=None):
    """
    Transforme an observable into a cube (conjunction of literals) according to the tree
    """
    output = []
    if preference_order is None:
      for key in self.map_features_to_id_binaries.keys():
        sign = 1 if observation[key[0]-1] >= key[1] else -1
        output.append(sign * self.map_features_to_id_binaries[key][0])
      return CNFencoding.format(output)
    assert True, "To implement"
    return output
  

  def get_id_features(self, implicant):
    return tuple(self.map_id_binaries_to_features[abs(l)][0] for l in implicant)

  def to_features(self, implicant):
    """
    Convert each literal to a triplet ('id_feature', 'threshold', 'sign'). 
    The last element 'sign' say if the feature was POSITIVE (TRUE) or NEGATIVE (FALSE) in the implicant. 
    Remark: call self.eliminate_redundant_features()
    """
    result = []
    implicant = self.eliminate_redundant_features(implicant)
    for l in implicant:
      id_feature = self.map_id_binaries_to_features[abs(l)][0]
      threshold = self.map_id_binaries_to_features[abs(l)][1]
      sign = True if l > 0 else False
      weight = implicant[l] if isinstance(implicant, dict) else None
      result.append((id_feature, threshold, sign, weight))
    return tuple(result)

  

  def eliminate_redundant_features(self, implicant):
    """
    A implicant without redundant features i.e. If we have 'feature_a > 3' and 'feature_a > 2', we keep only the id_binary linked to the boolean corresponding to the 'feature_a > 3'
    Warning, the 'implicant' parameter can be a list of literals or a dict of literals. 
    In the last case, it is a map literal -> weight.
    """
    positive = {}
    positive_weights = {}
    negative = {}
    negative_weights = {}
    n_redundant_features = 0
    
    #Search redundant features
    for l in implicant:
      key = self.map_id_binaries_to_features[abs(l)]
      id_feature = key[0]
      threshold = key[1]
      if l > 0:
        if id_feature not in positive:
          positive[id_feature] = [threshold]
          if isinstance(implicant, dict):
            positive_weights[id_feature] = implicant[l]
        else:
          n_redundant_features+=1
          positive[id_feature].append(threshold)
          if isinstance(implicant, dict):
            positive_weights[id_feature] += implicant[l]
      else:
        if id_feature not in negative:
          negative[id_feature] = [threshold]
          if isinstance(implicant, dict):
            negative_weights[id_feature] = implicant[l]
        else:
          n_redundant_features+=1
          negative[id_feature].append(threshold)
          if isinstance(implicant, dict):
            negative_weights[id_feature] += implicant[l]

    #Copy the new implicant without these redundant features
    if not isinstance(implicant, dict):
      output = [self.map_features_to_id_binaries[(k, max(positive[k]))][0] for k in positive.keys()]
      output += [-self.map_features_to_id_binaries[(k, min(negative[k]))][0] for k in negative.keys()]
      return tuple(output)
    
    output = [(self.map_features_to_id_binaries[(k, max(positive[k]))][0], positive_weights[k]) for k in positive.keys()]
    output += [(-self.map_features_to_id_binaries[(k, min(negative[k]))][0], negative_weights[k]) for k in negative.keys()]
    output = {t[0]:t[1] for t in output} #To dict
    return output
    