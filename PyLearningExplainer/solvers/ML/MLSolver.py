

import pandas
from collections import OrderedDict

from PyLearningExplainer.core.structure.type import TypeReason

class MLSolverResults():
    def __init__(self, tree, training_index, test_index, group, accuracy):
      self.tree = tree
      self.training_index = training_index 
      self.test_index = test_index
      self.group = group
      self.accuracy = accuracy

class MLSolver():
    

    """
    Load the dataset, rename the attributes and separe the prediction from the data
    instance = observation 
    labels != prediction
    attributes = features
    """
    def __init__(self, datasetname):
      if datasetname == None:
        self.data = None
        self.labels = None
        self.results = []
        return None
      self.load_data(datasetname)     

    def count_lines(self, filename):
      with open(filename) as f:
        return sum(1 for _ in f)

    def load_data_limited(self, datasetname, possibles_indexes, n):
      self.datasetname = datasetname
      n_indexes = self.count_lines(datasetname) - 1 #to skip the first line
      skip = [i+1 for i in range(n_indexes) if i not in possibles_indexes]
      
      # create a map to get the good order of instances
      sorted_possibles_indexes = sorted(possibles_indexes)
      map_possibles_indexes = [sorted_possibles_indexes.index(index) for index in possibles_indexes]

      data = pandas.read_csv(
        datasetname,
        skiprows=skip if possibles_indexes is not None else None,
        nrows=n
        )
      
       
      #numpy_data = data.to_numpy()
      #sorted_data = [numpy_data[map_possibles_indexes[i]].tolist() for i in range(len(numpy_data))]
      
      # recreate the dataframe object but with the good order of instances
      sorted_data = pandas.DataFrame(columns = data.columns).astype(data.dtypes)
      for i in range(data.shape[0]):
        sorted_data = sorted_data.append(data.loc[map_possibles_indexes[i]].to_dict(),ignore_index=True)
      sorted_data = sorted_data.astype(data.dtypes)
      
      n_instances, n_features = data.shape
      self.rename_attributes(data)
      data, labels = self.remove_labels(data, n_features)
      labels = self.labelsToValues(labels)
      data = data.to_numpy()
      
      return data, labels

    def load_data(self, datasetname):
      self.datasetname = datasetname 
      self.data = pandas.read_csv(datasetname).copy()
      print(self.data)
      self.n_instances, self.n_features = self.data.shape
      self.feature_name = self.data.columns.values.tolist() 
      
      self.rename_attributes(self.data)
      self.data, self.labels = self.remove_labels(self.data, self.n_features)
      self.create_dict_labels(self.labels)
      self.labels = self.labelsToValues(self.labels)
      
      self.n_labels = len(set(self.labels))
      self.data = self.data.to_numpy() #remove the first line (attributes) and now the first dimension represents the instances :)!
      
      self.results = []
      print("---------------   Information   ---------------")
      print("Dataset name:", datasetname)
      print("nFeatures (nAttributes, with the labels):", self.n_features)
      print("nInstances (nObservations):", self.n_instances)
      print("nLabels:", self.n_labels)

    """
    Rename attributes in self.data in string of integers from 0 to 'self.n_attributes'
    """
    def rename_attributes(self, data):
      rename_dictionary = {element: str(i) for i, element in enumerate(data.columns)}
      data.rename(columns=rename_dictionary, inplace=True)

    def create_dict_labels(self, labels):
      index = 0
      self.dict_labels = OrderedDict()
      for p in labels:
        if str(p) not in self.dict_labels:
          self.dict_labels[str(p)] = index
          index+=1

    """
    Convert labels (predictions) into binary values
    Using of OrderedDict in order to be reproducible.
    """
    def labelsToValues(self, labels):
      return [self.dict_labels[str(element)] for element in labels]

    """
    Remove and get the prediction: it is the last attribute (column) in the file
    """
    def remove_labels(self, data, n_features):
      prediction = data[str(n_features-1)].copy().to_numpy()
      data = data.drop(columns=[str(n_features-1)])
      return data, prediction
      
    def cross_validation(self, *, n_tree=4):
      assert False, "Error: this method have to be overload"

    """
    Return an observation from results that is either correct or incorrect.  
    """
    def get_instances(self, tree, n_instances=TypeReason.All, correct=None):
      assert False, "Erreur: this method have to be overload"

    """
    Convert the Scikitlearn's decision trees into the program-specific objects called 'DecisionTree'.
    """
    def to_decision_trees(self):  
      assert False, "Erreur: this method have to be overload"
      
    """
    Convert a specific Scikitlearn's decision tree into a program-specific object called 'DecisionTree'.
    """
    def to_decision_tree(self, id_solver_tree):
      assert False, "Erreur: this method have to be overload"

 