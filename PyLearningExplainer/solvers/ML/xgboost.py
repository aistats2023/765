

from PyLearningExplainer.solvers.ML.MLSolver import MLSolver, MLSolverResults
from PyLearningExplainer.core.tools.utils import flatten, shuffle, compute_accuracy
from PyLearningExplainer.core.structure.decisionTree import DecisionTree, DecisionNode, LeafNode
from PyLearningExplainer.core.structure.type import EvaluationMethod, EvaluationOutput, TypeReason, TypeTree, Indexes
from PyLearningExplainer.core.structure.boostedTrees import BoostedTrees

import xgboost
import numpy
import copy
import json
import os
import shutil

from sklearn.model_selection import LeaveOneGroupOut, train_test_split

class Xgboost(MLSolver):
    
    
    """
    Load the dataset, rename the attributes and separe the prediction from the data
    """
    def __init__(self, datasetname=None):
      super().__init__(datasetname)
      
             
    def fit_and_predict(self, instances_training, instances_test, labels_training, labels_test):
      # Training phase
      xgb_classifier = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
      xgb_classifier.fit(instances_training, labels_training)
      # Test phase
      result = xgb_classifier.predict(instances_test)
      return (copy.deepcopy(xgb_classifier), compute_accuracy(result, labels_test))

    def evaluate(self, *, method, output, model_directory=None):
      print("---------------   Evaluation   ---------------")
      print("method:", str(method))
      print("output:", str(output))
      
      if method == EvaluationMethod.HoldOut:
        self.hold_out_evaluation()
      elif method == EvaluationMethod.LeaveOneGroupOut:
        self.leave_one_group_out_evaluation()
      elif method == EvaluationMethod.LoadModel:
        assert model_directory is not None, "The parameter 'model' have to indicate the model directory !"
        self.load_model_evaluation(model_directory)
      else:
        assert True, "Not implemented !"

      print("---------   Evaluation Information   ---------")
      for i, result in enumerate(self.results):
        print("For the evaluation number " + str(i) + ":")
        print("accuracy:", result.accuracy)
        print("nTraining instances:",len(result.training_index))
        print("nTest instances:", len(result.test_index))

      print("---------------   Explainer   ----------------")
      result_output = None
      if output == EvaluationOutput.BoostedTrees:
        result_output = self.to_boosted_trees()
      if output == EvaluationOutput.SaveModel:
        self.save_model(model_directory)
        result_output = self.to_boosted_trees()
      else:
        assert True, "Not implemented !"

      for i, result in enumerate(result_output):
        print("For the evaluation number " + str(i) + ":")
        print(result)

      return result_output

    def save_model(self, model_directory):
      name = self.datasetname.split(os.sep)[-1].split('.')[0]
      if model_directory is not None:
        base_directory = model_directory + os.sep + name + "_model" 
      else:
        base_directory = name + "_model"
        
      shutil.rmtree(base_directory, ignore_errors=True)
      os.mkdir(base_directory)

      for i, result in enumerate(self.results):
        namefile = base_directory + os.sep + name + '.' + str(i)

        # model:
        result.tree.save_model(namefile + ".model")
        
        # map of indexes for training and test part
        data = {"training_index": result.training_index.tolist(), 
                "test_index": result.test_index.tolist(),
                "accuracy": result.accuracy,
                "n_features": self.n_features, 
                "n_labels": self.n_labels,
                "dict_labels": self.dict_labels}

        json_string = json.dumps(data)
        with open(namefile + ".map", 'w') as outfile:
          json.dump(json_string, outfile)
        
        print("Model saved in:", base_directory)


    def load_model_evaluation(self, model_directory, with_tests=False):
      self.results.clear()
      
      #get the files
      files = []
      index = 0
      found = True
      while found:
        found = False
        for filename in os.listdir(model_directory): 
          model_file = os.path.join(model_directory, filename) 
          if os.path.isfile(model_file) and model_file.endswith(str(index)+".model"):
            map_file = model_file.replace(".model", ".map")
            assert os.path.isfile(map_file), "A '.model' file must be accompanied by a '.map' file !"
            files.append((model_file, map_file)) 
            index += 1
            found = True
            break
          
      for _, model in enumerate(files):
        model_file, map_file = model
        #load model
        xgb_classifier = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        xgb_classifier.load_model(model_file)

        #recuperate map
        f = open(map_file)
        data = json.loads(json.load(f))
        training_index = data['training_index']
        test_index = data['test_index']
        accuracy_saved = data['accuracy']
        self.n_features = data['n_features']
        self.n_labels = data["n_labels"]
        self.dict_labels = data["dict_labels"]
        f.close()

        print("----------   Loading Information   -----------")
        print("mapping file:", map_file)
        print("nFeatures (nAttributes, with the labels):", self.n_features)
        print("nInstances (nObservations):", len(training_index) + len(test_index))
        print("nLabels:", self.n_labels)
        if with_tests is True:
          # Test phase
          instances_test = [self.data[i] for i in test_index]
          labels_test = [self.labels[i] for i in test_index]
          result = xgb_classifier.predict(instances_test)
          accuracy = compute_accuracy(result, labels_test)
          assert accuracy == accuracy_saved, "The accuracy between the model loaded and the one determined at its creation is not the same !"
        self.results.append(MLSolverResults(copy.deepcopy(xgb_classifier),training_index,test_index,None,accuracy_saved))
      return self

    def hold_out_evaluation(self):
      self.results.clear()
      assert self.data is not None, "You have to put the dataset in the class parameters."
      #spliting
      indices = numpy.arange(len(self.data))
      instances_training, instances_test, labels_training, labels_test, training_index, test_index = train_test_split(self.data, self.labels, indices, test_size = 0.3, random_state = 0)
     
      #solving
      tree, accuracy = self.fit_and_predict(instances_training, instances_test, labels_training, labels_test)
      self.results.append(MLSolverResults(tree,training_index,test_index,None,accuracy))
      return self
                    
    def leave_one_group_out_evaluation(self, *, n_trees=10):
      assert self.data is not None, "You have to put the dataset in the class parameters."
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
    Return couples (instance, prediction) from data and the MLsolver results.
    
    'indexes': take only into account some indexes of instances
      - Indexes.Training: indexes from the training instances of a particular model 
      - Indexes.Test: indexes from the test instances of a particular model
      - Indexes.Mixed: take firsly indexes from the test set and next from the training set in order to have at least 'n' instances. 
      - Indexes.All: all indexes are take into account
      - string: A file contening specific indexes 
    
    'dataset': 
      - can be None if the dataset is already loaded
      - the dataset if you have not loaded it yet
      
    'model':
      - a model for the 'type=training' or 'type=test'
       
    'n': The desired number of instances (None for all).

    'correct': only available if a model is given 
      - None: all instances
      - True: only correctly classified instances by the model 
      - False: only misclassified instances by the model 

    'classes': 
      - None: all instances
      - []: List of integers representing the classes/labels for the desired instances  

    'backup_directory': save the instance indexes in a file in the directory given by this parameter 
    """
    def get_instances(self, indexes=Indexes.All, dataset=None, model=None, n=None, correct=None, predictions=None, backup_directory=None, backup_id=None):
      print("---------------   Instances   ----------------")
      print("Correctness of instances : ", correct)
      print("Predictions of instances: ", predictions)
      
      assert isinstance(indexes, (Indexes, str)), "Bad value in the parameter 'indexes'"
      # starting by get the correct indexes:
      possible_indexes = None

      if isinstance(indexes, str):
        assert model is not None, "For the indexes=str, you need to provide a model (boosted tree, random forest, ...) in the model parameter !"
        id_solver_results = model.forest[0].id_solver_results
        xgboost_tree = self.results[id_solver_results].tree
        results = self.results[id_solver_results]
        
        if os.path.isfile(indexes):
          files_indexes = indexes
        elif os.path.isdir(indexes):
          if backup_id is None:
            found = False
            for filename in os.listdir(indexes): 
              file = os.path.join(indexes, filename)
              if os.path.isfile(file) and file.endswith(".instances"):
                  files_indexes = file
                  assert found is False, "Too many .instances files in the directory: " + indexes + " Please put directly the good file in the option or use the backup_id parameter !"
                  found = True
          else:
            found = False
            for filename in os.listdir(indexes): 
              file = os.path.join(indexes, filename)
              if os.path.isfile(file) and file.endswith("." + str(backup_id) + ".instances"):
                  files_indexes = file
                  found = True
                  break
            assert found is True, "No ." + str(backup_id) + ".instances" + " files in the directory: " + indexes + " Please put directly the good file in the option or use the backup_id parameter !"
                          
        print("instances file:", files_indexes)
        f = open(files_indexes)
        data = json.loads(json.load(f))
        possible_indexes = data['indexes']
        f.close()

      elif indexes == Indexes.Training or indexes == Indexes.Test or indexes == Indexes.Mixed:
        assert model is not None, "For the indexes=Indexes.Training or Indexes.Test parameters, you need to provide a model (boosted tree, random forest, ...) in the model parameter !"
        id_solver_results = model.forest[0].id_solver_results
        xgboost_tree = self.results[id_solver_results].tree
        results = self.results[id_solver_results]
        possible_indexes = results.training_index if indexes == Indexes.Training else results.test_index
        if indexes == Indexes.Mixed and n is not None and len(possible_indexes) < n:
          for i in range(n + 1 - len(possible_indexes)):
            if i < len(results.training_index):      
              possible_indexes = numpy.append(possible_indexes, results.training_index[i])

      
      # load data and get instances
      if self.data is None:
        assert dataset is not None, "Data are not loaded yet. You have to put your dataset filename through the 'dataset' parameter !"
        data, labels = self.load_data_limited(dataset, possible_indexes, n)
      else:
        if possible_indexes is None: 
          data = self.data
          labels = self.labels 
        else:
          data = numpy.array([self.data[x] for x in possible_indexes])
          labels = numpy.array([self.labels[x] for x in possible_indexes])
      
      if possible_indexes is None:
        possible_indexes = [i for i in range(len(data))]
      
      if isinstance(possible_indexes, numpy.ndarray):
        possible_indexes = possible_indexes.tolist()

      # select instance according to parameters
      instances = []
      instances_indexes = []
      for j in range(len(data)):
        current_index = possible_indexes[j]
        #TO DO: where no model, an error occurs here: xgboost_tree not defined :)
        prediction_solver = xgboost_tree.predict(data[j].reshape(1, -1))[0]
        label = labels[j]
        if (correct is True and prediction_solver == label) \
          or (correct is False and prediction_solver != label) \
          or (correct is None):
          if predictions == None or prediction_solver in predictions:
            instances.append((data[j], prediction_solver))
            instances_indexes.append(current_index)
            
        if isinstance(n, int) and len(instances) >= n:
          break
      
      if backup_directory is not None:
        # we want to save the instances indexes in a file
        name = self.datasetname.split(os.sep)[-1].split('.')[0]
        base_directory = backup_directory + os.sep + name + "_model" 
        if not os.path.isdir(base_directory):
          os.mkdir(base_directory)
        if backup_id is None:  
          complete_name = base_directory + os.sep + name + ".instances"
        else:
          complete_name = base_directory + os.sep + name + "." + str(backup_id) + ".instances"
        data = {"dataset": name, 
                "n": len(instances_indexes),
                "indexes": instances_indexes}

        json_string = json.dumps(data)
        with open(complete_name, 'w') as outfile:
          json.dump(json_string, outfile)
        
        print("Indexes of selected instances saved in:", complete_name)
      print("number of instances selected:", len(instances))
      print("----------------------------------------------")
      return instances

    def to_boosted_trees(self):   
      self.id_features = {"f{}".format(i):i for i in range(self.n_features)}
      BTs = [BoostedTrees(self.results_to_trees(id_solver_results), n_classes=self.n_labels) for id_solver_results,_ in enumerate(self.results)]
      return BTs

    def results_to_trees(self, id_solver_results):
      xgb_BT = self.results[id_solver_results].tree.get_booster()
      xgb_JSON = self.xgboost_BT_to_JSON(xgb_BT)
      decision_trees = []
      target_class = 0
      for tree_JSON in xgb_JSON:
        #print(tree_JSON)
        tree_JSON = json.loads(tree_JSON)
        root = self.recuperate_nodes(tree_JSON)
        decision_trees.append(DecisionTree(TypeTree.WEIGHT, self.n_features, root, target_class=target_class, id_solver_results=id_solver_results))
        if self.n_labels > 2: # Special case for a 2-classes prediction ! 
          target_class = target_class + 1 if target_class != self.n_labels - 1 else 0
        
      return decision_trees

    def recuperate_nodes(self, tree_JSON):
      if "children" in tree_JSON:
        assert tree_JSON["split"] in self.id_features, "A feature is not correct during the parsing from xgb_JSON to DT !"
        id_feature = self.id_features[tree_JSON["split"]]
        #print("id_features:", id_feature)
        threshold = tree_JSON["split_condition"]
        probabilities = [0.5,0.5]
        decision_node = DecisionNode(int(id_feature + 1), threshold, probabilities, left=None, right=None)
        id_right = tree_JSON["no"] #It is the inverse here, right for no, left for yes
        for child in tree_JSON["children"]:
          if child["nodeid"] == id_right:
            decision_node.right = LeafNode(float(child["leaf"])) if "leaf" in child else self.recuperate_nodes(child)
          else:
            decision_node.left = LeafNode(float(child["leaf"])) if "leaf" in child else self.recuperate_nodes(child)
        return decision_node
      elif "leaf" in tree_JSON:
        #Special case when the tree is just a leaf, this append when no split is realized by the solver, but the weight have to be take into account
        return LeafNode(float(tree_JSON["leaf"]))

    def xgboost_BT_to_JSON(self, xgboost_BT):
      save_names = xgboost_BT.feature_names
      xgboost_BT.feature_names = None
      xgboost_JSON = xgboost_BT.get_dump(with_stats=True, dump_format="json")
      xgboost_BT.feature_names = save_names
      return xgboost_JSON