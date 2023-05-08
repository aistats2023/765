from PyLearningExplainer import *

stopwatch = Stopwatch()

MLsolver = Xgboost(options.data)

cross_validation_BTs = MLsolver.evaluate(method=EvaluationMethod.LeaveOneGroupOut, output=EvaluationOutput.SaveModel, model_directory="dataset/models")

print("time evaluate:", stopwatch.elapsed_time(reset=True), "seconds")

for id, BTs in enumerate(cross_validation_BTs):

  instances = MLsolver.get_instances(indexes=Indexes.Test, n=10, model=BTs, backup_directory="dataset/models", backup_id=id)

  print("time get instances:", stopwatch.elapsed_time(reset=True), "seconds")
  print("n selected instances:", len(instances))

