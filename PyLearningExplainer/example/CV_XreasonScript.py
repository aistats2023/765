from PyLearningExplainer import *
import subprocess
import os
import sys
import json
from threading import Timer

def execute(command, time):
  p = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, preexec_fn=os.setsid)
  
  my_timer = Timer(time, lambda process: process.kill(), [p])
  
  results = None
  try:
    my_timer.start()
    for line in p.stdout:
      sys.stdout.write(line)
      if "solution" in line:
        results = line

  finally:
    my_timer.cancel()
  p.wait()
  p.terminate()
  if results is not None:
    results = results.split(":")[1]
    results = results.replace("[", "").replace("]","").split(",")
    results = [int(element) for element in results]
  return results

stopwatch = Stopwatch()
stopwatchtotal = Stopwatch()

MLsolver = Xgboost()

CV_BTs = MLsolver.evaluate(method=EvaluationMethod.LoadModel, output=EvaluationOutput.BoostedTrees, model_directory=options.model)

print("time evaluate:", stopwatch.elapsed_time(reset=True), "seconds")
print("time_limit: ", options.timelimit)

for id, BTs in enumerate(CV_BTs):
  instances = MLsolver.get_instances(dataset=options.data, indexes=options.model, model=BTs, backup_id=id)
  print("time get instances:", stopwatch.elapsed_time(reset=True), "seconds")

  explainer = ExplainerBT(BTs)

  id_instance = 0
  for instance, prediction in instances:
    explainer.set_instance(instance)
    print("Instance " + str(id_instance) + " of the " + str(id) + " CV in progress ...")
    print("len(instance):", len(instance))
    stopwatch.elapsed_time(reset=True)

    # xreason phase:
    program = os.getcwd() + "/../xreason/src/xreason.py"
    #program = os.getcwd() + "/xreason/src/xreason.py" (for the cluster)
    
    filename_instance = options.data.split("/")[-1].split(".")[0]+"_"+str(id)+"_"+str(id_instance) + ".instance"
    data_instance = instance.tolist()
    data_instance = {"data_instance": data_instance}
    json_string = json.dumps(data_instance)
    with open(filename_instance, 'w') as outfile:
      json.dump(json_string, outfile)

    instance_str = os.getcwd() + "/" + filename_instance
    print("instance str:", instance_str)
    filename_model = options.model + "/" + options.data.split("/")[-1].split(".")[0]+"."+str(id) + ".model"

    command = "python3.8 -u " + program + " -X abd -R lin -e mx -s g3 --instance-file " + instance_str + " -vvv --from-py-learning-explainer " + filename_model
    
    print("command:", command)
    results = execute(command, int(options.timelimit))
    print("results:", results)
    if results is None:
      reduction_instance = 0
      results = []
    else:
      reduction_instance = round(float(100-(len(results)*100)/len(instance)),2) 
    time = float(stopwatch.elapsed_time(reset=True))
    
    print("time:", str(time), "seconds")
    print("length instance:", len(instance))
    print("length reason:", len(results))
    print("number of features involved by the reason:", len(results))
    print("number of features not involved by the reason:", len(instance)-len(results))
    print("percentage of reduction of the instance:", reduction_instance, "%")
    print("total time:", stopwatchtotal.elapsed_time(), "seconds")
    
    id_instance += 1
    print()
  

