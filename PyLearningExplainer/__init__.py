import sys

from PyLearningExplainer.core.tools.option import options

from PyLearningExplainer.solvers.ML.scikitlearn import Scikitlearn
from PyLearningExplainer.solvers.ML.xgboost import Xgboost

from PyLearningExplainer.core.tools.utils import display_observation, Stopwatch

from PyLearningExplainer.core.explainer.explainerBT import ExplainerBT

from PyLearningExplainer.core.structure.type import TypeReason, TypeCount, TypeTree, EvaluationMethod, EvaluationOutput, Indexes, ReasonExpressivity
from PyLearningExplainer.core.tools.heatmap import HeatMap
from PyLearningExplainer.core.structure.decisionTree import DecisionTree, DecisionNode
from PyLearningExplainer.core.structure.randomForest import RandomForest

from PyLearningExplainer.core.structure.boostedTrees import BoostedTrees


DIRECT = TypeReason.Direct
SUFFICIENT = TypeReason.Sufficient
MINIMAL_SUFFICIENT = TypeReason.MinimalSufficient
ALL = TypeReason.All

options.set_values("data", "model", "instances", "niterations", "timelimit")
options.parse(sys.argv[1:])