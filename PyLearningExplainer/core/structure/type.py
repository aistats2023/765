from enum import Enum, unique

def auto(n_occurrences=1):
    def _auto():  # To be replaced by auto() in python 3.6 ?
        if not hasattr(auto, "cnt"):
            auto.cnt = 0
        auto.cnt += 1
        return auto.cnt
    return _auto() if n_occurrences == 1 else (_auto() for _ in range(n_occurrences))

@unique
class MethodCNF(Enum):
    TSEITIN, COMPLEMENTARY = auto(2)

    def __str__(self):
        return self.name


@unique
class TypeLeaf(Enum):
    LEFT, RIGHT = auto(2)
    
    def __str__(self):
        return self.name

@unique
class TypeTree(Enum):
    PREDICTION, WEIGHT = auto(2)
    
    def __str__(self):
        return self.name

@unique
class TypeReason(Enum):
    Direct, Sufficient, MinimalSufficient, Preferred, Contrastive, All = auto(6)
    
    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return self.name

@unique
class TypeCount(Enum):
    NSufficientReasons, NSufficientReasonsPerAttribute = auto(2)
    
    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return self.name


@unique
class EvaluationMethod(Enum):
    LoadModel, HoldOut, LeaveOneGroupOut = auto(3)
    
    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return self.name


@unique
class EvaluationOutput(Enum):
    DecisionTree, RandomForest, BoostedTrees, SaveModel = auto(4)
    
    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return self.name

class Indexes(Enum):
    Training, Test, Mixed, All = auto(4)
    
    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return self.name

@unique
class ReasonExpressivity(Enum):
    Features, Conditions = auto(2)
    
    def __eq__(self, other):
      if type(self) != type(other):
        raise TypeError("Bad type for : " + other + " (" + str(type(other)) + "). Must be of the type " + str(type(self)) + ".")
      return self.value == other.value

    def __int__(self):
        if self == ReasonExpressivity.Features:
          return 1
        elif self == ReasonExpressivity.Conditions:
          return 0
    def __str__(self):
        return self.name
