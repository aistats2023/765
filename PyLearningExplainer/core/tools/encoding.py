
from os import remove
from collections.abc import Iterable

class DNF(list):pass
class CNF(list):pass

class PrimeImplicantCNF():

  def __init__(self, core):
    self.cnf = [clause for clause in core if len(clause) != 1]
    self.necessary = {clause[0] for clause in core if len(clause) == 1}
    if self.cnf == []: #Special case when all clauses are of size 1.
      return None
    self.mapping_original_to_new = None
    self.mapping_new_to_original = None
    self.n_literals_mapping = 1
    self.n_literals = 0
    self.new_mapping_id_variables()

    self.bijection_clauses = []
    self.compute_bijection()
    self.cnf += self.bijection_clauses
    
  def get_reason_from_model(self, model):
    reason = [l for l in self.necessary] 
    reason += [self.mapping_new_to_original[l] for l in model if l > 0 and abs(l) < self.n_literals_mapping and self.mapping_new_to_original[l] not in reason]
    #TODO check self.mapping_new_to_original[l] not in sufficient is Ok or not ? 
    return reason

  def get_blocking_clause(self, model):
    return [-l for l in model if l > 0 and abs(l) < self.n_literals_mapping]
            
  def compute_bijection(self):
    occurrences = [[] for i in range(self.n_literals_mapping)]
    for i, clause in enumerate(self.cnf):
      for l in clause:
        occurrences[l].append(i)

    self.n_literals = self.n_literals_mapping
    for l in range(1, self.n_literals_mapping):
        new_clause1 = [-l]
        for j in occurrences[l]:
            clause = self.cnf[j]
            if len(clause) == 2:
                new_clause1.append(-clause[0] if clause[1] == l else -clause[1])
                continue
            new_clause1.append(self.n_literals)
            # nblit = -l1 and -l2 ... Add to bijection (tseitsin
            new_clause2 = [self.n_literals]
            for l2 in clause:
                if l2 == l:
                    continue
                self.bijection_clauses.append([-self.n_literals, -l2])
                new_clause2.append(l2)
            self.bijection_clauses.append(new_clause2)
            self.n_literals = self.n_literals + 1
        self.bijection_clauses.append(new_clause1)
    self.bijection_clauses = CNFencoding.format(self.bijection_clauses)

  def new_mapping_id_variables(self):
    """
    Create new id variables for the cnf in the form of list. 
    Allow to clean the current missing id variables.
    Warning: do not take into account the signs of literals
    Example cnf = [[-1, 4, 8], [1, 10]] -> [[1, 2, 3], [1, 4]] 
    Result list of the example: [0,1,0,0,2,0,0,0,3,0,4]
    """
    size = CNFencoding.compute_max_id_variable(self.cnf)
    mapping_original_to_new = [0 for i in range(size + 1)]
    mapping_new_to_original = [0]
    for clause in self.cnf:
      for l in clause:
        if mapping_original_to_new[abs(l)] != 0:
          continue
        mapping_original_to_new[abs(l)] = self.n_literals_mapping
        mapping_new_to_original.append(l)
        self.n_literals_mapping += 1
    self.mapping_original_to_new = mapping_original_to_new
    self.mapping_new_to_original = mapping_new_to_original
    

    # Apply the mapping
    for clause in self.cnf:
      for i, l in enumerate(clause):
        clause[i] = self.mapping_original_to_new[abs(l)]
    self.cnf = CNFencoding.format(self.cnf)

class CNFencoding():

  @staticmethod
  def compute_set_variables(formula):
    set_variables = set()
    for cube in formula:
      set_variables.update(abs(lit) for lit in cube)
    return set_variables
  
  @staticmethod
  def compute_n_variables(formula):
    return len(CNFencoding.compute_set_variables(formula))

  @staticmethod
  def compute_max_id_variable(formula):
    return max(abs(l) for c in formula for l in c)

  @staticmethod
  def tseitin(dnf):
    n_variables = CNFencoding.compute_n_variables(dnf)
    id_aux_var = n_variables + 1 # for new auxiliary variables
    clauses = CNF()
    
    for cube in dnf:
      clauses.append([-1 * cube[j] for j in range(len(cube))]+[id_aux_var])
      
      for lit in cube:
        clauses.append([lit,-1 * id_aux_var])
      id_aux_var+=1
    
    clauses.append([i for i in range(n_variables+1,n_variables+len(dnf)+1)])
    return CNF(clauses)

  @staticmethod
  def complementary(dnf):
    return CNF([[-1 * lit for lit in cube] for cube in dnf])
  
  @staticmethod
  def literals_in_implicant(cnf, implicant):
    return CNF([[l for l in clause if l in implicant] for clause in cnf])
  
  @staticmethod
  def remove_subsumed(cnf):
    cnf = sorted(cnf, key=lambda clause: len(clause))
    subsumed = [False for _ in range(len(cnf) + 1)]
    flags = [False for _ in range(CNFencoding.compute_max_id_variable(cnf) + 1)]
    for i, clause in enumerate(cnf):
      if subsumed[i] is True: continue
      for l in clause: flags[abs(l)] = True
      for j in range(i + 1, len(cnf)):
        nLiteralsInside = tuple(flags[abs(l)] == True for l in cnf[j]).count(True)
        if nLiteralsInside == len(clause): 
          subsumed[j] = True
      for l in clause: flags[abs(l)] = False
    return CNF([clause for i, clause in enumerate(cnf) if subsumed[i] is False])

  @staticmethod
  def format(cnf):
    if len(cnf) == 0:
      return tuple()
    if not isinstance(cnf[0], Iterable):
      return tuple(sorted(cnf, key=lambda l: abs(l)))
    return tuple(tuple(sorted(clause, key=lambda l: abs(l))) for clause in cnf)

  @staticmethod
  def extract_core(cnf, implicant):
    """
    A core is the CNF with only the literals in the CNF.
    """
    core = CNFencoding.literals_in_implicant(cnf, implicant)
    return CNFencoding.remove_subsumed(core)


  @staticmethod
  def apply_mapping_id_variables(cnf, mapping_id_variables):
    """
    Apply a mapping from new_mapping_id_variables(cnf)
    Warning: therefore remove the original signs of literals 
    """
    return CNF([[mapping_id_variables[abs(l)] for l in clause] for clause in cnf]) 

        
  @staticmethod
  def to_prime_implicant_CNF(cnf, implicant):
    """
    Generate formula used to enumerate all prime implicants.
    Based on Saïd Jabbour, João Marques-Silva, Lakhdar Sais, Yakoub Salhi:
    Enumerating Prime Implicants of Propositional Formulae in Conjunctive Normal Form. JELIA 2014: 152-165
    """
    
    core = CNFencoding.extract_core(cnf, implicant)  
    return PrimeImplicantCNF(core)
