
#include "ExplainerBT.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <map>

static bool abs_compare(int a, int b) {return (std::abs(a) < std::abs(b));}

void PyLE::ExplainerBT::addTree(PyObject *tree_obj) {
  //std::cout << "add_tree2" << std::endl;
    
  Tree* tree = new Tree(tree_obj);
  trees.push_back(tree);
}

void PyLE::ExplainerBT::compute_reason_features(std::vector<int> &instance, std::vector<int> &features, int prediction, std::vector<int> &reason) {
    int max  = abs(*std::max_element(instance.begin(), instance.end(), abs_compare));
    int n_current_iterations = 0;
    PyLE::TimerHelper::initializeTime();
    std::vector<bool> polarity_instance(max + 1, true);
    std::vector<bool> active_lits (max + 1, false);
    
    std::set<int> possible_features(features.begin(), features.end());
    std::vector<int> order(possible_features.begin(), possible_features.end());
    std::map<int, std::vector<int>> features_to_lits;
    
    for(auto it_order = order.begin(); it_order != order.end(); it_order++){
      std::vector<int> elements;
      int feature = *it_order;
      for(unsigned int i = 0; i < instance.size(); i++){
        if (features[i] == feature) elements.push_back(instance[i]);
      }
      features_to_lits[feature] = elements;;
    }

    unsigned int best_size = order.size() + 1;
    unsigned int current_size = order.size();

    for (auto l: instance)
        polarity_instance[std::abs(l)] = l > 0;

    // Before computing a reason, reduce the size of the tree wrt considered instance
    for(Tree *tree : trees)
        tree->initialize(polarity_instance, (n_classes == 2 ? prediction == 1 : tree->target_class == prediction));

    while(true){
        std::random_shuffle(order.begin(), order.end());
        for(auto l : instance) active_lits[abs(l)] = true; // Init
        current_size = order.size();

        for(int feature: order) {
            std::vector<int>& lits = features_to_lits[feature]; 
            for(auto it_lits = lits.begin(); it_lits != lits.end(); it_lits++){
              active_lits[abs(*it_lits)] = false;
            }
            if(is_implicant(polarity_instance, active_lits, prediction) == false){
              // not a implicant
              for(auto it_lits = lits.begin(); it_lits != lits.end(); it_lits++){
                active_lits[abs(*it_lits)] = true;
              }
            }else{
              // alway a implicant
              current_size--;
            }
        }
        
        if(current_size < best_size) {
          //We are find a better reason :)
          best_size = current_size;
          //Save this new reason
          reason.clear();
          for(auto l : instance)
            if(active_lits[abs(l)])
              reason.push_back(l);
        }
        n_current_iterations++;
        
        if ((time_limit != 0 && PyLE::TimerHelper::realTime() > time_limit)
            || (time_limit == 0 && n_current_iterations > n_iterations)) 
            return;
    }
}

void PyLE::ExplainerBT::compute_reason_conditions(std::vector<int> &instance, int prediction, std::vector<int> &reason) {
    int max  = abs(*std::max_element(instance.begin(), instance.end(), abs_compare));
    int n_current_iterations = 0;
    PyLE::TimerHelper::initializeTime();
    std::vector<bool> polarity_instance(max + 1, true);
    std::vector<bool> active_lits (max + 1, false);
    std::vector<int> order(instance);

     
    unsigned int best_size = instance.size() + 1, current_size = instance.size();
    for (auto l: instance)
        polarity_instance[std::abs(l)] = l > 0;

    int nb = 0;
    for(Tree *tree: trees) nb+=tree->nb_nodes();

    // Before computing a reason, reduce the size of the tree wrt considered instance
    for(Tree *tree : trees)
        tree->initialize(polarity_instance, (n_classes == 2 ? prediction == 1 : tree->target_class == prediction));
    int nb2 = 0;
        for(Tree *tree: trees) nb2+=tree->nb_nodes();

    std::cout << "before: " << nb << " " << "after "<< nb2 << std::endl;
    while(true){
        std::random_shuffle(order.begin(), order.end());
        for(auto l : instance) active_lits[abs(l)] = true; // Init
        current_size = instance.size();

        for(int l: order) {
            active_lits[abs(l)] = false;
            if(is_implicant(polarity_instance, active_lits, prediction) == false)
                active_lits[abs(l)] = true;
            else
                current_size--;
        }

        if(current_size < best_size) {
          //std::cout << "current_size:" << current_size << std::endl;
            best_size = current_size;
            reason.clear();
            for(auto l : instance)
                if(active_lits[abs(l)])
                    reason.push_back(l);
        }
        n_current_iterations++;
        
        if ((time_limit != 0 && PyLE::TimerHelper::realTime() > time_limit)
            || (time_limit == 0 && n_current_iterations > n_iterations)) 
            return;
    }
}

bool PyLE::ExplainerBT::is_implicant(std::vector<bool> &instance, std::vector<bool> &active_lits, unsigned int prediction) {
    if(n_classes == 2) {
        double weight = 0;
        for(Tree *tree : trees)
            weight += tree->compute_weight(instance, active_lits, prediction == 1);

        return prediction == (weight > 0);
    }

    // Multi classes case
    std::fill(weights.begin(), weights.end(), 0.0);
    std::vector<double> weights(n_classes, 0.0);
    for(Tree *tree : trees)
        weights[tree->target_class] += tree->compute_weight(instance, active_lits, tree->target_class == prediction);

    double target = weights[prediction];
    for(unsigned int i = 0; i < weights.size(); i++) {
        if(i != prediction && target < weights[i])
            return false;
    }
    return true;
}



















