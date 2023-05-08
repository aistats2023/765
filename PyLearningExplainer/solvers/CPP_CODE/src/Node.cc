
#include "Node.h"

double PyLE::Node::compute_weight(std::vector<bool> &instance, std::vector<bool> &active_lits, bool get_min) {
    if (is_leaf())
        return weight;
    if (active_lits[lit]) { // Literal in implicant
        if (instance[lit]) // positive lit in instance
            return true_branch->compute_weight(instance, active_lits, get_min);
        else
            return false_branch->compute_weight(instance, active_lits, get_min);
    }

    // Literal not in implicant
    double wf = false_branch->compute_weight(instance, active_lits, get_min);
    // Do not traverse right branch // TODO CHeck
    //if(get_min && wf < true_min) return wf;
    //if(!get_min && wf > true_max) return wf;
    double wt = true_branch->compute_weight(instance, active_lits, get_min);
    if (get_min)
        return std::min(wf, wt);
    return std::max(wf, wt);
}


void PyLE::Node::reduce_with_instance(std::vector<bool> &instance, bool get_min) {
    if(is_leaf()) return; // Nothing to do

    false_branch->reduce_with_instance(instance, get_min);
    true_branch->reduce_with_instance(instance, get_min);
    if(false_branch->is_leaf() && true_branch->is_leaf()) { // V1
        // TODO : si lit de base n'est pas dans l'instance ?????
        double instance_w = instance[lit] ? true_branch->weight : false_branch->weight;
        double not_instance_w = instance[lit] ? false_branch->weight : true_branch->weight;
        if((get_min && instance_w < not_instance_w) || (!get_min && instance_w > not_instance_w)) {
            artificial_leaf = true;
            weight = instance_w;
        }
    }
}

double PyLE::Node::extremum_true_branch(bool get_min) {
    if(is_leaf()) return weight;

    double wf = false_branch->extremum_true_branch(get_min);
    double tf = true_branch->extremum_true_branch(get_min);
    if(get_min)
        true_min = tf;
    else
        true_max = tf;
    return get_min ? std::min(wf, tf) : std::max(wf, tf);
}

int PyLE::Node::nb_nodes() {
    if(is_leaf()) return 1;
    return 1 + true_branch->nb_nodes() + false_branch->nb_nodes();
}