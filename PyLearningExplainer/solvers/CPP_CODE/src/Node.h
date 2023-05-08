
#ifndef CPP_CODE_NODE_H
#define CPP_CODE_NODE_H

#include <iostream>
#include<vector>

namespace PyLE {
    class Node {
    public:
        int lit;
        double weight;
        Node *false_branch, *true_branch;
        double true_min, true_max; // The min and max possible values for true branch
        bool artificial_leaf;

        Node(double w) : lit(0), weight(w), false_branch(nullptr), true_branch(nullptr), artificial_leaf(false) {}
        Node(int l, Node *f, Node *t) : lit(l), false_branch(f), true_branch(t), artificial_leaf(false) {}


        bool is_leaf() { return artificial_leaf || (false_branch == nullptr && true_branch == nullptr);}
        void display() {
            if (is_leaf()){
              std::cout << "[" << weight << "]";
            }else{
              std::cout << "[" << lit << ",";
              false_branch->display(); std::cout << "\n";
              true_branch->display(); std::cout << "\n";
              std::cout << "]";
            }
        }

        double compute_weight(std::vector<bool> &instance, std::vector<bool> &active_lits, bool get_min);
        void reduce_with_instance(std::vector<bool> &instance, bool get_min);
        double extremum_true_branch(bool get_min); // The extremum value (min or max) of the true branch
        int nb_nodes();
    };
}

#endif //CPP_CODE_NODE_H
