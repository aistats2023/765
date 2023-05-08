
#ifndef CPP_CODE_TREE_H
#define CPP_CODE_TREE_H

#include "Node.h"
#include <Python.h>

namespace PyLE {

    class Tree {
    public :
        unsigned int target_class;
        u_char *memory = nullptr;
        Node *root = nullptr;
        std::vector<Node *> all_nodes;
        Tree(PyObject *tree_obj){
          root = parse(tree_obj);
        }

        void display() { root->display(); std::cout << std::endl;}
        ~Tree();
        Node* parse(PyObject *tree_obj);
        Node* parse_recurrence(PyObject *tree_obj);
        double compute_weight(std::vector<bool> &instance, std::vector<bool> &active_lits, bool get_min);
        void initialize(std::vector<bool> &instance, bool get_min);
        int nb_nodes() { return root->nb_nodes();}
    };

}
#endif //CPP_CODE_TREE_H
