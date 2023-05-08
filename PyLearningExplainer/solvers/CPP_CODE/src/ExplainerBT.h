
#ifndef CPP_CODE_EXPLAINERBT_H
#define CPP_CODE_EXPLAINERBT_H

#include <Python.h>
#include <vector>
#include "Tree.h"
#include "utils/TimerHelper.h"

namespace PyLE {
    class ExplainerBT {
      public:
        int n_classes;
        int n_iterations;
        int time_limit; //in seconds
        std::vector<double> weights; //useful only for multiclasses

        ExplainerBT(int _n_classes) : n_classes(_n_classes), n_iterations(50), time_limit(0) {
            if(n_classes > 2)
                for(int i = 0; i < n_classes; i++) weights.push_back(0.0);
        }

        void addTree(PyObject *tree_obj);
        std::vector<Tree*> trees;
        void compute_reason_conditions(std::vector<int> &instance, int prediction, std::vector<int> &reason);
        void compute_reason_features(std::vector<int> &instance, std::vector<int> &features, int prediction, std::vector<int> &reason);
        
        bool is_implicant(std::vector<bool> &instance, std::vector<bool> &active_lits, unsigned int prediction);

        inline void set_n_iterations(int _n_iterations){n_iterations = _n_iterations;}
        inline void set_time_limit(int _time_limit){time_limit = _time_limit;}
    };
}


#endif //CPP_CODE_EXPLAINERBT_H
