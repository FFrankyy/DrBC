#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <set>
#include <memory>
#include "graph.h"
#include <cmath>


class Utils
{
public:
    Utils();

    std::vector<double> Betweenness_Batch(std::vector<std::shared_ptr<Graph>> _g_list);
    std::vector<double> Betweenness(std::shared_ptr<Graph> _g);
    std::vector<int> bc_bool;
    std::vector<double> bc_log;
    double getRobustness(std::shared_ptr<Graph> graph, std::vector<int> solution);
    std::vector<double> convertToLog(std::vector<double> CB);


};

#endif
